import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.common.logger_utils import logger
from src.models.multitask_cls import BaseMultitaskModel


class MultitaskIFSFuzzyQueryBasedModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 do_blindly_decode: bool = True, lambda_other: float = 1.0):
        super(MultitaskIFSFuzzyQueryBasedModel, self).__init__()
        self.bert = bert
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_to_idx = {task.value: idx for idx, task in enumerate(self.tasks)}
        self.idx_to_task = {idx: task.value for idx, task in enumerate(self.tasks)}

        # Query vectors for fuzzy relation.
        self.query_vectors = nn.Parameter(torch.randn(self.num_tasks, self.hidden_size))
        nn.init.xavier_uniform_(self.query_vectors)

        # Learnable weights for fuzzy relation: w_ik.
        self.relation_weights = nn.Parameter(torch.ones(self.num_tasks, self.hidden_size))

        # Learnable sigma for Gaussian in fuzzy relation.
        self.relation_sigma = nn.Parameter(torch.ones(self.num_tasks))

        # Fuzzy set centers for IFS (mu: positive, pi: unknown/hesitation, nu: negative).
        # Using Gaussian membership functions.
        self.fuzzy_centers = nn.Parameter(torch.tensor([[-1.0, 0.0, 1.0] for _ in range(self.num_tasks)]))
        self.fuzzy_std = nn.Parameter(torch.ones(self.num_tasks))

        self.do_blindly_decode = do_blindly_decode
        self.lambda_other = lambda_other

        logger.info(f"Task to Index: {self.task_to_idx}")
        logger.info(f"DO BLINDLY DECODE: {do_blindly_decode}")
        logger.info(f"Lambda for other tasks: {lambda_other}")

    def gaussian_membership(self, input_value, center, std):
        return torch.exp(-0.5 * ((input_value - center) / std) ** 2)

    def s_norm_prob(self, value_a, value_b):
        return value_a + value_b - value_a * value_b

    def s_norm_normalize(self, mu_raw, pi_raw, nu_raw):
        # Step 1: S1 = pi + nu - pi*nu.
        s_norm_pi_nu = self.s_norm_prob(pi_raw, nu_raw)
        # Step 2: Z = mu + S1 - mu*S1.
        normalization_factor = self.s_norm_prob(mu_raw, s_norm_pi_nu)

        # Normalize each membership by Z.
        mu_normalized = mu_raw / (normalization_factor + 1e-8)
        pi_normalized = pi_raw / (normalization_factor + 1e-8)
        nu_normalized = nu_raw / (normalization_factor + 1e-8)

        return mu_normalized, pi_normalized, nu_normalized

    def compute_fuzzy_relation(self, embeddings):
        batch_size = embeddings.shape[0]

        # Expand dimensions for broadcasting.
        embeddings_expanded = embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size].
        query_expanded = self.query_vectors.unsqueeze(0)  # [1, num_tasks, hidden_size].
        weights_expanded = self.relation_weights.unsqueeze(0)  # [1, num_tasks, hidden_size].
        sigma_expanded = self.relation_sigma.unsqueeze(0).unsqueeze(-1)  # [1, num_tasks, 1].

        # Compute Gaussian: exp(-(Q_ik - e_k)^2 / 2sigma^2).
        diff_squared = (query_expanded - embeddings_expanded) ** 2  # [batch_size, num_tasks, hidden_size].
        gaussian_values = torch.exp(-diff_squared / (2 * sigma_expanded ** 2 + 1e-8))

        # Apply weights: w_ik * gaussian.
        weighted_relation = weights_expanded * gaussian_values  # [batch_size, num_tasks, hidden_size].

        # Max pooling over hidden dimension: s_i = max_k [R_i(Q_ik, e_k)].
        scores, _ = torch.max(weighted_relation, dim=-1)  # [batch_size, num_tasks].

        return scores

    def lukasiewicz_t_norm(self, value_a, value_b):
        return torch.clamp(value_a + value_b - 1, min=0)

    def lukasiewicz_s_norm(self, value_a, value_b):
        return torch.clamp(value_a + value_b, max=1)

    def compute_satisfaction(self, label, mu_value, nu_value):
        t_norm_positive = self.lukasiewicz_t_norm(label, mu_value)
        t_norm_negative = self.lukasiewicz_t_norm(1 - label, nu_value)
        satisfaction = self.lukasiewicz_s_norm(t_norm_positive, t_norm_negative)
        return satisfaction

    def compute_uncertainty_loss(self, mu_value, nu_value, pi_value):
        epsilon = 1e-8
        log_mu = torch.log(mu_value + epsilon)
        log_nu = torch.log(nu_value + epsilon)
        log_pi = torch.log(pi_value + epsilon)

        # Using T-norm with log values (scaled to [0,1] range approximately).
        # Since log values are negative, we use absolute values for T-norm.
        uncertainty = (
                -self.lukasiewicz_t_norm(mu_value, torch.abs(log_mu) / 10)
                - self.lukasiewicz_t_norm(nu_value, torch.abs(log_nu) / 10)
                - self.lukasiewicz_t_norm(pi_value, torch.abs(log_pi) / 10)
        )
        return uncertainty

    def compute_ifs_loss(self, mu_memberships, pi_memberships, nu_memberships,
                         task_labels, correct_task_indices):
        batch_size = mu_memberships.shape[0]
        total_loss = 0.0

        for batch_idx in range(batch_size):
            correct_task = correct_task_indices[batch_idx]
            label = task_labels[batch_idx].float()

            # Loss for correct task: L_task = 1 - Sat(y, mu, nu).
            satisfaction = self.compute_satisfaction(
                label,
                mu_memberships[batch_idx, correct_task],
                nu_memberships[batch_idx, correct_task]
            )
            task_loss = 1 - satisfaction

            # Loss for other tasks: maximize uncertainty/hesitation.
            other_tasks_loss = 0.0
            for task_idx in range(self.num_tasks):
                if task_idx != correct_task:
                    # Encourage high hesitation (pi) for unrelated tasks.
                    # Loss = 1 - pi (minimize when pi is high).
                    other_tasks_loss += (1 - pi_memberships[batch_idx, task_idx])

            if self.num_tasks > 1:
                other_tasks_loss /= (self.num_tasks - 1)

            total_loss += task_loss + self.lambda_other * other_tasks_loss

        return total_loss / batch_size

    def defuzzy_ifs(self, mu_memberships, pi_memberships, nu_memberships, correct_task_indices=None):
        batch_size = mu_memberships.shape[0]
        predictions = torch.zeros(batch_size, 2, dtype=torch.float, device=mu_memberships.device)

        # Compute support values.
        support_positive = mu_memberships * (1 - pi_memberships)  # [batch_size, num_tasks].
        support_negative = nu_memberships * (1 - pi_memberships)  # [batch_size, num_tasks].
        confidence = (1 - pi_memberships) * torch.max(mu_memberships, nu_memberships)  # [batch_size, num_tasks].

        for batch_idx in range(batch_size):
            if correct_task_indices is not None:
                # Use known task.
                task_idx = correct_task_indices[batch_idx]
                if support_positive[batch_idx, task_idx] > support_negative[batch_idx, task_idx]:
                    predictions[batch_idx] = torch.tensor([0, 1], dtype=torch.float)  # Positive.
                else:
                    predictions[batch_idx] = torch.tensor([1, 0], dtype=torch.float)  # Negative.
            else:
                # Blind decoding: find task with highest confidence.
                max_confidence_idx = torch.argmax(confidence[batch_idx]).item()
                if support_positive[batch_idx, max_confidence_idx] > support_negative[batch_idx, max_confidence_idx]:
                    predictions[batch_idx] = torch.tensor([0, 1], dtype=torch.float)  # Positive.
                else:
                    predictions[batch_idx] = torch.tensor([1, 0], dtype=torch.float)  # Negative.

        return predictions

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value for key, value in batch.items()
                       if key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}

        bert_outputs = self.bert(**input_batch)
        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        # Compute fuzzy relation scores.
        task_logits = self.compute_fuzzy_relation(pooled_outputs)  # [batch_size, num_tasks].

        # Compute raw memberships using Gaussian.
        task_logits_expanded = task_logits.unsqueeze(-1)  # [batch_size, num_tasks, 1].
        centers_expanded = self.fuzzy_centers.unsqueeze(0)  # [1, num_tasks, 3].
        std_expanded = self.fuzzy_std.unsqueeze(0).unsqueeze(-1)  # [1, num_tasks, 1].

        # Raw memberships: [batch_size, num_tasks, 3] -> (positive, unknown, negative).
        raw_memberships = self.gaussian_membership(task_logits_expanded, centers_expanded, std_expanded)

        mu_raw = raw_memberships[:, :, 0]  # Positive membership.
        pi_raw = raw_memberships[:, :, 1]  # Unknown/hesitation membership.
        nu_raw = raw_memberships[:, :, 2]  # Negative membership.

        # Apply S-norm normalization.
        mu_normalized, pi_normalized, nu_normalized = self.s_norm_normalize(mu_raw, pi_raw, nu_raw)

        # Prepare labels and task indices.
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        task_names = batch[TASK_NAME]
        task_indices = [self.task_to_idx[task_name] for task_name in task_names]

        # Compute IFS loss.
        loss = self.compute_ifs_loss(mu_normalized, pi_normalized, nu_normalized, labels, task_indices)

        # Perform defuzzification.
        predictions = self.defuzzy_ifs(
            mu_normalized, pi_normalized, nu_normalized,
            task_indices if not self.do_blindly_decode else None
        )

        # Log evaluation data.
        for idx in range(len(batch[TASK_NAME])):
            if not self.training:
                self.all_task_names.append(batch[TASK_NAME][idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())

                # Store IFS memberships.
                membership_data = {
                    "mu": mu_normalized[idx].detach().cpu().tolist(),
                    "pi": pi_normalized[idx].detach().cpu().tolist(),
                    "nu": nu_normalized[idx].detach().cpu().tolist(),
                    "support_pos": (mu_normalized[idx] * (1 - pi_normalized[idx])).detach().cpu().tolist(),
                    "support_neg": (nu_normalized[idx] * (1 - pi_normalized[idx])).detach().cpu().tolist(),
                }

                # Determine predicted task and label.
                confidence = (1 - pi_normalized[idx]) * torch.max(mu_normalized[idx], nu_normalized[idx])
                predicted_task_idx = torch.argmax(confidence).item()
                predicted_task = self.idx_to_task[predicted_task_idx]

                support_pos = mu_normalized[idx, predicted_task_idx] * (1 - pi_normalized[idx, predicted_task_idx])
                support_neg = nu_normalized[idx, predicted_task_idx] * (1 - pi_normalized[idx, predicted_task_idx])
                predicted_label = 1 if support_pos > support_neg else 0

                membership_data["predicted_task"] = predicted_task
                membership_data["predicted_label"] = predicted_label

                self.all_membership_scores.append(membership_data)

        return loss, predictions

    def log_eval_stats(self, verbose=True):
        memberships = self.all_membership_scores
        labels = self.all_evaluated_labels
        task_names = self.all_task_names

        if not memberships or not labels or not task_names:
            if verbose:
                print("No evaluation logs to analyze.")
            return

        total_samples = len(memberships)
        task_correct = 0
        label_correct = 0
        both_correct = 0
        incorrect_task_samples = 0
        label_correct_when_task_wrong = 0

        for sample_idx in range(total_samples):
            pred_data = memberships[sample_idx]
            true_task = task_names[sample_idx]
            true_label = labels[sample_idx]

            predicted_task = pred_data.get("predicted_task")
            predicted_label = pred_data.get("predicted_label")

            task_is_correct = (predicted_task == true_task)
            label_is_correct = (predicted_label == true_label)

            if task_is_correct:
                task_correct += 1
            if label_is_correct:
                label_correct += 1
            if task_is_correct and label_is_correct:
                both_correct += 1

            if not task_is_correct:
                incorrect_task_samples += 1
                if label_is_correct:
                    label_correct_when_task_wrong += 1

        if verbose:
            print("\n--- [IFS Model] Performance Report ---")
            print(f"1. Task Accuracy: {task_correct / total_samples * 100:.2f}% ({task_correct}/{total_samples})")
            print(f"2. Label Accuracy: {label_correct / total_samples * 100:.2f}% ({label_correct}/{total_samples})")
            print(f"3. Combined Accuracy: {both_correct / total_samples * 100:.2f}% ({both_correct}/{total_samples})")

            print(f"4. Label Accuracy (when task was INCORRECTLY identified):")
            if incorrect_task_samples == 0:
                print("   No incorrect task predictions were made.")
            else:
                accuracy = (label_correct_when_task_wrong / incorrect_task_samples) * 100
                print(f"   Accuracy: {accuracy:.2f}% ({label_correct_when_task_wrong}/{incorrect_task_samples})")

            # Additional IFS-specific stats.
            avg_hesitation = np.mean([np.mean(membership["pi"]) for membership in memberships])
            print(f"5. Average Hesitation (pi): {avg_hesitation:.4f}")
