import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import BertModel

from src.common.consts import BERT_SINGLE_TASK_DIR, IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.common.logger_utils import logger
from src.models.membership_functions import MEMBERSHIP_FUNCTIONS
from src.models.utils import load_pretrained_classifier


class FuzzySet(Enum):
    POSITIVE = 0
    UNKNOWN = 1
    NEGATIVE = 2


class BaseMultitaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.all_evaluated_labels: List[int] = []
        self.all_task_names: List[str] = []
        self.all_membership_scores: List[Dict[str, Any]] = []

    def save_eval_logs(self, out_dir: str, prefix: str = "eval", epoch: Optional[int] = None):
        os.makedirs(out_dir, exist_ok=True)
        epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
        task_names_path = os.path.join(out_dir, f"{prefix}_task_names{epoch_str}.txt")
        labels_path = os.path.join(out_dir, f"{prefix}_labels{epoch_str}.csv")
        memberships_path = os.path.join(out_dir, f"{prefix}_memberships{epoch_str}.json")

        # On vast.ai overlay-fs, small writes can be lost without explicit fsync
        # under sustained pressure. Flush + fsync each file to guarantee persistence.
        for path, payload_writer in [
            (task_names_path, lambda f: f.writelines(f"{n}\n" for n in self.all_task_names)),
            (labels_path, lambda f: f.writelines(f"{l}\n" for l in self.all_evaluated_labels)),
            (memberships_path, lambda f: json.dump(self.all_membership_scores, f, indent=2)),
        ]:
            with open(path, "w", encoding="utf-8") as f:
                payload_writer(f)
                f.flush()
                os.fsync(f.fileno())

    def log_eval_stats(self, verbose=True):
        predictions = self.all_membership_scores
        true_labels = self.all_evaluated_labels
        true_tasks = self.all_task_names

        if not predictions or not true_labels or not true_tasks:
            if verbose:
                print("No evaluation logs to analyze.")
            return

        total_samples = len(predictions)
        if total_samples == 0:
            if verbose:
                print("No samples to analyze.")
            return

        total_task_correct = 0
        total_label_correct = 0
        total_both_correct = 0
        total_incorrect_task_samples = 0
        label_correct_when_task_was_wrong = 0

        for i in range(total_samples):
            pred_data = predictions[i]
            true_task = true_tasks[i]
            true_label = true_labels[i]

            # Handle different possible keys for the predicted task.
            predicted_task = pred_data.get("predicted_task") or pred_data.get("max_task_head")
            if predicted_task is None:
                # Skip sample if no task key found.
                continue

            predicted_label = pred_data["predicted_label"]

            task_is_correct = (predicted_task == true_task)
            label_is_correct = (predicted_label == true_label)

            if task_is_correct:
                total_task_correct += 1
            if label_is_correct:
                total_label_correct += 1
            if task_is_correct and label_is_correct:
                total_both_correct += 1

            if not task_is_correct:
                total_incorrect_task_samples += 1
                if label_is_correct:
                    label_correct_when_task_was_wrong += 1

        if verbose:
            print("\n--- [Baseline Model] Performance Report ---")
            task_accuracy = (total_task_correct / total_samples) * 100
            print(f"1. Task Accuracy: {task_accuracy:.2f}% ({total_task_correct}/{total_samples})")

            label_accuracy = (total_label_correct / total_samples) * 100
            print(f"2. Label Accuracy (Overall): {label_accuracy:.2f}% ({total_label_correct}/{total_samples})")

            combined_accuracy = (total_both_correct / total_samples) * 100
            print(
                f"3. Combined Accuracy (Task AND Label): {combined_accuracy:.2f}% "
                f"({total_both_correct}/{total_samples})")

            print(f"4. Label Accuracy (when task was INCORRECTLY identified):")
            if total_incorrect_task_samples == 0:
                print("   No incorrect task predictions were made.")
            else:
                incorrect_task_label_acc = (label_correct_when_task_was_wrong / total_incorrect_task_samples) * 100
                print(
                    f"   Accuracy: {incorrect_task_label_acc:.2f}% "
                    f"({label_correct_when_task_was_wrong}/{total_incorrect_task_samples})")

    def reset_eval_logs(self):
        self.all_task_names = []
        self.all_evaluated_labels = []
        self.all_membership_scores = []


class MultitaskRobustLossModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 loss_type: str = "ce",
                 gce_q: float = 0.7,
                 bootstrap_beta: float = 0.95) -> None:
        super().__init__()
        assert loss_type in ("ce", "gce", "bootstrapping", "mtlnl", "excessmtl"), \
            f"loss_type must be ce|gce|bootstrapping|mtlnl|excessmtl, got {loss_type!r}"
        self.bert = bert
        self.tasks = tasks
        self.task_to_idx = {t.value: i for i, t in enumerate(tasks)}
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_dict = nn.ModuleDict({
            task.value: nn.Linear(self.bert.config.hidden_size, 2)
            for task in tasks
        })
        for task in tasks:
            self.classifier_dict[task.value].weight.data.normal_(
                mean=0.0, std=self.bert.config.initializer_range)
            self.classifier_dict[task.value].bias.data.zero_()
        self.loss_type = loss_type
        self.gce_q = gce_q
        self.bootstrap_beta = bootstrap_beta
        # MTL-NL state: per-task transition matrix T[t, true, observed]
        # Initialised to identity (no correction); set externally after epoch 0 anchor estimation.
        self.register_buffer("mtlnl_T",
                             torch.eye(2).unsqueeze(0).expand(len(tasks), 2, 2).clone())
        self.mtlnl_T_frozen = False
        # ExcessMTL state: per-task weights, initialised uniform.
        self.register_buffer("excessmtl_weights", torch.ones(len(tasks)))

    def _per_sample_loss(self, logits: torch.Tensor,
                         labels: torch.Tensor,
                         tasks_in_batch: List[str] = None) -> torch.Tensor:
        if self.loss_type == "ce":
            return nn.functional.cross_entropy(logits, labels)
        if self.loss_type == "gce":
            probs = nn.functional.softmax(logits, dim=-1)
            p_y = probs[torch.arange(probs.size(0)), labels]
            return ((1.0 - p_y.pow(self.gce_q)) / self.gce_q).mean()
        if self.loss_type == "bootstrapping":
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp().detach()
            one_hot = nn.functional.one_hot(labels, num_classes=2).float()
            target = (self.bootstrap_beta * one_hot
                      + (1.0 - self.bootstrap_beta) * probs)
            return -(target * log_probs).sum(dim=-1).mean()
        if self.loss_type == "mtlnl":
            # Forward correction: NLL(log(softmax @ T), y_observed) per task.
            # T_per_task[t, true_class, observed] = P(observed | true_class)
            t_idx = torch.tensor([self.task_to_idx[t] for t in tasks_in_batch],
                                 device=logits.device, dtype=torch.long)
            T_batch = self.mtlnl_T[t_idx]  # [B, 2, 2]
            probs = nn.functional.softmax(logits, dim=-1)  # [B, 2]
            corrected = torch.einsum("bkj,bk->bj", T_batch, probs)  # [B, 2]
            return nn.functional.nll_loss(
                torch.log(corrected.clamp_min(1e-8)), labels)
        if self.loss_type == "excessmtl":
            # Per-sample CE weighted by per-task weight.
            t_idx = torch.tensor([self.task_to_idx[t] for t in tasks_in_batch],
                                 device=logits.device, dtype=torch.long)
            ce_per = nn.functional.cross_entropy(logits, labels, reduction="none")
            w_batch = self.excessmtl_weights[t_idx]
            return (ce_per * w_batch).mean()
        raise ValueError(f"Unknown loss_type {self.loss_type!r}")

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {k: v for k, v in batch.items()
                       if k not in (TASK_NAME, LABELS, "true_task_name",
                                    "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        bert_outputs = self.bert(**input_batch)
        pooled_outputs = self.dropout(bert_outputs.pooler_output)

        tasks_in_batch = batch[TASK_NAME]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()

        # Stack assigned-task logits for the loss + all-task logits for blind eval
        per_sample_logits = []
        for ii, t in enumerate(tasks_in_batch):
            cls = self.classifier_dict[t]
            per_sample_logits.append(cls(pooled_outputs[ii].unsqueeze(0)))
        per_sample_logits = torch.cat(per_sample_logits, dim=0)  # [B, 2]

        # eval-time labels=-1: skip in loss
        valid = (labels != -1)
        if valid.any():
            if self.loss_type in ("mtlnl", "excessmtl"):
                tasks_valid = [tasks_in_batch[i] for i in range(len(tasks_in_batch)) if valid[i]]
                loss = self._per_sample_loss(
                    per_sample_logits[valid], labels[valid], tasks_valid)
            else:
                loss = self._per_sample_loss(per_sample_logits[valid], labels[valid])
        else:
            loss = torch.tensor(0.0, device=per_sample_logits.device)

        if not self.training:
            for ii in range(len(tasks_in_batch)):
                pred_label = torch.argmax(per_sample_logits[ii]).item()
                self.all_task_names.append(tasks_in_batch[ii])
                self.all_evaluated_labels.append(batch[LABELS][ii].item())
                self.all_membership_scores.append({
                    "predicted_task": tasks_in_batch[ii],
                    "predicted_label": pred_label,
                })
        return loss, per_sample_logits


class PooledMultitaskModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 loss_type: str = "ce",
                 bootstrap_beta: float = 0.95,
                 gce_q: float = 0.7,
                 discard_ratio: float = 0.3) -> None:
        super().__init__()
        assert loss_type in ("ce", "gce", "bootstrapping", "hld"), \
            f"loss_type must be ce|gce|bootstrapping|hld, got {loss_type!r}"
        self.bert = bert
        self.tasks = tasks
        self.loss_type = loss_type
        self.bootstrap_beta = bootstrap_beta
        self.gce_q = gce_q
        self.discard_ratio = discard_ratio
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier.weight.data.normal_(
            mean=0.0, std=self.bert.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {k: v for k, v in batch.items()
                       if k not in (TASK_NAME, LABELS, "true_task_name",
                                    "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        bert_outputs = self.bert(**input_batch)
        pooled_outputs = self.dropout(bert_outputs.pooler_output)
        logits = self.classifier(pooled_outputs)  # [B, 2]

        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        # During eval, label is set to -1; emit zero loss
        valid = (labels != -1)
        if valid.any():
            v_logits = logits[valid]
            v_labels = labels[valid]
            if self.loss_type == "ce":
                loss = nn.functional.cross_entropy(v_logits, v_labels)
            elif self.loss_type == "gce":
                probs = nn.functional.softmax(v_logits, dim=-1)
                p_y = probs[torch.arange(probs.size(0)), v_labels]
                loss = ((1.0 - p_y.pow(self.gce_q)) / self.gce_q).mean()
            elif self.loss_type == "bootstrapping":
                log_probs = nn.functional.log_softmax(v_logits, dim=-1)
                probs = log_probs.exp().detach()
                one_hot = nn.functional.one_hot(v_labels, num_classes=2).float()
                target = (self.bootstrap_beta * one_hot
                          + (1.0 - self.bootstrap_beta) * probs)
                loss = -(target * log_probs).sum(dim=-1).mean()
            elif self.loss_type == "hld":
                per_sample_ce = nn.functional.cross_entropy(
                    v_logits, v_labels, reduction="none")
                n_keep = max(1, int(per_sample_ce.numel()
                                    * (1.0 - self.discard_ratio)))
                keep_idx = per_sample_ce.argsort()[:n_keep]
                loss = per_sample_ce[keep_idx].mean()
        else:
            loss = torch.tensor(0.0, device=logits.device)

        if not self.training:
            tasks_in_batch = batch[TASK_NAME]
            for idx in range(len(tasks_in_batch)):
                self.all_task_names.append(tasks_in_batch[idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                pred = torch.argmax(logits[idx]).item()
                # No routing: predicted_task = true task (we don't claim to
                # identify the task; just record true task for per-task
                # accuracy bookkeeping in the eval framework).
                self.all_membership_scores.append({
                    "predicted_task": tasks_in_batch[idx],
                    "predicted_label": pred,
                })
        return loss, logits


class MultitaskClassificationModel(BaseMultitaskModel):
    def __init__(self, bert: BertModel, tasks: List[GlueTask], load_trained_classifier: bool = False):
        super(MultitaskClassificationModel, self).__init__()
        self.bert = bert
        self.tasks = tasks
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier_dict = nn.ModuleDict({task.value: nn.Linear(self.bert.config.hidden_size, 2) for task in tasks})
        if not load_trained_classifier:
            for task in tasks:
                self.classifier_dict[task.value].weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                self.classifier_dict[task.value].bias.data.zero_()
        else:
            load_pretrained_classifier(BERT_SINGLE_TASK_DIR, self.classifier_dict)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}
        bert_outputs = self.bert(**input_batch)

        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        tasks = batch[TASK_NAME]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        logits = []
        losses = []
        for ii in range(len(tasks)):
            label = labels[ii].unsqueeze(0)
            pooled_output = pooled_outputs[ii].unsqueeze(0)
            classifier = self.classifier_dict[tasks[ii]]
            logit = classifier(pooled_output)
            losses.append(self.loss_fct(logit, label))
            logits.append(logit)
        logits = torch.cat(logits)

        if not self.training:
            for ii in range(len(tasks)):
                pred_label = torch.argmax(logits[ii]).item()
                self.all_task_names.append(tasks[ii])
                self.all_evaluated_labels.append(batch[LABELS][ii].item())
                self.all_membership_scores.append({
                    "predicted_task": tasks[ii],
                    "predicted_label": pred_label,
                })

        return torch.sum(torch.stack(losses)), logits


class MultitaskTaskIdentificationModel(BaseMultitaskModel):
    def __init__(self, bert: BertModel, tasks: List[GlueTask], load_trained_classifier: bool = False):
        super(MultitaskTaskIdentificationModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.task_identification = nn.Linear(self.bert.config.hidden_size, len(tasks))
        self.task_to_index = {task.value: ii for ii, task in enumerate(tasks)}
        self.index_to_task = {value: task_name for task_name, value in self.task_to_index.items()}
        print("---->>> self.index_to_task : ", self.index_to_task)
        self.classifier_dict = nn.ModuleDict({task.value: nn.Linear(self.bert.config.hidden_size, 2) for task in tasks})
        if not load_trained_classifier:
            for task in tasks:
                self.classifier_dict[task.value].weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                self.classifier_dict[task.value].bias.data.zero_()
        else:
            load_pretrained_classifier(BERT_SINGLE_TASK_DIR, self.classifier_dict)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}

        bert_outputs = self.bert(**input_batch)
        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        task_identification_output = self.task_identification(pooled_outputs)
        # Get predicted task IDs for each sample in the batch.
        predicted_task_ids = torch.argmax(task_identification_output, dim=1)

        tasks = batch[TASK_NAME]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()

        # Create proper task identification labels.
        task_identification_label = torch.tensor([self.task_to_index[task] for task in tasks],
                                                 dtype=torch.long,
                                                 device=labels.device)
        task_identification_loss = self.loss_fct(task_identification_output, task_identification_label)

        # Eval-mode routing:
        # - K mode (self._eval_blind is False): route by oracle batch[TASK_NAME].
        # - B mode (self._eval_blind is True OR None OR attribute absent):
        #   route by gate's argmax (predicted_task_ids).
        # During training we always use gate routing (consistent with the
        # gate-supervision loss).
        use_oracle = (not self.training) and (getattr(self, "_eval_blind", None) is False)

        logits = []
        losses = []
        for ii in range(len(tasks)):
            pooled_output = pooled_outputs[ii].unsqueeze(0)
            classifier = self.classifier_dict[tasks[ii]]

            # Loss is always computed at the oracle task (training signal).
            loss_logit = classifier(pooled_output)
            if labels[ii] != -1:
                label = labels[ii].unsqueeze(0)
                losses.append(self.loss_fct(loss_logit, label))
            else:
                losses.append(torch.tensor(0))

            # Routing for the saved prediction:
            if use_oracle:
                route_task = tasks[ii]  # KNOWN: oracle-routed
            else:
                route_task = self.index_to_task[predicted_task_ids[ii].item()]
            routed_classifier = self.classifier_dict[route_task]
            routed_logits = routed_classifier(pooled_output)
            logits.append(routed_logits)

        logits = torch.cat(logits)

        for idx in range(len(batch[TASK_NAME])):
            if not self.training:
                self.all_task_names.append(batch[TASK_NAME][idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                if use_oracle:
                    pred_task_name = batch[TASK_NAME][idx]
                else:
                    pred_task_name = self.index_to_task[predicted_task_ids[idx].item()]
                pred_labels = torch.argmax(logits[idx]).item()
                self.all_membership_scores.append({"predicted_task": pred_task_name, "predicted_label": pred_labels})
        return torch.sum(torch.stack(losses)) / len(tasks) + task_identification_loss, logits


class MultitaskBinaryTaskIdentificationModel(BaseMultitaskModel):
    def __init__(self, bert: BertModel, tasks: List[GlueTask], load_trained_classifier: bool = False):
        super(MultitaskBinaryTaskIdentificationModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.tasks = tasks
        self.task_identification = nn.Linear(self.bert.config.hidden_size, len(self.tasks))
        self.task_to_index = {task.value: ii for ii, task in enumerate(tasks)}
        self.index_to_task = {value: task_name for task_name, value in self.task_to_index.items()}
        print("---->>> self.index_to_task : ", self.index_to_task)
        self.classifier_dict = nn.ModuleDict({task.value: nn.Linear(self.bert.config.hidden_size, 2) for task in tasks})
        if not load_trained_classifier:
            for task in tasks:
                self.classifier_dict[task.value].weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
                self.classifier_dict[task.value].bias.data.zero_()
        else:
            load_pretrained_classifier(BERT_SINGLE_TASK_DIR, self.classifier_dict)
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}

        bert_outputs = self.bert(**input_batch)
        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        task_identification_output = self.task_identification(pooled_outputs)
        # Convert per-task logits to fuzzy degrees via sigmoid, then take argmax over tasks to get the predicted task ID per sample.
        probabilities = torch.sigmoid(task_identification_output)
        predicted_task_ids = torch.argmax(probabilities, dim=1)

        tasks = batch[TASK_NAME]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()

        # Create proper task identification labels.
        labels = [[0 if ii != self.task_to_index[task] else 1 for ii in range(len(self.tasks))] for task in tasks]
        task_identification_label = torch.tensor(labels,
                                                 dtype=torch.long,
                                                 device=labels.device)
        task_identification_loss = self.loss_fct(task_identification_output, task_identification_label)

        # Eval-mode routing:
        # - K mode (self._eval_blind is False): route by oracle batch[TASK_NAME].
        # - B mode (self._eval_blind is True OR None OR attribute absent):
        #   route by gate's argmax (predicted_task_ids).
        # During training we always use gate routing (consistent with the
        # gate-supervision loss).
        use_oracle = (not self.training) and (getattr(self, "_eval_blind", None) is False)

        logits = []
        losses = []
        for ii in range(len(tasks)):
            pooled_output = pooled_outputs[ii].unsqueeze(0)
            classifier = self.classifier_dict[tasks[ii]]

            # Loss is always computed at the oracle task (training signal).
            loss_logit = classifier(pooled_output)
            if labels[ii] != -1:
                label = labels[ii].unsqueeze(0)
                losses.append(self.loss_fct(loss_logit, label))
            else:
                losses.append(torch.tensor(0))

            # Routing for the saved prediction:
            if use_oracle:
                route_task = tasks[ii]  # KNOWN: oracle-routed
            else:
                route_task = self.index_to_task[predicted_task_ids[ii].item()]
            routed_classifier = self.classifier_dict[route_task]
            routed_logits = routed_classifier(pooled_output)
            logits.append(routed_logits)

        logits = torch.cat(logits)

        for idx in range(len(batch[TASK_NAME])):
            if not self.training:
                self.all_task_names.append(batch[TASK_NAME][idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                if use_oracle:
                    pred_task_name = batch[TASK_NAME][idx]
                else:
                    pred_task_name = self.index_to_task[predicted_task_ids[idx].item()]
                pred_labels = torch.argmax(logits[idx]).item()
                self.all_membership_scores.append({"predicted_task": pred_task_name, "predicted_label": pred_labels})
        return torch.sum(torch.stack(losses)) / len(tasks) + task_identification_loss, logits


class MultitaskTaskMaximumHeadModel(MultitaskClassificationModel, BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask], load_trained_classifier: bool = False):
        super(MultitaskTaskMaximumHeadModel, self).__init__(bert=bert, tasks=tasks,
                                                            load_trained_classifier=load_trained_classifier)

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value.cuda() for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}
        bert_outputs = self.bert(**input_batch)

        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        tasks = batch[TASK_NAME]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        logits = []
        losses = []
        for ii in range(len(tasks)):
            label = labels[ii].unsqueeze(0)
            # Calculate loss.
            pooled_output = pooled_outputs[ii].unsqueeze(0)
            correct_classifier = self.classifier_dict[tasks[ii]]
            correct_logit = correct_classifier(pooled_output)
            if label[0] != -1:
                # During evaluation, label is set as -1.
                losses.append(self.loss_fct(correct_logit, label))
            else:
                losses.append(torch.tensor(0))
            # Calculate predictions. Each prediction is a tensor with dim 2 corresponding to positive and negative. We
            # get the logit corresponding to the highest softmax probability.
            all_logits = [classifier(pooled_output) for classifier in self.classifier_dict.values()]
            all_probs = [torch.softmax(logits, dim=-1) for logits in all_logits]
            # Find classifier with highest confidence.
            max_idx = torch.argmax(torch.tensor([torch.max(prob) for prob in all_probs])).item()
            logits.append(all_logits[max_idx])
        logits = torch.cat(logits)

        for idx in range(len(batch[TASK_NAME])):
            if not self.training:
                self.all_task_names.append(batch[TASK_NAME][idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                # Membership score: max head index and predicted label
                # Find which head (task) produced the max confidence for this sample
                pooled_output = pooled_outputs[idx].unsqueeze(0)
                all_logits = [classifier(pooled_output) for classifier in self.classifier_dict.values()]
                all_probs = [torch.softmax(logit, dim=-1) for logit in all_logits]
                max_idx = torch.argmax(torch.tensor([torch.max(prob.detach()) for prob in all_probs])).item()
                max_task = list(self.classifier_dict.keys())[max_idx]
                pred_label = torch.argmax(all_logits[max_idx]).item()
                self.all_membership_scores.append({"max_task_head": max_task, "predicted_label": pred_label})
        return torch.sum(torch.stack(losses)), logits


class MultitaskFuzzyQueryBasedModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 do_blindly_decode: bool = True, is_max_opposite: bool = True):
        super(MultitaskFuzzyQueryBasedModel, self).__init__()
        self.bert = bert
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_to_idx = {task.value: ii for ii, task in enumerate(self.tasks)}
        self.query_vectors = nn.Parameter(torch.randn(self.hidden_size, self.num_tasks))
        self.query_vectors = nn.init.xavier_uniform_(self.query_vectors)
        # Fuzzy set centers (learnable or fixed). For positive, unknown, negative fuzzy sets.
        self.fuzzy_centers = nn.Parameter(torch.tensor([[-1.0, 0.0, 1.0] for _ in range(self.num_tasks)]))  # N x 3.
        self.fuzzy_std = nn.Parameter(torch.tensor([1.0 for _ in range(self.num_tasks)]))
        self.do_blindly_decode = do_blindly_decode
        self.is_max_opposite = is_max_opposite

        logger.info(f"Task to Index: {self.task_to_idx}")
        logger.info(f"DO BLINDLY DECODE: {do_blindly_decode}")
        logger.info(f"USE MAX OPPOSITE: {self.is_max_opposite}")

    def gaussian_membership(self, xx, center, std):
        return torch.exp(-0.5 * ((xx - center) / std) ** 2)

    def compute_fuzzy_loss(self, fuzzy_memberships: torch.Tensor, task_labels: torch.Tensor,
                           correct_task_idx: List[int]) -> torch.Tensor:
        batch_size = fuzzy_memberships.shape[0]
        total_loss = 0.0

        for bs in range(batch_size):
            correct_task = correct_task_idx[bs]
            task_label = task_labels[bs]

            # Loss for the correct task (should be positive or negative, not unknown).
            if task_label == 1:  # Positive
                # Maximize positive membership, minimize unknown and negative.
                correct_task_loss = -torch.log(fuzzy_memberships[bs, correct_task, FuzzySet.POSITIVE.value] + 1e-8)
            else:  # Negative (task_label == 0)
                # Maximize negative membership, minimize unknown and positive.
                correct_task_loss = -torch.log(fuzzy_memberships[bs, correct_task, FuzzySet.NEGATIVE.value] + 1e-8)

            # Loss for other tasks (should be unknown).
            other_tasks_loss = 0.0
            for task_idx in range(self.num_tasks):
                if task_idx != correct_task:
                    # Maximize unknown membership for non-relevant tasks.
                    unknown_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.UNKNOWN.value] + 1e-8)

                    if self.is_max_opposite:
                        # Additionally, maximize the opposite membership for non-relevant tasks.
                        if task_label == 1:  # Positive
                            opposite_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.NEGATIVE.value] + 1e-8)
                        else:  # Negative
                            opposite_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.POSITIVE.value] + 1e-8)
                        other_tasks_loss += (unknown_loss + opposite_loss) / 2
                    else:
                        other_tasks_loss += unknown_loss
            if self.num_tasks > 1:
                other_tasks_loss /= (self.num_tasks - 1)  # Average over other tasks.
            total_loss += correct_task_loss + other_tasks_loss
        return total_loss / batch_size

    def defuzzy(self, fuzzy_memberships: torch.Tensor, correct_task_idx: Optional[List[int]] = None):
        batch_size = fuzzy_memberships.shape[0]
        predictions = torch.zeros(batch_size, 2, dtype=torch.float, device=fuzzy_memberships.device)

        for bs in range(batch_size):
            task_memberships = fuzzy_memberships[bs]
            positive_membership = task_memberships[:, FuzzySet.POSITIVE.value]  # mu_i
            negative_membership = task_memberships[:, FuzzySet.NEGATIVE.value]  # nu_i
            unknown_membership = task_memberships[:, FuzzySet.UNKNOWN.value]  # pi_i

            if correct_task_idx:
                correct_task = correct_task_idx[bs]

                # Compare only positive vs negative (ignore unknown).
                correct_positive_membership = positive_membership[correct_task]
                correct_negative_membership = negative_membership[correct_task]

                # Predict positive if positive membership > negative membership.
                if correct_positive_membership > correct_negative_membership:
                    predictions[bs] = torch.tensor([0, 1], dtype=torch.float)  # Positive.
                else:
                    predictions[bs] = torch.tensor([1, 0], dtype=torch.float)  # Negative.
            else:
                # Product T-norm approach:
                # Sup_pos_i = mu_i * (1-pi_i)
                # Sup_neg_i = nu_i * (1-pi_i)
                # conf_i = max(Sup_pos_i, Sup_neg_i) = (1-pi_i) * max(mu_i, nu_i)

                defuzzied_candidate_results: List[Tuple[torch.Tensor, float]] = []
                for ii in range(self.num_tasks):
                    mu_i = positive_membership[ii]
                    nu_i = negative_membership[ii]
                    pi_i = unknown_membership[ii]

                    # Calculate support values using Product T-norm
                    sup_pos = mu_i * (1 - pi_i)
                    sup_neg = nu_i * (1 - pi_i)

                    # Confidence is the maximum support
                    conf_i = torch.max(sup_pos, sup_neg)

                    # Determine the predicted label based on which support is higher
                    if sup_pos > sup_neg:
                        defuzzied_candidate_results.append((torch.tensor([0, 1], dtype=torch.float), conf_i.item()))
                    else:
                        defuzzied_candidate_results.append((torch.tensor([1, 0], dtype=torch.float), conf_i.item()))

                if not defuzzied_candidate_results:
                    # If the model is uncertain about a sample, return negative.
                    predictions[bs] = torch.tensor([1, 0], dtype=torch.float)  # Negative.
                else:
                    # Select the task with the highest confidence
                    sorted_candidate_results = sorted(defuzzied_candidate_results, key=lambda xx: xx[1], reverse=True)
                    predictions[bs] = sorted_candidate_results[0][0]
        return predictions

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value.cuda() for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}
        bert_outputs = self.bert(**input_batch)

        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        task_logits = torch.matmul(pooled_outputs, self.query_vectors)  # bs x N.

        # Expand dimensions for broadcasting.
        task_logits_expanded = task_logits.unsqueeze(-1)  # bs x N x 1.
        centers_expanded = self.fuzzy_centers.unsqueeze(0)  # 1 x N x 3.
        std_expanded = self.fuzzy_std.unsqueeze(-1)  # 1 x N x 1

        # Compute Gaussian membership for all tasks and fuzzy sets at once.
        fuzzy_memberships = self.gaussian_membership(task_logits_expanded, centers_expanded, std_expanded)

        # Normalize memberships.
        fuzzy_memberships = fuzzy_memberships / (torch.sum(fuzzy_memberships, dim=-1, keepdim=True) + 1e-8)

        for idx in range(len(batch[TASK_NAME])):
            if not self.training:
                self.all_task_names.append(batch[TASK_NAME][idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                self.all_membership_scores.append(fuzzy_memberships[idx].detach().cpu().tolist())
        labels = batch[LABELS].cuda()
        task_names = batch[TASK_NAME]
        task_idxes = [self.task_to_idx[task_name] for task_name in task_names]
        loss = self.compute_fuzzy_loss(fuzzy_memberships, labels, task_idxes)
        predictions = self.defuzzy(fuzzy_memberships, task_idxes if not self.do_blindly_decode else None)
        return loss, predictions

    def log_eval_stats(self, verbose=True):
        memberships = self.all_membership_scores
        labels = self.all_evaluated_labels
        task_names = self.all_task_names

        if not memberships or not labels or not task_names:
            print("No evaluation logs to analyze.")
            return
        task_map = self.task_to_idx
        membership_map = {"pos": 0, "unknown": 1, "neg": 2}

        NUM_MODELS = len(memberships[0]) if memberships else 0
        if NUM_MODELS == 0:
            print("No membership scores logged.")
            return

        NUM_MEMBERSHIPS = 3

        # 1. Distribution of 'unknown' max memberships per sample 
        samples_count_1 = []
        for sample_memberships in memberships:
            if len(sample_memberships) != NUM_MODELS:
                continue
            try:
                max_indices_for_sample = [np.argmax(model_scores) for model_scores in sample_memberships]
                unknown_count = max_indices_for_sample.count(membership_map["unknown"])
                samples_count_1.append(unknown_count)
            except Exception:
                continue

        total_samples = len(samples_count_1)
        if total_samples == 0:
            print("No valid membership samples processed.")
            return

        if verbose:
            print("\n--- Distribution of 'Unknown' Max Memberships per Sample ---")
            for k in range(NUM_MODELS + 1):  # Check for 0, 1, 2, 3 unknowns
                count = samples_count_1.count(k)
                percentage = (count / total_samples) * 100 if total_samples else 0
                print(f"  Samples with {k} 'unknown' model(s): {count: >5} / {total_samples} ({percentage:.2f}%)")

        # 2. Label accuracy when only one model is not 'unknown' 
        q2_total_samples_checked = 0
        q2_correct_label_matches = 0
        for i, sample_memberships in enumerate(memberships):
            if len(sample_memberships) != NUM_MODELS:
                continue
            try:
                max_memberships_by_model = [np.argmax(model_scores) for model_scores in sample_memberships]
            except Exception:
                continue
            non_unknown_model_indices = [
                idx for idx, max_mem in enumerate(max_memberships_by_model)
                if max_mem != membership_map["unknown"]
            ]
            if len(non_unknown_model_indices) == 1:
                q2_total_samples_checked += 1
                predicted_task_index = non_unknown_model_indices[0]
                scores_for_predicted_task = sample_memberships[predicted_task_index]
                pos_score = scores_for_predicted_task[membership_map["pos"]]
                neg_score = scores_for_predicted_task[membership_map["neg"]]
                predicted_label = 1 if pos_score > neg_score else 0
                true_label = int(labels[i])
                if predicted_label == true_label:
                    q2_correct_label_matches += 1
        if verbose:
            print("\n--- Label Prediction Accuracy (when all but one model are 'unknown') ---")
            if q2_total_samples_checked == 0:
                print("No samples matched the 'single non-unknown' logic.")
            else:
                accuracy = (q2_correct_label_matches / q2_total_samples_checked) * 100
                print(f"Samples matching logic ({NUM_MODELS - 1} 'unknowns'): {q2_total_samples_checked}")
                print(f"Correct label predictions: {q2_correct_label_matches}")
                print(f"Accuracy: {accuracy:.2f}%")

        # 3. Task accuracy when only one model is not 'unknown'
        q3_total_samples_checked = 0
        q3_correct_task_matches = 0
        for i, sample_memberships in enumerate(memberships):
            if len(sample_memberships) != NUM_MODELS:
                continue
            try:
                max_memberships_by_model = [np.argmax(model_scores) for model_scores in sample_memberships]
            except Exception:
                continue
            non_unknown_model_indices = [
                idx for idx, max_mem in enumerate(max_memberships_by_model)
                if max_mem != membership_map["unknown"]
            ]
            if len(non_unknown_model_indices) == 1:
                q3_total_samples_checked += 1
                predicted_task_index = non_unknown_model_indices[0]
                true_task_name = task_names[i]
                if true_task_name not in task_map:
                    continue
                true_task_index = task_map[true_task_name]
                if predicted_task_index == true_task_index:
                    q3_correct_task_matches += 1
        if verbose:
            print("\n--- Task Prediction Accuracy (when all but one model are 'unknown') ---")
            if q3_total_samples_checked == 0:
                print("No samples matched the 'single non-unknown' logic.")
            else:
                accuracy = (q3_correct_task_matches / q3_total_samples_checked) * 100
                print(f"Samples matching logic ({NUM_MODELS - 1} 'unknowns'): {q3_total_samples_checked}")
                print(f"Correct task predictions: {q3_correct_task_matches}")
                print(f"Accuracy: {accuracy:.2f}%")

        # 4. Combined accuracy (task and label) 
        total_samples_matching_logic = 0
        correct_task_and_label_matches = 0
        for i, sample_memberships in enumerate(memberships):
            if len(sample_memberships) != NUM_MODELS:
                continue
            try:
                max_memberships_by_model = [np.argmax(model_scores) for model_scores in sample_memberships]
            except Exception:
                continue
            non_unknown_model_indices = [
                idx for idx, max_mem in enumerate(max_memberships_by_model)
                if max_mem != membership_map["unknown"]
            ]
            if len(non_unknown_model_indices) == 1:
                total_samples_matching_logic += 1
                predicted_task_index = non_unknown_model_indices[0]
                true_task_name = task_names[i]
                if true_task_name not in task_map:
                    continue
                true_task_index = task_map[true_task_name]
                task_is_correct = (predicted_task_index == true_task_index)
                scores_for_predicted_task = sample_memberships[predicted_task_index]
                pos_score = scores_for_predicted_task[membership_map["pos"]]
                neg_score = scores_for_predicted_task[membership_map["neg"]]
                predicted_label = 1 if pos_score > neg_score else 0
                true_label = int(labels[i])
                label_is_correct = (predicted_label == true_label)
                if task_is_correct and label_is_correct:
                    correct_task_and_label_matches += 1
        if verbose:
            print("\n--- Combined Task & Label Accuracy (when all but one model are 'unknown') ---")
            if total_samples_matching_logic == 0:
                print("No samples matched the 'single non-unknown' logic.")
            else:
                accuracy = (correct_task_and_label_matches / total_samples_matching_logic) * 100
                print(f"Samples matching logic ({NUM_MODELS - 1} 'unknowns'): {total_samples_matching_logic}")
                print(f"Correct task AND label predictions: {correct_task_and_label_matches}")
                print(f"Combined Accuracy: {accuracy:.2f}%")

        # 5. Accuracy on Incorrect Task Predictions
        q5_total_incorrect_task_samples = 0
        q5_correct_label_matches = 0
        for i, sample_memberships in enumerate(memberships):
            if len(sample_memberships) != NUM_MODELS:
                continue
            try:
                max_memberships_by_model = [np.argmax(model_scores) for model_scores in sample_memberships]
            except Exception:
                continue

            non_unknown_model_indices = [
                idx for idx, max_mem in enumerate(max_memberships_by_model)
                if max_mem != membership_map["unknown"]
            ]

            if len(non_unknown_model_indices) == 1:
                predicted_task_index = non_unknown_model_indices[0]

                true_task_name = task_names[i]
                if true_task_name not in task_map:
                    continue
                true_task_index = task_map[true_task_name]

                # Check if task was incorrect
                if predicted_task_index != true_task_index:
                    q5_total_incorrect_task_samples += 1

                    # Check label prediction for the (incorrectly) predicted task
                    scores_for_predicted_task = sample_memberships[predicted_task_index]
                    pos_score = scores_for_predicted_task[membership_map["pos"]]
                    neg_score = scores_for_predicted_task[membership_map["neg"]]
                    predicted_label = 1 if pos_score > neg_score else 0
                    true_label = int(labels[i])

                    if predicted_label == true_label:
                        q5_correct_label_matches += 1

        if verbose:
            print("\n--- Label Accuracy (when task was INCORRECTLY identified) ---")
            if q5_total_incorrect_task_samples == 0:
                print("No samples matched the 'incorrect task' logic.")
            else:
                accuracy = (q5_correct_label_matches / q5_total_incorrect_task_samples) * 100
                print(f"Samples matching logic (incorrect task): {q5_total_incorrect_task_samples}")
                print(f"Correct label predictions (despite wrong task): {q5_correct_label_matches}")
                print(f"Accuracy: {accuracy:.2f}%")


class MultitaskDiffMembershipsFQBModel(MultitaskFuzzyQueryBasedModel):
    def __init__(self, membership_type: str = "gaussian", **kwargs):
        super(MultitaskDiffMembershipsFQBModel, self).__init__(**kwargs)
        self.membership_type = membership_type
        logger.info(f"Using {membership_type} membership function.")
        # Learnable parameters for membership functions
        init = {
            "gaussian": [(-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)],
            "triangular": [(-2.0, -1.0, 0.0), (-1.0, 0.0, 1.0), (0.0, 1.0, 2.0)],
            "trapezoidal": [(-2.0, -1.5, -0.5, 0.0), (-1.0, -0.5, 0.5, 1.0), (0.0, 0.5, 1.5, 2.0)],
            "sigmoid": [(-1.0, 1.0), (0.0, 1.0), (1.0, 1.0)],
            "bell": [(-1.0, 2.0, 1.0), (0.0, 2.0, 1.0), (1.0, 2.0, 1.0)],
        }
        fuzzy_params_init = torch.tensor([init[membership_type] for _ in range(self.num_tasks)], dtype=torch.float)
        logger.info(f"Initializing fuzzy membership parameters with: {fuzzy_params_init}")
        self.fuzzy_params = nn.Parameter(fuzzy_params_init)  # [num_tasks, 3, num_params]

        # Remove from computation graph.
        del self._parameters["fuzzy_centers"]
        del self._parameters["fuzzy_std"]
        self.membership_function = MEMBERSHIP_FUNCTIONS[membership_type]

        self.training_logs = {"steps": [], "train_loss": [], "unknown_loss": [], "opposite_loss": [],
                              "correct_task_loss": [], "fuzzy_params_grad_norm": [], "query_vectors_grad_norm": [],
                              "fuzzy_memberships": []}
        # Register gradient hooks
        self.fuzzy_params.register_hook(self._save_fuzzy_params_grad)
        self.query_vectors.register_hook(self._save_query_vectors_grad)

        self.current_grad = {}  # To store current gradients
        logger.info(f"Is max opposite enabled: {self.is_max_opposite}")

    def _save_fuzzy_params_grad(self, grad):
        self.current_grad["fuzzy_params"] = grad.norm().item()
        return grad

    def _save_query_vectors_grad(self, grad):
        self.current_grad["query_vectors"] = grad.norm().item()
        return grad

    def fuzzy_membership(self, xx, params):
        return self.membership_function(xx, *params)

    def compute_fuzzy_loss(self, fuzzy_memberships: torch.Tensor, task_labels: torch.Tensor,
                           correct_task_idx: List[int]) -> torch.Tensor:
        batch_size = fuzzy_memberships.shape[0]
        total_loss = 0.0
        total_correct_task_loss = 0.0
        total_unknown_loss = 0.0
        total_opposite_loss = 0.0

        for bs in range(batch_size):
            correct_task = correct_task_idx[bs]
            task_label = task_labels[bs]

            # Loss for the correct task (should be positive or negative, not unknown).
            if task_label == 1:  # Positive
                # Maximize positive membership, minimize unknown and negative.
                correct_task_loss = -torch.log(fuzzy_memberships[bs, correct_task, FuzzySet.POSITIVE.value] + 1e-8)
            else:  # Negative (task_label == 0)
                # Maximize negative membership, minimize unknown and positive.
                correct_task_loss = -torch.log(fuzzy_memberships[bs, correct_task, FuzzySet.NEGATIVE.value] + 1e-8)

            total_correct_task_loss += correct_task_loss.item()

            # Loss for other tasks (should be unknown).
            other_tasks_loss = 0.0
            for task_idx in range(self.num_tasks):
                if task_idx != correct_task:
                    # Maximize unknown membership for non-relevant tasks.
                    unknown_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.UNKNOWN.value] + 1e-8)

                    if self.is_max_opposite:
                        # Additionally, maximize the opposite membership for non-relevant tasks.
                        if task_label == 1:  # Positive
                            opposite_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.NEGATIVE.value] + 1e-8)
                        else:  # Negative
                            opposite_loss = -torch.log(fuzzy_memberships[bs, task_idx, FuzzySet.POSITIVE.value] + 1e-8)
                        other_tasks_loss += (unknown_loss + opposite_loss) / 2
                    else:
                        other_tasks_loss += unknown_loss
            if self.num_tasks > 1:
                other_tasks_loss /= (self.num_tasks - 1)  # Average over other tasks.

            total_unknown_loss += unknown_loss.item()
            if self.is_max_opposite:
                total_opposite_loss += opposite_loss.item()
            total_loss += correct_task_loss + other_tasks_loss

        # Log average losses
        loss_components = {"correct_task_loss": total_correct_task_loss / batch_size,
                           "unknown_loss": total_unknown_loss / batch_size,
                           "opposite_loss": total_opposite_loss / batch_size if self.is_max_opposite else 0.0}
        return total_loss / batch_size, loss_components

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        input_batch = {key: value.cuda() for key, value in batch.items() if
                       key not in (TASK_NAME, LABELS, "true_task_name", "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {key: value.cuda() for key, value in input_batch.items()}
        bert_outputs = self.bert(**input_batch)

        pooled_outputs = bert_outputs.pooler_output
        pooled_outputs = self.dropout(pooled_outputs)

        task_logits = torch.matmul(pooled_outputs, self.query_vectors)  # bs x N.

        # Expand dimensions for broadcasting.
        task_logits_expanded = task_logits.unsqueeze(-1)  # bs x N x 1.
        fuzzy_params_expanded = self.fuzzy_params.unsqueeze(0)

        # Compute memberships for all tasks and fuzzy sets
        # Output: [bs, N, 3]
        fuzzy_memberships = torch.zeros(task_logits_expanded.shape[0], self.num_tasks, 3, device=task_logits.device)
        for fuzzy_set_idx in range(3):
            params = fuzzy_params_expanded[:, :, fuzzy_set_idx, :]  # [1, N, num_params]
            # Broadcast params to [bs, N, num_params]
            params = params.expand(task_logits_expanded.shape[0], -1, -1)
            fuzzy_memberships[:, :, fuzzy_set_idx] = self.membership_function(
                task_logits_expanded.squeeze(-1), *[params[..., i] for i in range(params.shape[-1])]
            )

        # Normalize memberships.
        fuzzy_memberships = fuzzy_memberships / (torch.sum(fuzzy_memberships, dim=-1, keepdim=True) + 1e-8)

        labels = batch[LABELS].cuda()
        task_names = batch[TASK_NAME]
        task_idxes = [self.task_to_idx[task_name] for task_name in task_names]

        # Get loss with components
        loss, loss_components = self.compute_fuzzy_loss(fuzzy_memberships, labels, task_idxes)

        # Store loss components for logging (will be retrieved by trainer)
        self.current_loss_components = loss_components
        # Store fuzzy membership for all tasks
        self.current_fuzzy_memberships = {task: fuzzy_memberships[:, idx, :].detach().cpu().tolist()
                                          for task, idx in self.task_to_idx.items()}

        predictions = self.defuzzy(fuzzy_memberships, task_idxes if not self.do_blindly_decode else None)
        return loss, predictions
