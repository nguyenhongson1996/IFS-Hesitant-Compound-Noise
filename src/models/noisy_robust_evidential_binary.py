from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.models.evidential_binary import (
    EvidentialBinaryHead,
    evidential_mse_loss,
    evidential_predict,
    evidential_route_blind,
)
from src.models.multitask_cls import BaseMultitaskModel


class NoisyRobustEvidentialBinaryModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 kl_anneal_epochs: int = 10):
        super().__init__()
        self.bert = bert
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_to_idx = {t.value: i for i, t in enumerate(tasks)}
        self.idx_to_task = {i: t.value for i, t in enumerate(tasks)}
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.head = EvidentialBinaryHead(
            self.bert.config.hidden_size, num_tasks=len(tasks))
        # Annealing state - set externally by the trainer per epoch.
        self.kl_lambda: float = 0.0

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        input_batch = {k: v for k, v in batch.items()
                       if k not in (TASK_NAME, LABELS, "true_task_name",
                                    "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        bert_out = self.bert(**input_batch)
        pooled = self.dropout(bert_out.pooler_output)
        alpha = self.head(pooled)  # [B, T, 2]

        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        tasks_in_batch = batch[TASK_NAME]
        task_idx = torch.tensor(
            [self.task_to_idx[t] for t in tasks_in_batch],
            dtype=torch.long, device=alpha.device)

        # Loss on assigned task only (eval-time labels=-1 are skipped)
        valid = (labels != -1)
        if valid.any():
            loss = evidential_mse_loss(
                alpha[valid], labels[valid], task_idx[valid],
                kl_lambda=self.kl_lambda,
            )
        else:
            loss = torch.tensor(0.0, device=alpha.device)

        # Predictions: argmax over the assigned task's two classes
        b_idx = torch.arange(alpha.size(0), device=alpha.device)
        own_alpha = alpha[b_idx, task_idx]  # [B, 2]
        own_p = evidential_predict(own_alpha.unsqueeze(1)).squeeze(1)  # [B, 2]
        # logits-like quantity for trainer to compute downstream metrics:
        # use predictive probabilities (already on simplex) as "logits".
        logits = own_p

        if not self.training:
            # Compute blind-routed predictions for eval logging
            hat_t = evidential_route_blind(alpha)  # [B]
            p_blind_per_task = evidential_predict(alpha)  # [B, T, 2]
            for idx in range(len(tasks_in_batch)):
                self.all_task_names.append(tasks_in_batch[idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                pt = hat_t[idx].item()
                p_b = p_blind_per_task[idx, pt]  # [2]
                pred_label = int(p_b.argmax().item())
                self.all_membership_scores.append({
                    "alpha": alpha[idx].detach().cpu().tolist(),
                    "predicted_task": self.idx_to_task[pt],
                    "predicted_label": pred_label,
                })
        return loss, logits
