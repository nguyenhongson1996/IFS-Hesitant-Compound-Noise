from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertModel

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.models.multitask_cls import BaseMultitaskModel


class FQBertNoisyModel(BaseMultitaskModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask]):
        super().__init__()
        self.bert = bert
        self.hidden_size = bert.config.hidden_size
        self.dropout = nn.Dropout(bert.config.hidden_dropout_prob)
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_to_idx = {t.value: i for i, t in enumerate(tasks)}
        self.idx_to_task = {i: t.value for i, t in enumerate(tasks)}

        # Per-task query vector (Xavier-uniform init, paper Sec. 5.2)
        self.query_vectors = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.hidden_size, self.num_tasks)))

        # Per-task, per-channel Gaussian kernel parameters.
        # Centres init: (mu:-1.0, pi:0.0, nu:1.0). Stds init: 1.0.
        # Shape: (num_tasks, 3) - last dim order = (mu, pi, nu).
        init_centres = torch.tensor([[-1.0, 0.0, 1.0]] * self.num_tasks)
        self.kernel_centres = nn.Parameter(init_centres.clone())
        self.kernel_stds = nn.Parameter(torch.ones(self.num_tasks, 3))

        # Routing mode propagated by trainer at eval (None during training).
        self._eval_blind: Optional[bool] = None

    def _compute_ifs_triples(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_batch = {k: v for k, v in batch.items()
                       if k not in (TASK_NAME, LABELS, "true_task_name",
                                    "true_label", "sample_idx")}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        bert_out = self.bert(**input_batch)
        pooled = self.dropout(bert_out.pooler_output)  # (B, d)

        # Per-task affinity scores: s_{b,t} = pooled_b^T Q_t
        scores = pooled @ self.query_vectors  # (B, T)

        # Three Gaussian channels per task.
        # scores: (B, T, 1); centres/stds: (1, T, 3).
        s = scores.unsqueeze(-1)
        c = self.kernel_centres.unsqueeze(0)
        sigma = self.kernel_stds.unsqueeze(0).clamp_min(1e-3)
        evidence = torch.exp(-0.5 * ((s - c) / sigma) ** 2)  # (B, T, 3)

        # Channel-wise normalization (mu, pi, nu) per task.
        evidence = evidence / (evidence.sum(dim=-1, keepdim=True) + 1e-8)
        mu = evidence[..., 0]  # POSITIVE
        pi = evidence[..., 1]  # UNKNOWN / HESITATION
        nu = evidence[..., 2]  # NEGATIVE
        return mu, pi, nu

    def _fqbert_loss(self, mu: torch.Tensor, pi: torch.Tensor, nu: torch.Tensor,
                     labels: torch.Tensor, task_idxes: torch.Tensor,
                     lambda_hes: float = 1.0) -> torch.Tensor:
        B = mu.shape[0]
        rows = torch.arange(B, device=mu.device)

        mu_obs = mu[rows, task_idxes]  # (B,)
        nu_obs = nu[rows, task_idxes]  # (B,)
        pi_obs = pi[rows, task_idxes]  # (B,)

        # u_b: label-consistent evidence at observed head.
        y = labels.float()
        u = y * mu_obs + (1 - y) * nu_obs  # (B,)
        # v_b: model certainty at observed head.
        v = 1.0 - pi_obs  # (B,)
        L_task = -torch.log(u * v + 1e-8)  # (B,)

        # Hesitation loss
        c_per_task = torch.maximum(mu, nu)  # (B, T)
        # Mask out the observed task with -inf so it doesn't win the max.
        mask = torch.zeros_like(c_per_task)
        mask[rows, task_idxes] = float("-inf")
        c_masked = c_per_task + mask
        C_b = c_masked.max(dim=-1).values  # (B,)
        # Numerical guard: clamp C_b away from 1.
        C_b = C_b.clamp(max=1.0 - 1e-6)
        L_hes = -torch.log(1.0 - C_b + 1e-8)  # (B,)

        return (L_task + lambda_hes * L_hes).mean()

    @staticmethod
    def _confidence(mu: torch.Tensor, pi: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        return (1.0 - pi) * torch.maximum(mu, nu)  # (B, T)

    def forward(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        mu, pi, nu = self._compute_ifs_triples(batch)  # each (B, T)

        labels = batch[LABELS]
        task_names = batch[TASK_NAME]
        task_idxes_list = [self.task_to_idx[t] for t in task_names]
        task_idxes = torch.tensor(task_idxes_list, device=mu.device, dtype=torch.long)
        if IS_CUDA_AVAILABLE and not labels.is_cuda:
            labels = labels.cuda()

        loss = self._fqbert_loss(mu, pi, nu, labels, task_idxes,
                                 lambda_hes=getattr(self, "lambda_hes", 1.0))

        B = mu.shape[0]
        predictions = torch.zeros(B, 2, dtype=torch.float, device=mu.device)

        # K mode: oracle task; B mode: blind routing via conf.
        use_oracle = (not self.training) and (self._eval_blind is False)
        conf = self._confidence(mu, pi, nu)  # (B, T)

        for bi in range(B):
            t_pick = task_idxes_list[bi] if use_oracle else int(torch.argmax(conf[bi]).item())
            if mu[bi, t_pick] > nu[bi, t_pick]:
                predictions[bi] = torch.tensor([0.0, 1.0])
            else:
                predictions[bi] = torch.tensor([1.0, 0.0])

            if not self.training:
                pred_label = int(mu[bi, t_pick] > nu[bi, t_pick])
                self.all_task_names.append(task_names[bi])
                self.all_evaluated_labels.append(int(labels[bi].item()))
                self.all_membership_scores.append({
                    "mu": mu[bi].detach().cpu().tolist(),
                    "pi": pi[bi].detach().cpu().tolist(),
                    "nu": nu[bi].detach().cpu().tolist(),
                    "predicted_task": self.idx_to_task[t_pick],
                    "predicted_label": pred_label,
                })

        return loss, predictions
