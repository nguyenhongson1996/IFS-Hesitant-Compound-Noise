from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME


class TfidfMLPEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        x = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            x = x.cuda()
        return self.net(x)


class FQEncoderModel(nn.Module):

    def __init__(self, encoder: nn.Module, hidden_dim: int, tasks: List[str],
                 lambda_hes: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(tasks)}
        self.idx_to_task = {i: t for i, t in enumerate(tasks)}
        self.lambda_hes = lambda_hes

        # Per-task query Q_t in R^d (Xavier-uniform init, FQ paper Sec. 5.2)
        self.query_vectors = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(self.hidden_dim, self.num_tasks)))

        # Per-task per-channel Gaussian kernel params.
        # Centres init: (mu:-1.0, pi:0.0, nu:1.0). Stds init: 1.0.
        # Shape: (num_tasks, 3) - last dim order = (mu, pi, nu).
        init_centres = torch.tensor([[-1.0, 0.0, 1.0]] * self.num_tasks)
        self.kernel_centres = nn.Parameter(init_centres.clone())
        self.kernel_stds = nn.Parameter(torch.ones(self.num_tasks, 3))

        # Routing mode (None=training, True=blind eval, False=oracle eval).
        self._eval_blind: Optional[bool] = None

        self.all_task_names: List[str] = []
        self.all_evaluated_labels: List[int] = []
        self.all_membership_scores: List[Dict] = []

    def reset_eval_logs(self):
        self.all_task_names.clear()
        self.all_evaluated_labels.clear()
        self.all_membership_scores.clear()

    def _compute_ifs_triples(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(batch)
        if IS_CUDA_AVAILABLE:
            embeddings = embeddings.cuda()
        # Per-task affinity scores: s_{b,t} = pooled_b^T Q_t   shape (B, T).
        scores = embeddings @ self.query_vectors
        # Three Gaussian channels per task.
        s = scores.unsqueeze(-1)  # (B, T, 1).
        c = self.kernel_centres.unsqueeze(0)  # (1, T, 3).
        sigma = self.kernel_stds.unsqueeze(0).clamp_min(1e-3)
        evidence = torch.exp(-0.5 * ((s - c) / sigma) ** 2)  # (B, T, 3).
        # Channel-wise normalisation.
        evidence = evidence / (evidence.sum(dim=-1, keepdim=True) + 1e-8)
        return evidence[..., 0], evidence[..., 1], evidence[..., 2]  # mu, pi, nu.

    def _fq_loss(self, mu, pi, nu, labels, task_idxes) -> torch.Tensor:
        B = mu.shape[0]
        rows = torch.arange(B, device=mu.device)
        mu_obs = mu[rows, task_idxes]
        nu_obs = nu[rows, task_idxes]
        pi_obs = pi[rows, task_idxes]
        y = labels.float()
        u = y * mu_obs + (1 - y) * nu_obs  # label-consistent evidence.
        v = 1.0 - pi_obs  # model certainty.
        L_task = -torch.log(u * v + 1e-8)
        # Hesitation: -log(1 - max_{t!=t_b} max(mu_t, nu_t)).
        c_per_task = torch.maximum(mu, nu)
        mask = torch.zeros_like(c_per_task)
        mask[rows, task_idxes] = float("-inf")
        C_b = (c_per_task + mask).max(dim=-1).values.clamp(max=1.0 - 1e-6)
        L_hes = -torch.log(1.0 - C_b + 1e-8)
        return (L_task + self.lambda_hes * L_hes).mean()

    @staticmethod
    def _confidence(mu, pi, nu):
        return (1.0 - pi) * torch.maximum(mu, nu)

    def forward(self, batch: Dict[str, Any], blind_override: Optional[bool] = None):
        mu, pi, nu = self._compute_ifs_triples(batch)

        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE and not labels.is_cuda:
            labels = labels.cuda()
        task_names = batch[TASK_NAME]
        task_idxes_list = [self.task_to_idx[t] for t in task_names]
        task_idxes = torch.tensor(task_idxes_list, device=mu.device, dtype=torch.long)

        loss = self._fq_loss(mu, pi, nu, labels, task_idxes)

        # Routing for predictions.
        if blind_override is not None:
            use_oracle = (not self.training) and (not blind_override)
        else:
            use_oracle = (not self.training) and (self._eval_blind is False)
        conf = self._confidence(mu, pi, nu)

        B = mu.shape[0]
        predictions = torch.zeros(B, 2, dtype=torch.float, device=mu.device)
        for b in range(B):
            t_pick = task_idxes_list[b] if use_oracle else int(torch.argmax(conf[b]).item())
            if mu[b, t_pick] > nu[b, t_pick]:
                predictions[b] = torch.tensor([0.0, 1.0])
            else:
                predictions[b] = torch.tensor([1.0, 0.0])

            if not self.training:
                pred_label = int(mu[b, t_pick] > nu[b, t_pick])
                self.all_task_names.append(task_names[b])
                self.all_evaluated_labels.append(int(labels[b].item()))
                self.all_membership_scores.append({
                    "mu": mu[b].detach().cpu().tolist(),
                    "pi": pi[b].detach().cpu().tolist(),
                    "nu": nu[b].detach().cpu().tolist(),
                    "predicted_task": self.idx_to_task[t_pick],
                    "predicted_label": pred_label,
                })

        return loss, predictions
