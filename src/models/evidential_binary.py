import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def evidential_binary_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return F.softplus(evidence) + 1.0


def evidential_mse_loss(alpha: torch.Tensor,
                        labels: torch.Tensor,
                        task_idx: torch.Tensor,
                        kl_lambda: float = 0.0) -> torch.Tensor:
    B = alpha.size(0)
    device = alpha.device
    own = alpha[torch.arange(B, device=device), task_idx]  # [B, 2]
    S = own.sum(dim=-1, keepdim=True)  # [B, 1]
    p = own / S  # [B, 2]

    y_oh = F.one_hot(labels.long(), num_classes=2).float()  # [B, 2]
    err = (y_oh - p).pow(2)  # [B, 2]
    var = own * (S - own) / (S * S * (S + 1.0))  # [B, 2]
    mse = (err + var).sum(dim=-1).mean()

    if kl_lambda <= 0:
        return mse

    # alpha_tilde sets the target class's evidence to 1 (uniform), keeps
    # evidence on wrong classes - the KL pulls wrong-class evidence to 1.
    alpha_tilde = (1.0 - y_oh) * (own - 1.0) + 1.0  # [B, 2]
    S_tilde = alpha_tilde.sum(dim=-1)  # [B]
    K = 2
    log_gamma_K = math.lgamma(K)  # = 0 for K=2
    log_gamma_S_tilde = torch.lgamma(S_tilde)  # [B]
    log_gamma_alpha = torch.lgamma(alpha_tilde).sum(dim=-1)  # [B]
    digamma_alpha = torch.digamma(alpha_tilde)  # [B, 2]
    digamma_S_tilde = torch.digamma(S_tilde).unsqueeze(-1)  # [B, 1]
    first = log_gamma_S_tilde - log_gamma_K - log_gamma_alpha
    second = ((alpha_tilde - 1.0) * (digamma_alpha - digamma_S_tilde)).sum(dim=-1)
    kl = (first + second).mean()
    return mse + kl_lambda * kl


def evidential_predict(alpha: torch.Tensor) -> torch.Tensor:
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / S


def evidential_route_blind(alpha: torch.Tensor) -> torch.Tensor:
    p = evidential_predict(alpha)  # [B, T, 2]
    max_per_task = p.max(dim=-1).values  # [B, T]
    return max_per_task.argmax(dim=-1)  # [B]


class EvidentialBinaryHead(nn.Module):

    def __init__(self, hidden_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.evidence = nn.Linear(hidden_dim, num_tasks * 2)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        B = e.size(0)
        ev = self.evidence(e).view(B, self.num_tasks, 2)
        return evidential_binary_alpha(ev)
