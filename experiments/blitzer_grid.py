import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.common.seed_utils import set_seed, seeded_generator
from src.data_processors.blitzer_dataset import (
    BLITZER_TASKS, BlitzerDataset, NoisyBlitzerDataset,
)
from src.data_processors.noisy_ensemble_dataset import TRUE_TASK_NAME


def _make_mlp(input_dim: int, hidden_dims: List[int], output_dim: int,
              dropout: float = 0.0) -> nn.Sequential:
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _MTDNNModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleDict({
            t: nn.Linear(encoder_output, 2) for t in self.tasks
        })

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        # All-task logits for blind routing diagnostics (per-sample [K, 2])
        all_logits = torch.stack([
            self.classifiers[t](e) for t in self.tasks
        ], dim=1)  # [B, K, 2]
        # Pick the assigned-task logit for the loss (on observed task)
        labels = batch[LABELS]
        task_idx = torch.tensor(
            [self.task_to_idx[t] for t in batch[TASK_NAME]],
            dtype=torch.long,
        )
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
            task_idx = task_idx.cuda()
        own_logits = all_logits[torch.arange(all_logits.size(0)), task_idx]  # [B, 2]
        per_sample_ce = nn.functional.cross_entropy(own_logits, labels, reduction="none")
        return own_logits, per_sample_ce, all_logits


class _PooledModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)  # kept for per-task eval bookkeeping only
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.classifier = nn.Linear(encoder_output, 2)

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        logits = self.classifier(e)  # [B, 2]
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        per_sample_ce = nn.functional.cross_entropy(logits, labels,
                                                    reduction="none")
        return logits, per_sample_ce


class _CoteachingModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.model_a = _MTDNNModel(input_dim, encoder_hidden, encoder_output,
                                   tasks, dropout)
        self.model_b = _MTDNNModel(input_dim, encoder_hidden, encoder_output,
                                   tasks, dropout)


class _TaskIDModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleDict({
            t: nn.Linear(encoder_output, 2) for t in self.tasks
        })
        self.task_id_head = nn.Linear(encoder_output, len(self.tasks))

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        all_logits = torch.stack([
            self.classifiers[t](e) for t in self.tasks
        ], dim=1)  # [B, K, 2]
        task_id_logits = self.task_id_head(e)  # [B, K]
        labels = batch[LABELS]
        task_idx = torch.tensor(
            [self.task_to_idx[t] for t in batch[TASK_NAME]],
            dtype=torch.long,
        )
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
            task_idx = task_idx.cuda()
        own_logits = all_logits[torch.arange(all_logits.size(0)), task_idx]
        per_sample_ce = nn.functional.cross_entropy(own_logits, labels, reduction="none")
        return own_logits, per_sample_ce, all_logits, task_id_logits


class _IFSHead(nn.Module):

    def __init__(self, input_dim: int, num_tasks: int,
                 head_type: str = "sigmoid",
                 use_decoupled: bool = False) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        # Backward-compat: use_decoupled=True overrides head_type="sigmoid".
        if use_decoupled and head_type == "sigmoid":
            head_type = "factored"
        if head_type not in ("sigmoid", "factored", "softmax3"):
            raise ValueError(f"unknown head_type {head_type!r}")
        self.head_type = head_type
        self.use_decoupled = (head_type != "sigmoid")  # kept for callers
        n_out = 2 if head_type == "sigmoid" else 3
        self.proj = nn.Linear(input_dim, num_tasks * n_out)

    def forward(self, e: torch.Tensor):
        B = e.size(0)
        if self.head_type == "factored":
            logits = self.proj(e).view(B, self.num_tasks, 3)
            m, n, p = logits[:, :, 0], logits[:, :, 1], logits[:, :, 2]
            mn_soft = torch.softmax(torch.stack([m, n], dim=-1), dim=-1)
            mu_s = mn_soft[..., 0]
            nu_s = mn_soft[..., 1]
            pi = torch.sigmoid(p)
            mu = (1.0 - pi) * mu_s
            nu = (1.0 - pi) * nu_s
            return torch.stack([mu, pi, nu], dim=-1), m, n
        if self.head_type == "softmax3":
            logits = self.proj(e).view(B, self.num_tasks, 3)
            m, n, p = logits[:, :, 0], logits[:, :, 1], logits[:, :, 2]
            mnp_soft = torch.softmax(torch.stack([m, n, p], dim=-1), dim=-1)
            mu = mnp_soft[..., 0]
            nu = mnp_soft[..., 1]
            pi = mnp_soft[..., 2]
            # Return m, n so the decoupled class loss path (which expects
            # cached pre-softmax class logits) keeps working.
            return torch.stack([mu, pi, nu], dim=-1), m, n
        # sigmoid (legacy)
        logits = self.proj(e).view(B, self.num_tasks, 2)
        l_class = logits[:, :, 0]
        l_pi = logits[:, :, 1]
        pi = torch.sigmoid(l_pi)
        mu_bin = torch.sigmoid(l_class)
        nu_bin = 1.0 - mu_bin
        mu = (1.0 - pi) * mu_bin
        nu = (1.0 - pi) * nu_bin
        return torch.stack([mu, pi, nu], dim=-1), None, None


class _IFSModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0,
                 use_decoupled: bool = False,
                 head_type: str = "sigmoid") -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        # Backward-compat: use_decoupled=True maps to head_type="factored".
        if use_decoupled and head_type == "sigmoid":
            head_type = "factored"
        self.head_type = head_type
        self.use_decoupled = (head_type != "sigmoid")
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.head = _IFSHead(encoder_output, num_tasks=len(self.tasks),
                             head_type=head_type)
        # Cache for the decoupled class loss (m, n logits per task).
        self._last_m = None
        self._last_n = None

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        triples, m, n = self.head(e)
        if self.use_decoupled:
            self._last_m = m
            self._last_n = n
        return triples  # [B, K, 3]


class _IFSLinearHybridModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.head = _IFSHead(encoder_output, num_tasks=len(self.tasks))
        self.task_id_head = nn.Linear(encoder_output, len(self.tasks))

    def forward(self, batch: Dict[str, Any]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        triples = self.head(e)
        task_id_logits = self.task_id_head(e)
        return triples, task_id_logits


def _ifs_loss_warmup(triples: torch.Tensor, labels: torch.Tensor,
                     task_idx: torch.Tensor, lam: float = 1.0,
                     bootstrap_beta: Optional[float] = None,
                     warmup_discard_ratio: float = 0.0) -> torch.Tensor:
    B, K, _ = triples.shape
    eps = 1e-8
    own = triples[torch.arange(B), task_idx]  # [B, 3]
    mu, pi, nu = own[:, 0], own[:, 1], own[:, 2]
    sat = labels.float() * mu + (1.0 - labels.float()) * nu
    if bootstrap_beta is not None:
        with torch.no_grad():
            y_pred = (mu > nu).long()
        sat_pred = y_pred.float() * mu + (1.0 - y_pred.float()) * nu
        main = -(bootstrap_beta * torch.log(sat + eps)
                 + (1.0 - bootstrap_beta) * torch.log(sat_pred + eps))
    else:
        main = -torch.log(sat + eps)
    if K > 1:
        mask = torch.ones(B, K, dtype=torch.bool, device=triples.device)
        mask[torch.arange(B), task_idx] = False
        pi_others = triples[..., 1][mask].view(B, K - 1)
        barrier = -torch.log(pi_others + eps).mean(dim=1)
        per_sample = main + lam / max(K - 1, 1) * (K - 1) * barrier
    else:
        per_sample = main
    if warmup_discard_ratio and warmup_discard_ratio > 0.0 and B > 1:
        n_discard = int(B * warmup_discard_ratio)
        n_keep = max(1, B - n_discard)
        if n_keep < B:
            keep_idx = per_sample.detach().argsort()[:n_keep]
            per_sample = per_sample[keep_idx]
    return per_sample.mean()


def _ifs_loss_hesitant(triples: torch.Tensor, labels: torch.Tensor,
                       task_idx: torch.Tensor, lam: float, w_min: float,
                       bootstrap_beta: Optional[float] = None,
                       weight_norm: str = "mean",
                       weight_norm_target: float = 0.5,
                       trust_signal: str = "ifs",
                       trust_tnorm: str = "product",
                       trust_no_gate: bool = False,
                       trust_no_detach: bool = False,
                       decoupled_hesitant_beta: Optional[float] = None,
                       no_lpred: bool = False,
                       l_pred_alpha: float = 1.0,
                       bootstrap_hesitant: bool = False,
                       bayes_trust: bool = False,
                       head_type: str = "sigmoid",
                       ) -> torch.Tensor:
    import math
    B, K, _ = triples.shape
    eps = 1e-8
    own = triples[torch.arange(B), task_idx]  # [B, 3]
    mu_obs, pi_obs, nu_obs = own[:, 0], own[:, 1], own[:, 2]
    sat_obs = labels.float() * mu_obs + (1.0 - labels.float()) * nu_obs
    # trust weight (detached by default; ablation flags override)
    if trust_signal == "ifs":
        if bayes_trust:
            # Bayes-corrected: under factored head sat=(1-pi)*sat_tilde, so
            # w = (1-pi)*sat_tilde = sat directly. Avoids the (1-pi)^2 double-count.
            if head_type != "factored":
                raise ValueError(
                    f"--bayes_trust requires head_type='factored', got {head_type!r}")
            if trust_tnorm != "product":
                raise ValueError(
                    f"--bayes_trust only defined for trust_tnorm='product', got {trust_tnorm!r}")
            w_raw = sat_obs
        else:
            a = 1.0 - pi_obs
            b = sat_obs
            if trust_tnorm == "product":
                w_raw = a * b
            elif trust_tnorm == "min":
                w_raw = torch.minimum(a, b)
            elif trust_tnorm == "lukasiewicz":
                w_raw = torch.clamp(a + b - 1.0, min=0.0)
            elif trust_tnorm == "hamacher":
                denom = 2.0 - (a + b - a * b) + eps
                w_raw = a * b / denom
            else:
                raise ValueError(f"unknown trust_tnorm {trust_tnorm!r}")
        w = w_raw if trust_no_detach else w_raw.detach()
    elif trust_signal == "entropy":
        denom = mu_obs + nu_obs + eps
        p1 = mu_obs / denom
        p0 = nu_obs / denom
        ent = -(p1 * torch.log(p1 + eps) + p0 * torch.log(p0 + eps))
        w = (1.0 - ent / math.log(2.0)).detach()
    elif trust_signal == "margin":
        w = torch.abs(mu_obs - nu_obs).detach()
    elif trust_signal == "loss":
        ce = torch.where(labels.float() > 0.5,
                         -torch.log(mu_obs + eps),
                         -torch.log(nu_obs + eps))
        w = torch.exp(-ce).detach()
    else:
        raise ValueError(f"unknown trust_signal {trust_signal!r}")
    if weight_norm == "mean" and B > 0:
        w_mean = w.mean()
        if w_mean.item() > eps:
            w = w * (weight_norm_target / w_mean)
    elif weight_norm == "rank" and B > 1:
        ranks = w.argsort().argsort().float()
        w = w_min + (1.0 - w_min) * ranks / (B - 1)
    elif weight_norm not in ("none", "mean", "rank"):
        raise ValueError(f"unknown weight_norm {weight_norm!r}")
    if trust_no_detach:
        # In-place clamp would corrupt the autograd graph if w is gradient-tracked.
        w = w.clamp(w_min, 1.0)
    else:
        w = w.clamp_(w_min, 1.0)
    # Ablation: replace w with constant 1.0 to remove the trust gate entirely.
    # The L_pred branch (weighted (1-w)) then receives weight 0 and only L_sup runs.
    if trust_no_gate:
        w = torch.ones_like(w)
    # Self-predicted task and label (detached for selection)
    # conf is head-aware: under factored head, max(mu,nu)=(1-pi)*max(mu_tilde,nu_tilde) already,
    # so an extra (1-pi) factor would square - same correction as the trust weight.
    with torch.no_grad():
        max_class = torch.maximum(triples[..., 0], triples[..., 2])  # [B, K]
        if head_type == "factored":
            conf = max_class
        else:
            conf = (1.0 - triples[..., 1]) * max_class  # [B, K]
        hat_t = conf.argmax(dim=1)  # [B]
        own_self = triples[torch.arange(B), hat_t]  # [B, 3]
        hat_y = (own_self[:, 0] > own_self[:, 2]).long()
    # Observed pair NLL - optionally Bootstrap-softened at the observed task.
    if bootstrap_beta is not None:
        with torch.no_grad():
            y_pred_at_obs = (mu_obs > nu_obs).long()
        sat_pred_at_obs = (y_pred_at_obs.float() * mu_obs
                           + (1.0 - y_pred_at_obs.float()) * nu_obs)
        main_obs = -(bootstrap_beta * torch.log(sat_obs + eps)
                     + (1.0 - bootstrap_beta) * torch.log(sat_pred_at_obs + eps))
    else:
        main_obs = -torch.log(sat_obs + eps)
    sat_pred = hat_y.float() * own_self[:, 0] + (1.0 - hat_y.float()) * own_self[:, 2]
    main_pred = -torch.log(sat_pred + eps)
    if K > 1:
        # Barrier on pi at non-assigned heads (assigned = task_idx for observed, hat_t for self)
        mask_obs = torch.ones(B, K, dtype=torch.bool, device=triples.device)
        mask_obs[torch.arange(B), task_idx] = False
        bar_obs = -torch.log(triples[..., 1][mask_obs].view(B, K - 1) + eps).mean(dim=1)
        mask_pr = torch.ones(B, K, dtype=torch.bool, device=triples.device)
        mask_pr[torch.arange(B), hat_t] = False
        bar_pred = -torch.log(triples[..., 1][mask_pr].view(B, K - 1) + eps).mean(dim=1)
        L_obs_per = main_obs + lam * bar_obs
        L_pred_per = main_pred + lam * bar_pred
    else:
        L_obs_per = main_obs
        L_pred_per = main_pred
    # CE-on-3-softmax design: drop L_pred entirely. Loss is just
    # trust-weighted (class CE + barrier) at the observed task.
    # Class margin is collapse-immune since noisy samples (low w) contribute
    # ~0 class gradient. Combined with use_decoupled_head=True (3-logit IFS-v3
    # head) and warmup_epochs=0 to use this loss for all epochs.
    if no_lpred:
        eps_p = 1e-8
        base = w * L_obs_per
        return base.mean()

    # IFS-BSH bootstrap-safe hesitant
    if bootstrap_hesitant:
        eps_b = 1e-8
        # Recover class softmax (no pi factor): mu_tilde = mu/(mu+nu), nu_tilde = nu/(mu+nu).
        mu_all = triples[..., 0]  # [B, K]
        nu_all = triples[..., 2]  # [B, K]
        pi_all = triples[..., 1]  # [B, K]
        denom_class = (mu_all + nu_all).clamp_min(eps_b)
        mu_s_all = mu_all / denom_class  # [B, K]  P(y=1 | task t)
        nu_s_all = nu_all / denom_class  # [B, K]  P(y=0 | task t)
        # Class probabilities at the observed task.
        mu_s_obs = mu_s_all[torch.arange(B), task_idx]  # [B]
        nu_s_obs = mu_s_all.new_zeros(B);
        nu_s_obs = nu_s_all[torch.arange(B), task_idx]
        pi_obs = pi_all[torch.arange(B), task_idx]
        labels_f_ = labels.float() if labels.dtype != torch.float else labels
        # Soft target at the observed task: w * onehot + (1 - w) * uniform.
        prob_obs_observed = labels_f_ * mu_s_obs + (1.0 - labels_f_) * nu_s_obs
        L_class_obs = (- w * torch.log(prob_obs_observed + eps_b)
                       - 0.5 * (1.0 - w) * torch.log(mu_s_obs + eps_b)
                       - 0.5 * (1.0 - w) * torch.log(nu_s_obs + eps_b))  # [B]
        # Trust-weighted pi push at t_tilde
        L_pi_obs = w * (-torch.log(1.0 - pi_obs + eps_b))  # [B]
        # Predicted task t_hat (= hat_t, already detached above for L_pred computation)
        pi_pred = pi_all[torch.arange(B), hat_t]
        mu_s_pred = mu_s_all[torch.arange(B), hat_t]
        nu_s_pred = nu_s_all[torch.arange(B), hat_t]
        y_hat_pred = (mu_s_pred > nu_s_pred).float().detach()
        prob_pred_predicted = y_hat_pred * mu_s_pred + (1.0 - y_hat_pred) * nu_s_pred
        L_class_pred = -torch.log(prob_pred_predicted + eps_b)  # [B]
        L_pi_pred = -torch.log(1.0 - pi_pred + eps_b)  # [B]
        agree_f = (hat_t == task_idx).float()  # [B]
        L_disagree = (1.0 - agree_f) * (L_class_pred + L_pi_pred)  # [B]
        # Barrier on S = {j != t_tilde} union (when agree=0: union excl t_hat) - exclude t_hat
        # from the barrier set when disagree.
        in_obs = torch.zeros(B, K, dtype=torch.bool, device=triples.device)
        in_obs[torch.arange(B), task_idx] = True
        in_pred = torch.zeros(B, K, dtype=torch.bool, device=triples.device)
        in_pred[torch.arange(B), hat_t] = True
        exclude_pred = in_pred & (agree_f.unsqueeze(1).bool() == False)
        in_barrier = ~(in_obs | exclude_pred)  # [B, K]
        log_pi = -torch.log(pi_all + eps_b)
        barrier_sum_b = (log_pi * in_barrier.float()).sum(dim=1)
        barrier_count_b = in_barrier.float().sum(dim=1).clamp_min(1.0)
        L_barrier = barrier_sum_b / barrier_count_b
        return (L_class_obs + L_pi_obs + L_disagree + lam * L_barrier).mean()
    # IFS-v3 decoupled-hesitant: when t_pred != t_obs, dampen L_pred by beta.
    if decoupled_hesitant_beta is not None:
        agree = (hat_t == task_idx).float()
        # alpha = (1-w) if agree else beta*(1-w)
        alpha = agree * (1.0 - w) + (1.0 - agree) * decoupled_hesitant_beta * (1.0 - w)
        return (w * L_obs_per + alpha * L_pred_per).mean()
    # Ablation: when --trust_no_gate AND L_pred is enabled, return the
    # ungated sum L_obs + alpha*L_pred (no (1-w) factor on L_pred).
    if trust_no_gate:
        return (L_obs_per + l_pred_alpha * L_pred_per).mean()
    return (w * L_obs_per + (1.0 - w) * l_pred_alpha * L_pred_per).mean()


from src.models.evidential_binary import (
    EvidentialBinaryHead,
    evidential_mse_loss,
    evidential_predict,
    evidential_route_blind,
)


class _EvidentialModel(nn.Module):

    def __init__(self, input_dim: int, encoder_hidden: List[int],
                 encoder_output: int, tasks: List[str],
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.encoder = _make_mlp(
            input_dim=input_dim,
            hidden_dims=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout,
        )
        self.head = EvidentialBinaryHead(encoder_output, num_tasks=len(self.tasks))

    def alpha(self, batch: Dict[str, Any]) -> torch.Tensor:
        feats = batch["input_features"]
        if IS_CUDA_AVAILABLE:
            feats = feats.cuda()
        e = self.encoder(feats)
        return self.head(e)  # [B, T, 2]

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.alpha(batch)


def _evidential_loss(alpha: torch.Tensor, labels: torch.Tensor,
                     task_idx: torch.Tensor, lam: float = 1.0,
                     kl_lambda: float = 0.0) -> torch.Tensor:
    return evidential_mse_loss(alpha, labels, task_idx, kl_lambda=kl_lambda)


def _predict_known_blind(model: nn.Module, ds: BlitzerDataset,
                         tasks: List[str], batch_size: int,
                         model_kind: str, hybrid_alpha: float = 0.5,
                         return_preds: bool = False,
                         ):
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    ds.set_mode(Mode.VALIDATION)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=ds.collate_fn)
    model.eval()
    correct_known = correct_blind = 0
    per_known = {t: [0, 0] for t in tasks}  # [correct, total]
    per_blind = {t: [0, 0] for t in tasks}
    preds_log = {"true_task": [], "true_label": [], "pred_known": [], "pred_blind": []} if return_preds else None
    with torch.no_grad():
        for batch in loader:
            labels = batch[LABELS]
            if IS_CUDA_AVAILABLE:
                labels = labels.cuda()
            true_tasks = batch[TASK_NAME]
            true_t_idx = torch.tensor(
                [task_to_idx[t] for t in true_tasks], dtype=torch.long,
            )
            if IS_CUDA_AVAILABLE:
                true_t_idx = true_t_idx.cuda()
            B = labels.size(0)
            if model_kind == "ifs":
                triples = model(batch)  # [B, K, 3]
                # Known: read off triple at true_t_idx
                own = triples[torch.arange(B), true_t_idx]
                pred_known = (own[:, 0] > own[:, 2]).long()
                # Blind: confidence-routed (head-aware to avoid (1-pi)^2 double-count
                # under factored head, where max(mu,nu)=(1-pi)*max(mu_tilde,nu_tilde) already)
                _max_class = torch.maximum(triples[..., 0], triples[..., 2])
                if getattr(model, "head_type", "sigmoid") == "factored":
                    conf = _max_class
                else:
                    conf = (1.0 - triples[..., 1]) * _max_class
                hat_t = conf.argmax(dim=1)
                own_blind = triples[torch.arange(B), hat_t]
                pred_blind = (own_blind[:, 0] > own_blind[:, 2]).long()
            elif model_kind == "evidential":
                # Evidential head: predict from per-task alpha.
                alpha = model(batch)  # [B, T, 2]
                p = evidential_predict(alpha)  # [B, T, 2]
                # Known: argmax(p[true_task])
                p_known = p[torch.arange(B), true_t_idx]  # [B, 2]
                pred_known = p_known.argmax(dim=1)
                # Blind: route to argmax_t max_k p_{t,k}, then argmax_k
                hat_t = evidential_route_blind(alpha)  # [B]
                p_blind = p[torch.arange(B), hat_t]  # [B, 2]
                pred_blind = p_blind.argmax(dim=1)
            elif model_kind == "ifs_hybrid":
                triples, tid_logits = model(batch)
                # Known: triple at true task
                own = triples[torch.arange(B), true_t_idx]
                pred_known = (own[:, 0] > own[:, 2]).long()
                # Blind: alpha-blend of IFS conf and softmax task-id probability
                conf_ifs = (1.0 - triples[..., 1]) * torch.maximum(
                    triples[..., 0], triples[..., 2],
                )
                p_tid = torch.softmax(tid_logits, dim=-1)
                alpha = hybrid_alpha
                conf_blend = alpha * conf_ifs + (1.0 - alpha) * p_tid
                hat_t = conf_blend.argmax(dim=1)
                own_blind = triples[torch.arange(B), hat_t]
                pred_blind = (own_blind[:, 0] > own_blind[:, 2]).long()
            elif model_kind == "task_identity":
                _, _, all_logits, task_id_logits = model(batch)
                # Known: pick true task head
                own_logits = all_logits[torch.arange(B), true_t_idx]
                pred_known = own_logits.argmax(dim=1)
                # Blind: route via task-id head
                hat_t = task_id_logits.argmax(dim=1)
                pred_blind = all_logits[torch.arange(B), hat_t].argmax(dim=1)
            elif model_kind == "coteaching":
                # Inference uses model A only (Han et al. 2018 protocol)
                _, _, all_logits = model.model_a(batch)
                own_logits = all_logits[torch.arange(B), true_t_idx]
                pred_known = own_logits.argmax(dim=1)
                # Blind: max-softmax routing (same as mtdnn)
                probs = torch.softmax(all_logits, dim=-1)
                max_conf = probs.max(dim=-1).values
                hat_t = max_conf.argmax(dim=1)
                pred_blind = all_logits[torch.arange(B), hat_t].argmax(dim=1)
            elif model_kind == "pooled":
                # Single classifier - no routing. Known and blind are
                # identical because there is no per-task structure.
                logits, _ = model(batch)
                pred = logits.argmax(dim=1)
                pred_known = pred
                pred_blind = pred
            else:  # mtdnn / standard_noisy
                _, _, all_logits = model(batch)  # [B, K, 2]
                # Known: pick true task head
                own_logits = all_logits[torch.arange(B), true_t_idx]
                pred_known = own_logits.argmax(dim=1)
                # Blind: pick task with max softmax confidence
                probs = torch.softmax(all_logits, dim=-1)
                max_conf = probs.max(dim=-1).values  # [B, K]
                hat_t = max_conf.argmax(dim=1)
                pred_blind = all_logits[torch.arange(B), hat_t].argmax(dim=1)
            correct_known += (pred_known == labels).sum().item()
            correct_blind += (pred_blind == labels).sum().item()
            for i in range(B):
                t = true_tasks[i]
                per_known[t][1] += 1
                per_blind[t][1] += 1
                if pred_known[i].item() == labels[i].item():
                    per_known[t][0] += 1
                if pred_blind[i].item() == labels[i].item():
                    per_blind[t][0] += 1
                if preds_log is not None:
                    preds_log["true_task"].append(t)
                    preds_log["true_label"].append(int(labels[i].item()))
                    preds_log["pred_known"].append(int(pred_known[i].item()))
                    preds_log["pred_blind"].append(int(pred_blind[i].item()))
    n = len(ds)
    acc_known = correct_known / n
    acc_blind = correct_blind / n
    pk = {t: round(c / max(tot, 1), 4) for t, (c, tot) in per_known.items()}
    pb = {t: round(c / max(tot, 1), 4) for t, (c, tot) in per_blind.items()}
    if return_preds:
        return acc_known, acc_blind, pk, pb, preds_log
    return acc_known, acc_blind, pk, pb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=("mtdnn", "standard_noisy",
                            "ifs_hesitant", "ifs_hesitant_bootstrap",
                            "ifs_hesitant_hybrid",
                            "evidential", "bootstrapping", "gce",
                            "high_loss_discard", "task_identity",
                            "coteaching", "pooled_ce",
                            "pooled_gce", "pooled_bootstrapping",
                            "pooled_hld", "forward_correction",
                            "mtlnl", "excessmtl"))
    p.add_argument("--hybrid_alpha", type=float, default=0.5,
                   help="Hybrid inference: route by alpha*conf_IFS + (1-alpha)*p_taskid.")
    p.add_argument("--hybrid_lambda_taskid", type=float, default=1.0,
                   help="Hybrid training: weight on auxiliary task-id CE.")
    p.add_argument("--bootstrap_beta", type=float, default=0.95,
                   help="Bootstrapping soft-target weight on observed label.")
    p.add_argument("--gce_q", type=float, default=0.7,
                   help="GCE q parameter; q->0 = CE, q=1 = MAE.")
    p.add_argument("--discard_ratio", type=float, default=0.3,
                   help="High-loss-discard baseline: per-batch drop top-k fraction.")
    p.add_argument("--warmup_discard_ratio", type=float, default=0.0,
                   help="IFS warmup-phase per-batch high-loss discard ratio. "
                        "Drops the top-K%% per-sample warmup losses. Mirror of "
                        "the BERT NoisyRobustIFSModel intervention. 0 disables.")
    p.add_argument("--kl_anneal_epochs", type=int, default=10,
                   help="Evidential: anneal kl_lambda 0->1 across this many "
                        "epochs (Sensoy 2018 protocol). 0 disables KL term.")
    p.add_argument("--epsilon_t", type=float, default=0.0)
    p.add_argument("--class_noise_rho", type=float, default=0.0)
    p.add_argument("--class_noise_rho_indep", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--encoder_hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--encoder_output", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_features", type=int, default=10000)
    p.add_argument("--ngram_max", type=int, default=2,
                   help="upper end of TF-IDF ngram range (1..N).")
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="for ifs_hesitant: phase-1 warmup duration")
    p.add_argument("--w_min", type=float, default=0.1)
    p.add_argument("--weight_norm", choices=["none", "mean", "rank"],
                   default="mean",
                   help="Per-batch trust-weight normalisation in the hesitant loss.")
    p.add_argument("--weight_norm_target", type=float, default=0.5,
                   help="Target batch-mean trust weight when --weight_norm=mean.")
    p.add_argument("--trust_signal", choices=["ifs", "entropy", "margin", "loss"],
                   default="ifs",
                   help="Per-sample trust-weight signal in the hesitant loss.")
    p.add_argument("--trust_tnorm",
                   choices=["product", "min", "lukasiewicz", "hamacher"],
                   default="product",
                   help="T-norm used to aggregate (1-pi) and sat in the IFS trust weight (only when --trust_signal=ifs).")
    p.add_argument("--trust_no_gate", action="store_true",
                   help="Ablation: force w=1 (constant), removing the trust gate. With --no_lpred, loss = L_sup; without --no_lpred, loss = L_sup + l_pred_alpha * L_pred (unweighted both branches).")
    p.add_argument("--l_pred_alpha", type=float, default=1.0,
                   help="Coefficient on the L_pred branch. Default 1.0 reproduces "
                        "the legacy small-arch loss (1-w)*L_pred. Set <1 to dampen L_pred.")
    p.add_argument("--trust_no_detach", action="store_true",
                   help="Ablation: do NOT detach the trust weight from the autograd graph; gradients flow back through (1-pi) and sat.")
    p.add_argument("--lambda_other", type=float, default=1.0)
    p.add_argument("--use_decoupled_head", action="store_true",
                   help="IFS-v3: 3 logits per task (m, n, p). softmax(m,n)=class, sigma(p)=pi. "
                        "Equivalent to --head_type factored; kept for backward compat.")
    p.add_argument("--head_type", choices=["sigmoid", "factored", "softmax3"],
                   default=None,
                   help="Parameterisation of the per-task (mu,pi,nu) triple. "
                        "'sigmoid' (default): 2-logit legacy head. "
                        "'factored': 3-logit (mu_s,nu_s)=softmax(m,n), pi=sigma(p), then compress. "
                        "'softmax3': 3-logit (mu,pi,nu)=softmax(m,n,p) directly (ablation). "
                        "If unset, falls back to 'factored' when --use_decoupled_head "
                        "is given, otherwise 'sigmoid'.")
    p.add_argument("--no_lpred", action="store_true",
                   help="CE-on-3-softmax design: drop L_pred entirely so the "
                        "loss is just trust-weighted (class CE + barrier). "
                        "Class margin is collapse-immune. Use with "
                        "--use_decoupled_head --warmup_epochs 0.")
    p.add_argument("--bootstrap_hesitant", action="store_true",
                   help="IFS-BSH (bootstrap-safe hesitant): bootstrap CE at "
                        "observed task with beta=w, trust-weighted pi push at t_tilde, "
                        "IFS class+pi loss at t_hat only when t_hat!=t_tilde, barrier "
                        "excludes t_hat on disagree. Use with --use_decoupled_head.")
    p.add_argument("--decoupled_hesitant_beta", type=float, default=None,
                   help="Dampening on L_pred when t_pred != t_obs (decoupled hesitant). "
                        "Default None = canonical hesitant. Try 0.1 for slight push.")
    p.add_argument("--bayes_trust", action="store_true",
                   help="Bayes-corrected trust weight: under factored head, the "
                        "sat_t_tilde(y_tilde) coordinate already contains the (1-pi) factor "
                        "(since mu=(1-pi)*mu_tilde), so w = (1-pi)*sat = (1-pi)^2*sat_tilde "
                        "double-counts task-fit. With this flag and head_type=factored, "
                        "we use w = sat directly = (1-pi)*sat_tilde, matching the paper's "
                        "Bayesian factorisation literally. Requires --head_type factored.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dataset", choices=["blitzer", "synthetic"], default="blitzer", help="Dataset to use.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.noise_seed)
    t0 = time.time()
    if args.dataset == "synthetic":
        from src.data_processors.synthetic_dataset import NoisySyntheticDataset
        logger.info("Loading synthetic Gaussian-feature multi-task dataset")
        _syn = NoisySyntheticDataset(
            epsilon_t=args.epsilon_t,
            class_noise_rho=args.class_noise_rho,
            class_noise_rho_indep=args.class_noise_rho_indep,
            seed=args.noise_seed,
        )

        # Adapter: synthetic uses "features" key; blitzer_grid expects "input_features".
        class _SynAdapter:
            def __init__(self, inner):
                self.inner = inner
                self.task_names = inner.task_names
                self.input_dim = inner.input_dim

            def set_mode(self, mode): self.inner.set_mode(mode)

            def __len__(self): return len(self.inner)

            def __getitem__(self, i):
                s = self.inner[i]
                return {**s, "input_features": s["features"]}

            def collate_fn(self, examples):
                # Match Blitzer collate output shape.
                import torch
                return {
                    "input_features": torch.stack([e["input_features"] for e in examples]),
                    "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
                    "task_name": [e["task_name"] for e in examples],
                    "true_task_name": [e["true_task_name"] for e in examples],
                    "true_label": torch.tensor([e["true_label"] for e in examples], dtype=torch.long),
                    "sample_idx": torch.tensor([e["sample_idx"] for e in examples], dtype=torch.long),
                }

        train_ds = _SynAdapter(_syn)
        val_ds = train_ds
    else:
        logger.info(f"Loading Blitzer corpus + TF-IDF (max_features={args.max_features}, ngram=(1,{args.ngram_max}))")
        train_ds = NoisyBlitzerDataset(
            epsilon_t=args.epsilon_t,
            class_noise_rho=args.class_noise_rho,
            class_noise_rho_indep=args.class_noise_rho_indep,
            seed=args.noise_seed,
            max_features=args.max_features,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
        )
        val_ds = train_ds
    TASKS = train_ds.task_names
    train_ds.set_mode(Mode.TRAIN)
    n_train = len(train_ds)
    train_ds.set_mode(Mode.VALIDATION) if hasattr(train_ds, '_labels') else train_ds.set_mode(Mode.TEST)
    n_val = len(train_ds)
    train_ds.set_mode(Mode.TRAIN)
    logger.info(f"input_dim={train_ds.input_dim}, tasks={TASKS}, "
                f"train={n_train}, val={n_val}, "
                f"setup time {time.time() - t0:.1f}s")

    # Build model
    if args.method in ("mtdnn", "standard_noisy", "bootstrapping",
                       "gce", "high_loss_discard", "forward_correction",
                       "mtlnl", "excessmtl"):
        model = _MTDNNModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "mtdnn"
    elif args.method in ("ifs_hesitant", "ifs_hesitant_bootstrap"):
        # Resolve head_type: explicit --head_type wins; otherwise fall back to
        # legacy --use_decoupled_head (True -> factored; False -> sigmoid).
        _head_type = args.head_type
        if _head_type is None:
            _head_type = "factored" if args.use_decoupled_head else "sigmoid"
        model = _IFSModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
            use_decoupled=args.use_decoupled_head,
            head_type=_head_type,
        )
        model_kind = "ifs"
    elif args.method == "ifs_hesitant_hybrid":
        model = _IFSLinearHybridModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "ifs_hybrid"
    elif args.method == "evidential":
        model = _EvidentialModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "evidential"
    elif args.method == "task_identity":
        model = _TaskIDModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "task_identity"
    elif args.method == "coteaching":
        model = _CoteachingModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "coteaching"
    elif args.method in ("pooled_ce", "pooled_gce", "pooled_bootstrapping",
                         "pooled_hld"):
        model = _PooledModel(
            input_dim=train_ds.input_dim,
            encoder_hidden=list(args.encoder_hidden),
            encoder_output=args.encoder_output,
            tasks=TASKS,
            dropout=args.dropout,
        )
        model_kind = "pooled"
    if IS_CUDA_AVAILABLE:
        model.cuda()
    if model_kind == "coteaching":
        opt_a = AdamW(model.model_a.parameters(), lr=args.lr)
        opt_b = AdamW(model.model_b.parameters(), lr=args.lr)
        optimizer = None  # not used; coteaching does its own dual-opt steps
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)
    task_to_idx = {t: i for i, t in enumerate(TASKS)}

    history: List[Dict[str, Any]] = []
    for epoch in range(args.num_epochs):
        train_ds.set_mode(Mode.TRAIN)
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_ds.collate_fn,
            generator=seeded_generator(args.noise_seed + epoch),
        )
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        post_warmup = (epoch >= args.warmup_epochs)
        for batch in loader:
            labels = batch[LABELS]
            t_idx = torch.tensor(
                [task_to_idx[t] for t in batch[TASK_NAME]], dtype=torch.long,
            )
            if IS_CUDA_AVAILABLE:
                labels = labels.cuda()
                t_idx = t_idx.cuda()
            if optimizer is not None:
                optimizer.zero_grad()
            if model_kind == "mtdnn":
                own_logits, per_sample_ce, _ = model(batch)
                if args.method == "bootstrapping":
                    # Bootstrapping (Reed et al. 2015): soft target between
                    # observed label and model's own prediction.
                    log_probs = nn.functional.log_softmax(own_logits, dim=-1)
                    probs = log_probs.exp().detach()
                    one_hot = nn.functional.one_hot(labels, num_classes=2).float()
                    target = args.bootstrap_beta * one_hot + (1.0 - args.bootstrap_beta) * probs
                    loss = -(target * log_probs).sum(dim=-1).mean()
                elif args.method == "gce":
                    # GCE (Zhang & Sabuncu 2018): (1 - p_y^q) / q.
                    probs = nn.functional.softmax(own_logits, dim=-1)
                    p_y = probs[torch.arange(probs.size(0)), labels]
                    loss = ((1.0 - p_y.pow(args.gce_q)) / args.gce_q).mean()
                elif args.method == "high_loss_discard":
                    # Drop the top-k fraction by per-sample CE loss in this batch.
                    n_keep = max(1, int(per_sample_ce.numel() * (1.0 - args.discard_ratio)))
                    keep_idx = per_sample_ce.argsort()[:n_keep]
                    loss = per_sample_ce[keep_idx].mean()
                elif args.method == "forward_correction":
                    # Patrini et al. 2017: apply T^T * softmax(z) before CE.
                    # T is the oracle binary symmetric matrix from known noise rates.
                    p_flip = (args.epsilon_t * args.class_noise_rho
                              + (1.0 - args.epsilon_t) * args.class_noise_rho_indep)
                    p_flip = max(min(p_flip, 0.499), 1e-6)
                    T = torch.tensor([[1.0 - p_flip, p_flip],
                                      [p_flip, 1.0 - p_flip]],
                                     device=own_logits.device)
                    corrected = torch.matmul(nn.functional.softmax(own_logits, dim=-1), T)
                    loss = nn.functional.nll_loss(torch.log(corrected + 1e-8), labels)
                elif args.method == "mtlnl":
                    # MTL-NL (Gu et al. 2023): per-task forward correction with
                    # anchor-points-estimated transition matrix T_t. Until the
                    # first estimation happens (epoch 0), train with plain CE.
                    T_per_task = getattr(args, "_mtlnl_T", None)
                    if T_per_task is None:
                        loss = per_sample_ce.mean()
                    else:
                        T_batch = T_per_task[t_idx]  # [B, 2, 2]
                        probs = nn.functional.softmax(own_logits, dim=-1)  # [B, 2]
                        # corrected[i, j] = sum_k T[t_i, k, j] * probs[i, k]
                        corrected = torch.einsum("bkj,bk->bj", T_batch, probs)
                        loss = nn.functional.nll_loss(
                            torch.log(corrected.clamp_min(1e-8)), labels)
                elif args.method == "excessmtl":
                    # ExcessMTL (He et al. 2024): per-task weights from excess
                    # risk on clean validation. Update task weights at epoch start;
                    # warmup with uniform weights.
                    w = getattr(args, "_excessmtl_weights", None)
                    if w is None:
                        loss = per_sample_ce.mean()
                    else:
                        w_batch = w[t_idx]  # [B]
                        loss = (per_sample_ce * w_batch).mean()
                else:
                    # mtdnn / standard_noisy - plain CE
                    loss = per_sample_ce.mean()
            elif model_kind == "ifs":
                triples = model(batch)
                # Methods with Bootstrap-soft target use it during BOTH warmup
                # and (if applicable) the post-warmup hesitant phase.
                use_boot = args.method.endswith("_bootstrap")
                boot_beta = args.bootstrap_beta if use_boot else None
                if post_warmup and args.method in (
                        "ifs_hesitant", "ifs_hesitant_bootstrap",
                ):
                    loss = _ifs_loss_hesitant(
                        triples, labels, t_idx,
                        lam=args.lambda_other, w_min=args.w_min,
                        bootstrap_beta=boot_beta,
                        weight_norm=args.weight_norm,
                        weight_norm_target=args.weight_norm_target,
                        trust_signal=args.trust_signal,
                        trust_tnorm=args.trust_tnorm,
                        trust_no_gate=args.trust_no_gate,
                        trust_no_detach=args.trust_no_detach,
                        decoupled_hesitant_beta=args.decoupled_hesitant_beta,
                        no_lpred=args.no_lpred,
                        l_pred_alpha=args.l_pred_alpha,
                        bootstrap_hesitant=args.bootstrap_hesitant,
                        bayes_trust=args.bayes_trust,
                        head_type=getattr(model, "head_type", "sigmoid"),
                    )
                else:
                    loss = _ifs_loss_warmup(triples, labels, t_idx,
                                            lam=args.lambda_other,
                                            bootstrap_beta=boot_beta,
                                            warmup_discard_ratio=args.warmup_discard_ratio)
            elif model_kind == "ifs_hybrid":
                triples, tid_logits = model(batch)
                if post_warmup:
                    ifs_loss = _ifs_loss_hesitant(
                        triples, labels, t_idx,
                        lam=args.lambda_other, w_min=args.w_min,
                        weight_norm=args.weight_norm,
                        weight_norm_target=args.weight_norm_target,
                        trust_signal=args.trust_signal,
                        trust_tnorm=args.trust_tnorm,
                        trust_no_gate=args.trust_no_gate,
                        trust_no_detach=args.trust_no_detach,
                        l_pred_alpha=args.l_pred_alpha,
                        bayes_trust=args.bayes_trust,
                        head_type=getattr(model.head, "head_type", "sigmoid"),
                    )
                else:
                    ifs_loss = _ifs_loss_warmup(triples, labels, t_idx,
                                                lam=args.lambda_other,
                                                warmup_discard_ratio=args.warmup_discard_ratio)
                tid_loss = nn.functional.cross_entropy(tid_logits, t_idx)
                loss = ifs_loss + args.hybrid_lambda_taskid * tid_loss
            elif model_kind == "evidential":
                alpha = model.alpha(batch)
                kl_l = (min(1.0, epoch / max(args.kl_anneal_epochs, 1))
                        if args.kl_anneal_epochs > 0 else 0.0)
                loss = _evidential_loss(alpha, labels, t_idx,
                                        lam=args.lambda_other, kl_lambda=kl_l)
            elif model_kind == "task_identity":
                _, per_sample_ce, _, task_id_logits = model(batch)
                # Per-task classification CE on the (noisy) assigned head
                cls_loss = per_sample_ce.mean()
                # Task-id CE - supervise on the (noisy) assigned task label
                tid_loss = nn.functional.cross_entropy(task_id_logits, t_idx)
                loss = cls_loss + tid_loss
            elif model_kind == "pooled":
                logits, per_sample_ce = model(batch)
                if args.method == "pooled_bootstrapping":
                    log_probs = nn.functional.log_softmax(logits, dim=-1)
                    probs = log_probs.exp().detach()
                    one_hot = nn.functional.one_hot(labels, num_classes=2).float()
                    target = (args.bootstrap_beta * one_hot
                              + (1.0 - args.bootstrap_beta) * probs)
                    loss = -(target * log_probs).sum(dim=-1).mean()
                elif args.method == "pooled_gce":
                    probs = nn.functional.softmax(logits, dim=-1)
                    p_y = probs[torch.arange(probs.size(0)), labels]
                    loss = ((1.0 - p_y.pow(args.gce_q)) / args.gce_q).mean()
                elif args.method == "pooled_hld":
                    n_keep = max(1, int(per_sample_ce.numel()
                                        * (1.0 - args.discard_ratio)))
                    keep_idx = per_sample_ce.argsort()[:n_keep]
                    loss = per_sample_ce[keep_idx].mean()
                else:  # pooled_ce
                    loss = per_sample_ce.mean()
            elif model_kind == "coteaching":
                # First pass (no grad): per-sample CE for selection
                with torch.no_grad():
                    _, ce_a_eval, _ = model.model_a(batch)
                    _, ce_b_eval, _ = model.model_b(batch)
                if not post_warmup:
                    # Warm-up: each model trains on full batch independently
                    _, ce_a, _ = model.model_a(batch)
                    loss_a = ce_a.mean()
                    opt_a.zero_grad();
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_a.parameters(), 1.0)
                    opt_a.step()
                    _, ce_b, _ = model.model_b(batch)
                    loss_b = ce_b.mean()
                    opt_b.zero_grad();
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_b.parameters(), 1.0)
                    opt_b.step()
                else:
                    # Co-teach: each model selects bottom (1 - epsilon_t) by
                    # OWN per-sample CE -> those samples train the OTHER model.
                    keep_ratio = 1.0 - args.epsilon_t
                    n_keep = max(1, int(ce_a_eval.size(0) * keep_ratio))
                    idx_a_clean = ce_a_eval.argsort()[:n_keep]
                    idx_b_clean = ce_b_eval.argsort()[:n_keep]
                    # A trains on samples B selected as clean
                    _, ce_a, _ = model.model_a(batch)
                    loss_a = ce_a[idx_b_clean].mean()
                    opt_a.zero_grad();
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_a.parameters(), 1.0)
                    opt_a.step()
                    # B trains on samples A selected as clean
                    _, ce_b, _ = model.model_b(batch)
                    loss_b = ce_b[idx_a_clean].mean()
                    opt_b.zero_grad();
                    loss_b.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_b.parameters(), 1.0)
                    opt_b.step()
                epoch_loss += loss_a.item()
                n_batches += 1
                continue  # skip the standard backward / step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_ds.set_mode(Mode.VALIDATION)
        # Save per-sample preds on the FINAL epoch only (for paper-correct MCC scoring)
        is_final_epoch = (epoch == args.num_epochs - 1)
        eval_result = _predict_known_blind(
            model, train_ds, TASKS, args.batch_size, model_kind,
            hybrid_alpha=args.hybrid_alpha,
            return_preds=is_final_epoch,
        )
        if is_final_epoch:
            acc_known, acc_blind, per_known, per_blind, preds_log = eval_result
            # Save per-sample predictions to predictions.json for MCC computation
            import json as _json
            with open(os.path.join(args.out_dir, "predictions.json"), "w") as _f:
                _json.dump(preds_log, _f)
        else:
            acc_known, acc_blind, per_known, per_blind = eval_result
        avg_loss = epoch_loss / max(n_batches, 1)
        rec = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "acc_known": round(acc_known, 4),
            "acc_blind": round(acc_blind, 4),
            "per_task_known": per_known,
            "per_task_blind": per_blind,
            "phase": "post_warmup" if post_warmup else "warmup",
        }
        history.append(rec)
        logger.info(
            f"epoch {epoch}: loss {avg_loss:.4f} | "
            f"known {acc_known:.4f} | blind {acc_blind:.4f} | "
            f"per-task blind {per_blind}"
        )

        # End-of-epoch state updates for MTL-NL and ExcessMTL
        # MTL-NL: anchor-points T must be estimated EARLY (before model
        # overfits the noisy labels and anchors look identity-clean).
        # Estimate once after epoch 0 only; freeze T for the rest of training.
        if args.method == "mtlnl" and epoch == 0 and not hasattr(args, "_mtlnl_T_frozen"):
            # Anchor-points T_t estimation: for each (task t, predicted class c),
            # top-K samples by model's P(y=c|x,t) are "anchor points"; T_t[c,c'] =
            # fraction of anchors observed as c' (Patrini 2017 / MTL-NL).
            n_tasks = len(TASKS)
            anchor_K = max(20, int(0.02 * n_train))  # ~2% of train, min 20
            T = torch.eye(2).repeat(n_tasks, 1, 1)
            # Collect (prob_class0, prob_class1, observed_label) per (task) for assigned-head samples
            by_task = {t: [] for t in range(n_tasks)}
            model.eval()
            train_ds.set_mode(Mode.TRAIN)
            eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=train_ds.collate_fn)
            with torch.no_grad():
                for batch in eval_loader:
                    own_logits, _, _ = model(batch)
                    probs = nn.functional.softmax(own_logits, dim=-1).cpu()
                    labels_b = batch["labels"].cpu()
                    tn = [task_to_idx[t] for t in batch["task_name"]]
                    for i in range(len(labels_b)):
                        by_task[tn[i]].append((probs[i, 0].item(), probs[i, 1].item(), labels_b[i].item()))
            for t in range(n_tasks):
                rows = by_task[t]
                if not rows: continue
                # Class 0 anchors: top-K by P(y=0|x)
                rows0 = sorted(rows, key=lambda r: -r[0])[:anchor_K]
                # Class 1 anchors: top-K by P(y=1|x)
                rows1 = sorted(rows, key=lambda r: -r[1])[:anchor_K]
                for true_class, anchors in [(0, rows0), (1, rows1)]:
                    if not anchors: continue
                    n_obs0 = sum(1 for r in anchors if r[2] == 0)
                    n_obs1 = sum(1 for r in anchors if r[2] == 1)
                    total = n_obs0 + n_obs1
                    if total > 0:
                        T[t, true_class, 0] = n_obs0 / total
                        T[t, true_class, 1] = n_obs1 / total
            args._mtlnl_T = T.cuda() if IS_CUDA_AVAILABLE else T
            args._mtlnl_T_frozen = True
            model.train()
            logger.info(f"MTL-NL T frozen at epoch {epoch}: "
                        f"per-task off-diagonal means = {[round(((T[t, 0, 1] + T[t, 1, 0]) / 2).item(), 3) for t in range(n_tasks)]}")
        if args.method == "excessmtl" and epoch >= max(args.warmup_epochs - 1, 0):
            # Per-task weights from excess risk on (clean) validation labels.
            # Excess risk = val_loss - train_loss. Weight scales with excess_risk.
            # (He et al. 2024 min-max: up-weight worst tasks).
            n_tasks = len(TASKS)
            model.eval()

            def per_task_loss(mode):
                train_ds.set_mode(mode)
                ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=train_ds.collate_fn)
                sums = [0.0] * n_tasks;
                counts = [0] * n_tasks
                with torch.no_grad():
                    for batch in ld:
                        own_logits, _, _ = model(batch)
                        labels_b = batch["labels"]
                        if IS_CUDA_AVAILABLE: labels_b = labels_b.cuda()
                        ce = nn.functional.cross_entropy(own_logits, labels_b, reduction="none")
                        tn = [task_to_idx[t] for t in batch["task_name"]]
                        ce_cpu = ce.cpu().tolist()
                        for i, ti in enumerate(tn):
                            sums[ti] += ce_cpu[i];
                            counts[ti] += 1
                return [s / max(c, 1) for s, c in zip(sums, counts)]

            train_loss_pt = per_task_loss(Mode.TRAIN)
            val_loss_pt = per_task_loss(Mode.VALIDATION if hasattr(train_ds, '_labels') else Mode.TEST)
            train_ds.set_mode(Mode.TRAIN)
            # He et al. 2024 ICML: weights proportional to excess_risk (up-weight worst tasks).
            excess = [max(v - t, 1e-3) for v, t in zip(val_loss_pt, train_loss_pt)]
            mean_excess = sum(excess) / len(excess)
            weights = [e / mean_excess for e in excess]  # normalised: mean=1
            w_tensor = torch.tensor(weights, dtype=torch.float32)
            args._excessmtl_weights = w_tensor.cuda() if IS_CUDA_AVAILABLE else w_tensor
            model.train()
            logger.info(f"ExcessMTL weights at epoch {epoch}: {[round(w, 3) for w in weights]} "
                        f"(excess risk: {[round(e, 3) for e in excess]})")

    # Noise detection AUROC at end-of-training. For IFS methods we compute
    # multiple scoring functions (each tries to identify noisy task assignments):
    #   - ifs       : pi_{t_obs} + max_{j!=t}[(1-pi_j)*max(mu_j,nu_j)]  (full IFS)
    #   - pi_only   : pi_{t_obs}                                        (hesitancy)
    #   - loss      : -log(sat_{t_obs}(y))                              (IFS NLL)
    #   - conf_inv  : 1 - max(mu_{t_obs}, nu_{t_obs})                   (1-max-confidence)
    # For non-IFS classifier methods we compute the loss-based AUROC.
    try:
        from sklearn.metrics import roc_auc_score
        model.eval()
        train_ds.set_mode(Mode.TRAIN)
        scan_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=train_ds.collate_fn,
        )
        scores_ifs, scores_pi, scores_loss, scores_conf = [], [], [], []
        # P3 IFS-distance noise scores:
        #   hamming      = 1 - sat_t_tilde(y_tilde)            (= Atanassov-Burillo
        #                                                       Hamming distance to ideal
        #                                                       prototype I^+/I^-)
        #   neg_log_w    = -log(1-pi_t_tilde) - log(sat_t_tilde(y_tilde))
        #                                                       (additive product-T-norm
        #                                                       log, CE-aligned)
        #   one_minus_w  = 1 - (1-pi_t_tilde)*sat_t_tilde(y_tilde)
        #                                                       (multiplicative product-T-norm,
        #                                                       monotone-equiv to neg_log_w)
        scores_hamming, scores_neg_log_w, scores_one_minus_w = [], [], []
        scores_ce = []  # for non-IFS methods
        corrupted: List[int] = []
        with torch.no_grad():
            for batch in scan_loader:
                obs = batch[TASK_NAME]
                true_t = batch.get(TRUE_TASK_NAME, obs)
                t_idx = torch.tensor(
                    [model.task_to_idx[t] for t in obs], dtype=torch.long,
                )
                labels_b = batch[LABELS]
                if IS_CUDA_AVAILABLE:
                    t_idx = t_idx.cuda()
                    labels_b = labels_b.cuda()
                B = len(obs)
                if model_kind in ("ifs", "ifs_hybrid"):
                    triples = model(batch)[0] if model_kind == "ifs_hybrid" else model(batch)
                    mu = triples[..., 0]
                    pi = triples[..., 1]
                    nu = triples[..., 2]
                    pi_obs = pi[torch.arange(B, device=pi.device), t_idx]
                    mu_obs = mu[torch.arange(B, device=mu.device), t_idx]
                    nu_obs = nu[torch.arange(B, device=nu.device), t_idx]
                    sat_obs = labels_b.float() * mu_obs + (1.0 - labels_b.float()) * nu_obs
                    conf_obs = torch.maximum(mu_obs, nu_obs)
                    conf_grid = (1 - pi) * torch.maximum(mu, nu)
                    K = conf_grid.shape[1]
                    mask = torch.ones(B, K, dtype=torch.bool, device=conf_grid.device)
                    mask[torch.arange(B, device=conf_grid.device), t_idx] = False
                    other_max = conf_grid[mask].view(B, K - 1).max(dim=-1).values
                    scores_ifs.extend((pi_obs + other_max).cpu().tolist())
                    scores_pi.extend(pi_obs.cpu().tolist())
                    scores_loss.extend((-torch.log(sat_obs + 1e-8)).cpu().tolist())
                    scores_conf.extend((1.0 - conf_obs).cpu().tolist())
                    # P3 IFS-distance noise scores
                    scores_hamming.extend((1.0 - sat_obs).cpu().tolist())
                    scores_neg_log_w.extend(
                        (-torch.log(1 - pi_obs + 1e-8)
                         - torch.log(sat_obs + 1e-8)).cpu().tolist())
                    scores_one_minus_w.extend(
                        (1.0 - (1 - pi_obs) * sat_obs).cpu().tolist())
                elif model_kind == "mtdnn":
                    own_logits, per_sample_ce, _ = model(batch)
                    scores_ce.extend(per_sample_ce.cpu().tolist())
                elif model_kind == "evidential":
                    alpha = model.alpha(batch)
                    own = alpha[torch.arange(B, device=alpha.device), t_idx]
                    p = own / own.sum(dim=-1, keepdim=True)
                    p_y = p[torch.arange(B, device=p.device), labels_b]
                    scores_ce.extend((-torch.log(p_y + 1e-8)).cpu().tolist())
                elif model_kind == "task_identity":
                    _, per_sample_ce, _, _ = model(batch)
                    scores_ce.extend(per_sample_ce.cpu().tolist())
                elif model_kind == "coteaching":
                    _, per_sample_ce, _ = model.model_a(batch)
                    scores_ce.extend(per_sample_ce.cpu().tolist())
                elif model_kind == "pooled":
                    _, per_sample_ce = model(batch)
                    scores_ce.extend(per_sample_ce.cpu().tolist())
                corrupted.extend([0 if o == t else 1 for o, t in zip(obs, true_t)])

        aurocs: Dict[str, float] = {}
        if len(set(corrupted)) >= 2:
            for name, vec in [("ifs", scores_ifs), ("pi_only", scores_pi),
                              ("loss", scores_loss), ("conf_inv", scores_conf),
                              ("hamming", scores_hamming),
                              ("neg_log_w", scores_neg_log_w),
                              ("one_minus_w", scores_one_minus_w),
                              ("ce", scores_ce)]:
                if vec:
                    aurocs[f"auroc_{name}"] = round(float(roc_auc_score(corrupted, vec)), 4)
            logger.info(f"noise detection AUROCs: {aurocs}")
            if history:
                history[-1].update(aurocs)
    except Exception as e:
        logger.warning(f"AUROC computation failed: {e}")

    out_path = os.path.join(args.out_dir, "history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved {out_path} ({time.time() - t0:.1f}s total)")
    # touch .done
    open(os.path.join(args.out_dir, ".done"), "w").close()


if __name__ == "__main__":
    main()
