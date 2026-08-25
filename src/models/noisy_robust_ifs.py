import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from transformers import BertModel

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.models.multitask_cls import MultitaskFuzzyQueryBasedModel


class NoisyRobustIFSModel(MultitaskFuzzyQueryBasedModel):

    def __init__(self, bert: BertModel, tasks: List[GlueTask],
                 do_blindly_decode: bool = True,
                 w_min: float = 0.1,
                 weight_norm: str = "mean",
                 weight_norm_target: float = 0.5,
                 lambda_other: float = 1.0,
                 lambda_pi: float = 1.0,
                 trust_signal: str = "ifs",
                 class_loss: str = "kl",
                 l_pred_alpha: float = 0.0,
                 l_pred_barrier: bool = False,
                 head_type: str = "ifs",
                 tnorm: str = "product",
                 bayes_trust: bool = False,
                 trust_no_gate: bool = False):
        super().__init__(bert, tasks, do_blindly_decode=do_blindly_decode,
                         is_max_opposite=False)
        # 3-logit decoupled head: per task (m, n, p).
        self.linear_decoupled_head = nn.Linear(self.hidden_size, self.num_tasks * 3)
        # Map idx -> task name (used to log the confidence-argmax routed task at eval).
        self.idx_to_task = {idx: task.value for idx, task in enumerate(self.tasks)}
        # Cache for the loss path to access m, n logits without recomputing.
        self._last_m: Optional[torch.Tensor] = None
        self._last_n: Optional[torch.Tensor] = None

        if trust_signal not in ("ifs", "entropy", "margin", "loss"):
            raise ValueError(f"unknown trust_signal {trust_signal!r}")
        if class_loss not in ("kl", "hamming"):
            raise ValueError(f"unknown class_loss {class_loss!r}")
        if not 0.0 <= l_pred_alpha <= 1.0:
            raise ValueError(f"l_pred_alpha must be in [0, 1], got {l_pred_alpha}")
        if head_type not in ("ifs", "softmax3"):
            raise ValueError(f"unknown head_type {head_type!r}")
        if tnorm not in ("product", "min", "lukasiewicz", "hamacher"):
            raise ValueError(f"unknown tnorm {tnorm!r}")
        if bayes_trust and head_type != "ifs":
            raise ValueError(
                "bayes_trust=True requires head_type='ifs' (factored). "
                "sm3 has no structural sat_tilde (separately parameterized "
                "class-conditional independent of pi), so the (1-pi)*sat_tilde "
                "Bayesian factorisation is not well-defined.")
        self.bayes_trust = bayes_trust
        self.head_type = head_type
        self.trust_signal = trust_signal
        self.trust_no_gate = trust_no_gate
        self.class_loss = class_loss
        self.l_pred_alpha = l_pred_alpha
        self.l_pred_barrier = l_pred_barrier
        self.w_min = w_min
        if weight_norm not in ("mean", "none"):
            raise ValueError(f"weight_norm must be 'mean' or 'none', got {weight_norm!r}")
        self.weight_norm = weight_norm
        self.weight_norm_target = weight_norm_target
        self.lambda_other = lambda_other
        self.lambda_pi = lambda_pi
        self.tnorm = tnorm

    @staticmethod
    def _apply_tnorm(a: torch.Tensor, b: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == "product":
            return a * b
        if kind == "min":
            return torch.minimum(a, b)
        if kind == "lukasiewicz":
            return torch.clamp(a + b - 1.0, min=0.0)
        if kind == "hamacher":
            # gamma=2 Hamacher product. Stable form: a*b / (a + b - a*b + eps)
            # so denominator never collapses when a=b=0 (returns 0).
            return a * b / (a + b - a * b + 1e-8)
        raise ValueError(f"unknown tnorm {kind!r}")

    def get_batch_embeddings(self, batch: Dict[str, Any]) -> torch.Tensor:
        keys_skip = {TASK_NAME, LABELS, "sample_idx", "true_task_name", "true_label"}
        input_batch = {k: v for k, v in batch.items() if k not in keys_skip}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        return self.dropout(self.bert(**input_batch).pooler_output)

    def memberships_from_embeddings(self, embeddings: torch.Tensor):
        B = embeddings.size(0)
        logits = self.linear_decoupled_head(embeddings).view(B, self.num_tasks, 3)
        m = logits[:, :, 0]
        n = logits[:, :, 1]
        p = logits[:, :, 2]
        if self.head_type == "ifs":
            mn_soft = torch.softmax(torch.stack([m, n], dim=-1), dim=-1)
            mu_s = mn_soft[..., 0]
            nu_s = mn_soft[..., 1]
            pi = torch.sigmoid(p)
            mu = (1.0 - pi) * mu_s
            nu = (1.0 - pi) * nu_s
        else:  # softmax3
            mnp_soft = torch.softmax(torch.stack([m, n, p], dim=-1), dim=-1)
            mu = mnp_soft[..., 0]
            nu = mnp_soft[..., 1]
            pi = mnp_soft[..., 2]
        # cache for the loss (avoids recomputing logits)
        self._last_m = m
        self._last_n = n
        return mu, pi, nu

    def compute_loss(self, m: torch.Tensor, n: torch.Tensor, pi: torch.Tensor,
                     labels: torch.Tensor, task_indices: List[int],
                     mu: Optional[torch.Tensor] = None,
                     nu: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the trust-weighted IFS loss on the observed task."""
        eps = 1e-8
        B = m.shape[0]
        T = m.shape[1]
        device = m.device
        idx_b = torch.arange(B, device=device)
        t_obs = torch.as_tensor(task_indices, device=device, dtype=torch.long)
        labels_f = labels.float() if labels.dtype != torch.float else labels
        pi_obs = pi[idx_b, t_obs]  # [B]

        # Build per-task (mu, nu) under the active head parameterization.
        if self.head_type == "ifs":
            # Factored head: split class softmax and hesitation.
            mn_soft = torch.softmax(torch.stack([m, n], dim=-1), dim=-1)  # [B,T,2]
            mu_full = (1.0 - pi) * mn_soft[..., 0]  # [B, T]
            nu_full = (1.0 - pi) * mn_soft[..., 1]
        else:  # softmax3
            assert mu is not None and nu is not None, \
                "head_type='softmax3' requires mu and nu kwargs to compute_loss"
            mu_full = mu  # [B, T]
            nu_full = nu

        mu_obs = mu_full[idx_b, t_obs]
        nu_obs = nu_full[idx_b, t_obs]
        sat_obs = labels_f * mu_obs + (1.0 - labels_f) * nu_obs  # [B]

        # Observed-head class term.
        if self.class_loss == "kl":
            ce_class = -torch.log(sat_obs + eps)  # [B]
        else:  # "hamming"
            ce_class = 1.0 - sat_obs  # [B]

        # Barrier over non-observed heads.
        if T > 1:
            mask_obs = torch.zeros(B, T, dtype=torch.bool, device=device)
            mask_obs[idx_b, t_obs] = True
            log_pi = -torch.log(pi + eps)
            barrier = (log_pi * (~mask_obs).float()).sum(dim=1) / (T - 1)  # [B]
        else:
            barrier = torch.zeros(B, device=device)

        # Supervised loss on the observed task.
        L_sup = ce_class + self.lambda_pi * self.lambda_other * barrier  # [B]

        # Trust weight on the observed task.
        # bayes_trust uses the conditional class score; otherwise use joint sat.
        if self.trust_signal == "ifs":
            if self.bayes_trust:
                mn_soft_trust = torch.softmax(torch.stack([m, n], dim=-1), dim=-1)
                mu_s_obs = mn_soft_trust[idx_b, t_obs, 0]
                nu_s_obs = mn_soft_trust[idx_b, t_obs, 1]
                sat_tilde_obs = labels_f * mu_s_obs + (1.0 - labels_f) * nu_s_obs
                w_raw = self._apply_tnorm(1.0 - pi_obs, sat_tilde_obs, self.tnorm)
            else:
                w_raw = self._apply_tnorm(1.0 - pi_obs, sat_obs, self.tnorm)
        elif self.trust_signal == "margin":
            w_raw = torch.abs(mu_obs - nu_obs)
        elif self.trust_signal == "loss":
            ce_obs = torch.where(labels_f > 0.5,
                                 -torch.log(mu_obs + eps),
                                 -torch.log(nu_obs + eps))
            w_raw = torch.exp(-ce_obs)
        elif self.trust_signal == "entropy":
            denom = mu_obs + nu_obs + eps
            p1 = mu_obs / denom
            p0 = nu_obs / denom
            ent = -(p1 * torch.log(p1 + eps) + p0 * torch.log(p0 + eps))
            w_raw = 1.0 - ent / math.log(2.0)
        w_raw = w_raw.detach()

        # Ablation: disable the trust gate.
        if self.trust_no_gate:
            w_raw = torch.ones_like(w_raw)

        # Optional batch-mean normalisation.
        if self.weight_norm == "mean" and B > 0:
            w_mean = w_raw.mean()
            if w_mean.item() > eps:
                w_raw = w_raw * (self.weight_norm_target / w_mean)

        # Clamp trust weights to the configured floor.
        w = w_raw.clamp(self.w_min, 1.0)  # [B]

        # Optional self-routed class-only branch. Skip the barrier here.
        if self.l_pred_alpha > 0.0:
            with torch.no_grad():
                # Route by blind confidence, then take the winning class.
                conf_all = self._blind_conf(mu_full, pi, nu_full)  # [B, T]
                t_pred = conf_all.argmax(dim=1)  # [B]
                y_pred = (mu_full[idx_b, t_pred] > nu_full[idx_b, t_pred]).long()  # [B]
            mu_pred = mu_full[idx_b, t_pred]
            nu_pred = nu_full[idx_b, t_pred]
            sat_pred = torch.where(y_pred > 0, mu_pred, nu_pred)  # [B]
            if self.class_loss == "kl":
                L_pred_class = -torch.log(sat_pred + eps)
            else:  # "hamming"
                L_pred_class = 1.0 - sat_pred

            if self.l_pred_barrier and T > 1:
                # Barrier on the non-predicted heads to encourage clean routing.
                mask_pred = torch.zeros(B, T, dtype=torch.bool, device=device)
                mask_pred[idx_b, t_pred] = True
                log_pi_pred = -torch.log(pi + eps)
                barrier_pred = (log_pi_pred * (~mask_pred).float()).sum(dim=1) / (T - 1)
                L_pred_total = L_pred_class + self.lambda_pi * self.lambda_other * barrier_pred
            else:
                L_pred_total = L_pred_class

            # w_int caps observed share so L_pred floor is alpha*w_min on noisy
            w_int = w_raw.clamp(self.w_min, 1.0 - self.l_pred_alpha * self.w_min)
            return (w_int * L_sup + (1.0 - w_int) * self.l_pred_alpha * L_pred_total).mean()

        # ---- L_i = w_bar_i * L_sup_i (Eq. hesitant_loss_clamped) ----
        return (w * L_sup).mean()

    def _blind_conf(self, mu: torch.Tensor, pi: torch.Tensor,
                    nu: torch.Tensor) -> torch.Tensor:
        if getattr(self, "head_type", "ifs") == "ifs":
            return torch.max(mu, nu)
        return (1.0 - pi) * torch.max(mu, nu)

    def defuzzy_ifs(self, mu: torch.Tensor, pi: torch.Tensor,
                    nu: torch.Tensor, correct_task_indices=None) -> torch.Tensor:
        B = mu.shape[0]
        predictions = torch.zeros(B, 2, dtype=torch.float, device=mu.device)
        confidence = self._blind_conf(mu, pi, nu)
        for b in range(B):
            t = (correct_task_indices[b] if correct_task_indices is not None
                 else torch.argmax(confidence[b]).item())
            predictions[b] = (torch.tensor([0, 1], dtype=torch.float)
                              if mu[b, t] > nu[b, t]
                              else torch.tensor([1, 0], dtype=torch.float))
        return predictions

    def forward(self, batch: Dict[str, Any],
                blind_override: Optional[bool] = None,
                **_unused):
        embeddings = self.get_batch_embeddings(batch)
        mu, pi, nu = self.memberships_from_embeddings(embeddings)

        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        task_names = batch[TASK_NAME]
        task_indices = [self.task_to_idx[t] for t in task_names]

        loss = self.compute_loss(self._last_m, self._last_n, pi, labels, task_indices,
                                 mu=mu, nu=nu)

        use_blind = self.do_blindly_decode if blind_override is None else blind_override
        predictions = self.defuzzy_ifs(mu, pi, nu,
                                       task_indices if not use_blind else None)

        # Save eval data for the aggregator. mu/pi/nu are saved so the
        # aggregator can compute the K column (oracle-routed at task t)
        # and the B column (model-routed via predicted_task) from the same
        # saved record, by indexing into mu[t]/nu[t] respectively.
        if not self.training:
            confidence = self._blind_conf(mu, pi, nu)  # [B, T]
            for idx in range(len(task_names)):
                pred_t = torch.argmax(confidence[idx]).item()
                self.all_task_names.append(task_names[idx])
                self.all_evaluated_labels.append(batch[LABELS][idx].item())
                self.all_membership_scores.append({
                    "mu": mu[idx].detach().cpu().tolist(),
                    "pi": pi[idx].detach().cpu().tolist(),
                    "nu": nu[idx].detach().cpu().tolist(),
                    "predicted_task": self.idx_to_task[pred_t],
                    "predicted_label": int(mu[idx, pred_t] > nu[idx, pred_t]),
                })

        return loss, predictions
