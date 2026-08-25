from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BLSConfig:
    m: int = 10  # number of feature groups
    p: int = 10  # nodes per feature group
    l: int = 1  # number of enhancement groups
    q: int = 200  # nodes per enhancement group
    C: float = 100.0  # ridge regularization (authors' MATLAB default)
    mu_kernel: Optional[float] = None  # Gaussian kernel width; None = median heuristic
    delta: float = 1e-4  # membership smoothing (authors' 1e-4)
    activation: str = "tansig"  # enhancement activation; tansig == tanh
    feature_normalize: bool = True  # mapminmax columnwise on F before concat
    orth_enhancement: bool = True  # orth() on enhancement weight matrix
    score_mode: str = "ifs"  # "ifs" (paper Eqs. 11-13, kernel space) or
    # "fbls" (paper Eq. 7, raw input space).
    # IF-BLS degenerates on high-dim sparse data
    # (curse of dimensionality on Gaussian kernel);
    # F-BLS is the paper's alternative for those.
    score_normalize: bool = False  # OPT-IN deviation from paper. Rescales
    # per-task scores so max=1; equivalent to
    # boosting C by 1/max(S)^2. Use only if raw
    # scores degenerate to ~0 magnitude.
    max_kernel_n: int = 8000  # If a task has more samples than this, the IFS
    # kernel matrix N*N exceeds available RAM
    # (8000^2 * 8B = 512MB). Subsample to this many
    # samples per task for the score computation;
    # the ridge solve still uses all samples.


def _activation(name: str):
    if name in ("tanh", "tansig"):
        return np.tanh
    if name == "sigmoid":
        return lambda z: 1.0 / (1.0 + np.exp(-z))
    if name == "relu":
        return lambda z: np.maximum(z, 0.0)
    raise ValueError(f"unknown activation {name}")


def _orthogonalize(W: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(W, full_matrices=False)
    tol = max(W.shape) * np.finfo(W.dtype).eps * (s[0] if s.size else 0.0)
    return u[:, s > tol]


def _mapminmax_fit(X: np.ndarray):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = xmax - xmin
    span_safe = np.where(span > 0, span, 1.0)
    X_s = 2.0 * (X - xmin) / span_safe - 1.0
    X_s = np.where(span > 0, X_s, 0.0)
    return X_s, {"xmin": xmin, "xmax": xmax, "span_safe": span_safe,
                 "active": span > 0}


def _mapminmax_apply(X: np.ndarray, params: Dict) -> np.ndarray:
    X_s = 2.0 * (X - params["xmin"]) / params["span_safe"] - 1.0
    return np.where(params["active"], X_s, 0.0)


def build_random_features(X: np.ndarray, cfg: BLSConfig,
                          rng: np.random.Generator
                          ) -> Tuple[np.ndarray, Dict]:
    N, D = X.shape
    m, p, l, q = cfg.m, cfg.p, cfg.l, cfg.q

    # Segment-1: feature groups F = [F_1, ..., F_m]; linear, then mapminmax
    W_F = rng.standard_normal((D, m * p)).astype(np.float64) * (1.0 / np.sqrt(D))
    b_F = rng.standard_normal((1, m * p)).astype(np.float64) * 0.1
    F_raw = X @ W_F + b_F  # (N, mp)
    if cfg.feature_normalize:
        F, mm_params = _mapminmax_fit(F_raw)
    else:
        F, mm_params = F_raw, None

    # Segment-2: enhancement E = tansig(F W_E + b_E); W_E is column-orthonormal
    W_E_raw = rng.standard_normal((m * p, l * q)).astype(np.float64)
    if cfg.orth_enhancement:
        W_E = _orthogonalize(W_E_raw)
    else:
        W_E = W_E_raw * (1.0 / np.sqrt(m * p))
    b_E = rng.standard_normal((1, W_E.shape[1])).astype(np.float64) * 0.1
    act = _activation(cfg.activation)
    E = act(F @ W_E + b_E)  # (N, lq')

    G = np.concatenate([F, E], axis=1)  # (N, mp + lq')
    return G, {"W_F": W_F, "b_F": b_F, "W_E": W_E, "b_E": b_E,
               "act": cfg.activation, "mm_params": mm_params}


def transform_random_features(X: np.ndarray, params: Dict) -> np.ndarray:
    F_raw = X @ params["W_F"] + params["b_F"]
    F = _mapminmax_apply(F_raw, params["mm_params"]) if params["mm_params"] else F_raw
    act = _activation(params["act"])
    E = act(F @ params["W_E"] + params["b_E"])
    return np.concatenate([F, E], axis=1)


def gaussian_kernel(X1: np.ndarray, X2: np.ndarray, mu: float) -> np.ndarray:
    sq1 = np.sum(X1 * X1, axis=1, keepdims=True)  # (N1, 1)
    sq2 = np.sum(X2 * X2, axis=1, keepdims=True).T  # (1, N2)
    sqd = np.maximum(sq1 + sq2 - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-sqd / (mu * mu))


def fbls_scores_per_task(X_task: np.ndarray, y_task: np.ndarray,
                         cfg: BLSConfig) -> np.ndarray:
    N = X_task.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.float64)
    pos = (y_task == 1)
    neg = ~pos
    X_pos = X_task[pos]
    X_neg = X_task[neg]
    C_pos = X_pos.mean(axis=0) if X_pos.shape[0] else np.zeros(X_task.shape[1])
    C_neg = X_neg.mean(axis=0) if X_neg.shape[0] else np.zeros(X_task.shape[1])
    d_pos_self = np.linalg.norm(X_pos - C_pos, axis=1) if X_pos.shape[0] else np.array([1.0])
    d_neg_self = np.linalg.norm(X_neg - C_neg, axis=1) if X_neg.shape[0] else np.array([1.0])
    R_pos = float(d_pos_self.max()) if d_pos_self.size else 1.0
    R_neg = float(d_neg_self.max()) if d_neg_self.size else 1.0
    d_pos = np.linalg.norm(X_task - C_pos, axis=1)
    d_neg = np.linalg.norm(X_task - C_neg, axis=1)
    theta = np.where(pos, 1.0 - d_pos / (R_pos + cfg.delta),
                     1.0 - d_neg / (R_neg + cfg.delta))
    return np.clip(theta, 0.0, 1.0)


def ifs_scores_per_task(X_task: np.ndarray, y_task: np.ndarray,
                        cfg: BLSConfig,
                        rng: np.random.Generator) -> np.ndarray:
    N = X_task.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.float64)

    if cfg.mu_kernel is not None:
        mu = float(cfg.mu_kernel)
    else:
        # Median heuristic: mu = median pairwise distance among task samples.
        # Robust to TF-IDF / sparse / high-dim feature spaces where sqrt(D) is wrong.
        n_sub = min(N, 256)
        idx = np.linspace(0, N - 1, n_sub, dtype=np.int64)
        Xs = X_task[idx]
        sq = np.sum(Xs * Xs, axis=1, keepdims=True)
        d2 = np.maximum(sq + sq.T - 2.0 * (Xs @ Xs.T), 0.0)
        d = np.sqrt(d2[np.triu_indices(n_sub, k=1)])
        mu = float(np.median(d)) if d.size else 1.0
        mu = max(mu, 1e-3)

    pos = (y_task == 1)
    neg = ~pos
    X_pos = X_task[pos]
    X_neg = X_task[neg]

    # Per-sample distance to its own class centroid (MATLAB radiusxp/xn)
    if X_pos.shape[0]:
        K1 = gaussian_kernel(X_pos, X_pos, mu)
        radiusxp = np.sqrt(np.maximum(1.0 - 2.0 * K1.mean(axis=1) + K1.mean(), 0.0))
        R_pos = float(radiusxp.max()) if radiusxp.size else 1.0
    else:
        radiusxp = np.array([])
        R_pos = 1.0
    if X_neg.shape[0]:
        K2 = gaussian_kernel(X_neg, X_neg, mu)
        radiusxn = np.sqrt(np.maximum(1.0 - 2.0 * K2.mean(axis=1) + K2.mean(), 0.0))
        R_neg = float(radiusxn.max()) if radiusxn.size else 1.0
    else:
        radiusxn = np.array([])
        R_neg = 1.0

    # Authors' alpha_d = max class radius (Eq. 12 neighbourhood)
    alpha_d = float(max(R_pos, R_neg))
    alpha_d = max(alpha_d, 1e-8)

    # Per-sample membership theta(x_r), class-conditional (Eq. 11)
    theta = np.empty(N, dtype=np.float64)
    j_pos = j_neg = 0
    for i in range(N):
        if pos[i]:
            theta[i] = 1.0 - radiusxp[j_pos] / (R_pos + cfg.delta)
            j_pos += 1
        else:
            theta[i] = 1.0 - radiusxn[j_neg] / (R_neg + cfg.delta)
            j_neg += 1
    # Authors don't clip; theta is naturally in [0,1] up to the +delta buffer.
    theta = np.clip(theta, 0.0, 1.0)

    # Non-membership Theta(x_r): heterogeneous-neighbor fraction (Eq. 12)
    K3 = gaussian_kernel(X_task, X_task, mu)
    DD = np.sqrt(np.maximum(2.0 * (1.0 - K3), 0.0))  # RKHS chord distance, real
    in_nbhd = DD < alpha_d  # MATLAB: strict <, includes self
    same_class = y_task[:, None] == y_task[None, :]
    hetero = in_nbhd & (~same_class)
    nbhd_sizes = in_nbhd.sum(axis=1).astype(np.float64)
    Theta = np.where(nbhd_sizes > 0,
                     hetero.sum(axis=1) / np.maximum(nbhd_sizes, 1.0),
                     0.0)
    nontheta = (1.0 - theta) * Theta  # nu_tilde(x_r) per Eq. 12

    # IFS score Eq. 13 (three-branch); authors do NOT clip
    score = np.where(
        nontheta == 0.0, theta,
        np.where(theta <= nontheta, 0.0,
                 (1.0 - nontheta) / np.maximum(2.0 - theta - nontheta, 1e-8)),
    )
    return score


def weighted_ridge_solve(G: np.ndarray, T: np.ndarray, S: np.ndarray,
                         C: float) -> np.ndarray:
    N, Fdim = G.shape
    S2 = S * S  # (N,)

    if Fdim <= N:
        # Eq. (25): W = (G' S^2 G + I/C)^-1 G' S^2 T - invert F x F
        GtS2 = G.T * S2[None, :]  # (F, N)
        A = GtS2 @ G + (1.0 / C) * np.eye(Fdim)  # (F, F)
        rhs = GtS2 @ T  # (F, C)
        return np.linalg.solve(A, rhs)
    else:
        # Eq. (30): W = G' (I/C + S^2 G G')^-1 S^2 T - invert N x N
        GGt = G @ G.T  # (N, N)
        A = (1.0 / C) * np.eye(N) + (S2[:, None] * GGt)  # (N, N)
        rhs = S2[:, None] * T  # (N, C)
        lam = np.linalg.solve(A, rhs)
        return G.T @ lam  # (F, C)


class IFBLSMultiTask:

    def __init__(self, tasks: List[str], cfg: BLSConfig, seed: int = 42):
        self.tasks = list(tasks)
        self.task_to_idx = {t: i for i, t in enumerate(self.tasks)}
        self.cfg = cfg
        self.seed = seed
        self._params: Optional[Dict] = None
        self._W: Optional[Dict[str, np.ndarray]] = None

    def fit(self, X: np.ndarray, observed_tasks: List[str],
            observed_labels: np.ndarray) -> Dict[str, float]:
        rng = np.random.default_rng(self.seed)
        G_all, self._params = build_random_features(X, self.cfg, rng)

        # Targets: map {0,1} -> {-1,+1} for ridge regression, single column.
        T_all = (2 * observed_labels.astype(np.int64) - 1).astype(np.float64)[:, None]

        diag: Dict[str, float] = {}
        self._W = {}
        for t in self.tasks:
            mask = np.array([ot == t for ot in observed_tasks])
            n_t = int(mask.sum())
            if n_t == 0:
                self._W[t] = np.zeros((G_all.shape[1], 1))
                diag[f"{t}_n"] = 0
                continue
            X_t = X[mask]
            y_t = observed_labels[mask].astype(np.int64)
            G_t = G_all[mask]
            T_t = T_all[mask]
            if self.cfg.score_mode == "fbls":
                S_t = fbls_scores_per_task(X_t, y_t, self.cfg)
            else:
                # Cap N for the IFS kernel computation: full N*N matrix
                # is intractable for >10k-sample tasks (e.g. SST2).
                # Subsample to a representative set, compute scores there,
                # propagate to other samples by NN-membership inheritance.
                if X_t.shape[0] > self.cfg.max_kernel_n:
                    N_full = X_t.shape[0]
                    sub_idx = rng.choice(N_full, size=self.cfg.max_kernel_n, replace=False)
                    sub_idx.sort()  # keep order for reproducibility
                    S_sub = ifs_scores_per_task(X_t[sub_idx], y_t[sub_idx], self.cfg, rng)
                    S_t = np.full(N_full, S_sub.mean())  # fallback for non-sampled
                    S_t[sub_idx] = S_sub
                    diag[f"{t}_subsampled"] = self.cfg.max_kernel_n
                else:
                    S_t = ifs_scores_per_task(X_t, y_t, self.cfg, rng)
            if self.cfg.score_normalize and S_t.size and S_t.max() > 0:
                S_t = S_t / S_t.max()
            W_t = weighted_ridge_solve(G_t, T_t, S_t, self.cfg.C)
            self._W[t] = W_t
            diag[f"{t}_n"] = n_t
            diag[f"{t}_score_mean"] = float(S_t.mean())
            diag[f"{t}_score_std"] = float(S_t.std())
        return diag

    def _scores(self, X: np.ndarray) -> np.ndarray:
        assert self._params is not None and self._W is not None, "fit() first"
        G = transform_random_features(X, self._params)  # (N, F)
        K = len(self.tasks)
        out = np.empty((X.shape[0], K), dtype=np.float64)
        for i, t in enumerate(self.tasks):
            out[:, i] = (G @ self._W[t]).squeeze(-1)
        return out

    def predict_known(self, X: np.ndarray, true_tasks: List[str]) -> np.ndarray:
        scores = self._scores(X)
        N = X.shape[0]
        chosen = np.array([self.task_to_idx[t] for t in true_tasks], dtype=np.int64)
        own = scores[np.arange(N), chosen]
        return (own > 0).astype(np.int64)

    def predict_blind(self, X: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        scores = self._scores(X)
        conf = np.abs(scores)
        hat_t = conf.argmax(axis=1)
        N = X.shape[0]
        chosen = scores[np.arange(N), hat_t]
        return (chosen > 0).astype(np.int64), hat_t
