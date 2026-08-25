import argparse
import json
import os
import time
from typing import List

import numpy as np

from src.common.consts import LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.common.seed_utils import set_seed
from src.models.if_bls import BLSConfig, IFBLSMultiTask


def _flatten_split(ds, mode: Mode):
    ds.set_mode(mode)
    n = len(ds)
    examples = [ds[i] for i in range(n)]
    batch = ds.collate_fn(examples)
    if "features" in batch:
        X = batch["features"].cpu().numpy().astype(np.float64)
    else:
        X = batch["input_features"].cpu().numpy().astype(np.float64)
    y = batch[LABELS].cpu().numpy().astype(np.int64)
    tasks = list(batch[TASK_NAME])
    return X, y, tasks


def load_synthetic(args):
    from src.data_processors.synthetic_dataset import (
        NoisySyntheticDataset, SyntheticGaussianDataset,
    )
    train_ds = NoisySyntheticDataset(
        epsilon_t=args.epsilon_t,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        n_per_class=args.n_per_class, sigma=args.sigma, seed=args.noise_seed,
    )
    val_ds = SyntheticGaussianDataset(
        n_per_class=args.n_per_class, sigma=args.sigma, seed=args.noise_seed,
    )
    return train_ds, val_ds


def load_blitzer(args):
    from src.data_processors.blitzer_dataset import NoisyBlitzerDataset
    train_ds = NoisyBlitzerDataset(
        epsilon_t=args.epsilon_t,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        seed=args.noise_seed,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
    )
    return train_ds, train_ds


LOADERS = {
    "synthetic": load_synthetic,
    "blitzer": load_blitzer
}


def main() -> None:
    p = argparse.ArgumentParser(description="IF-BLS baseline (closed-form, single solve)")
    p.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    p.add_argument("--epsilon_t", type=float, default=0.0)
    p.add_argument("--class_noise_rho", type=float, default=0.0)
    p.add_argument("--class_noise_rho_indep", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)

    # Synthetic-only knobs
    p.add_argument("--n_per_class", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.8)
    # Blitzer / GLUE-TFIDF knobs
    p.add_argument("--max_features", type=int, default=2000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=5)

    # IF-BLS hyperparameters 
    p.add_argument("--bls_m", type=int, default=10,
                   help="number of feature groups")
    p.add_argument("--bls_p", type=int, default=10,
                   help="nodes per feature group")
    p.add_argument("--bls_l", type=int, default=1,
                   help="number of enhancement groups")
    p.add_argument("--bls_q", type=int, default=200,
                   help="nodes per enhancement group")
    p.add_argument("--bls_C", type=float, default=100.0,
                   help="ridge regularization (1/C in the closed-form solve; "
                        "authors' MATLAB credit_approval default = 100)")
    p.add_argument("--bls_mu_kernel", type=float, default=None,
                   help="Gaussian kernel width; default sqrt(D)")
    p.add_argument("--bls_delta", type=float, default=1e-4)
    # NOTE: --bls_eps_quantile removed; the authors' MATLAB sets eps = max class radius.
    p.add_argument("--bls_activation", default="tanh",
                   choices=["tanh", "sigmoid", "relu"])
    p.add_argument("--bls_score_mode", default="ifs",
                   choices=["ifs", "fbls"],
                   help="ifs = paper Eqs. 11-13 kernel-space (default); "
                        "fbls = paper Eq. 7 raw-space (use for high-dim sparse "
                        "data where Gaussian kernel collapses).")
    p.add_argument("--bls_score_normalize", type=int, default=0,
                   help="1 = rescale per-task scores so max=1 (deviation from "
                        "paper, equivalent to scaling C by 1/max(S)^2); "
                        "0 = use raw scores per paper (default).")
    p.add_argument("--bls_feature_normalize", type=int, default=1,
                   help="mapminmax columnwise on F before concat (paper authors' default)")
    p.add_argument("--bls_orth_enhancement", type=int, default=1,
                   help="orth() on enhancement weight matrix (paper authors' default)")

    # Compatibility shims for existing dispatcher scripts (ignored here).
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_type", default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--max_gradient_clip", type=float, default=1.0)
    p.add_argument("--lambda_other", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.noise_seed)
    t0 = time.time()

    train_ds, val_ds = LOADERS[args.dataset](args)
    tasks = list(train_ds.task_names)
    logger.info(f"[IF-BLS] dataset={args.dataset} tasks={tasks} D={train_ds.input_dim}")

    X_train, y_train, t_train = _flatten_split(train_ds, Mode.TRAIN)
    X_val, y_val, t_val = _flatten_split(val_ds, Mode.VALIDATION)
    logger.info(f"[IF-BLS] N_train={len(y_train)} N_val={len(y_val)} D={X_train.shape[1]} "
                f"setup={time.time() - t0:.1f}s")

    cfg = BLSConfig(
        m=args.bls_m, p=args.bls_p, l=args.bls_l, q=args.bls_q,
        C=args.bls_C, mu_kernel=args.bls_mu_kernel, delta=args.bls_delta,
        activation=args.bls_activation,
        feature_normalize=bool(args.bls_feature_normalize),
        orth_enhancement=bool(args.bls_orth_enhancement),
        score_mode=args.bls_score_mode,
        score_normalize=bool(args.bls_score_normalize),
    )
    model = IFBLSMultiTask(tasks=tasks, cfg=cfg, seed=args.noise_seed)

    t1 = time.time()
    diag = model.fit(X_train, t_train, y_train)
    logger.info(f"[IF-BLS] fit done in {time.time() - t1:.1f}s")
    for k, v in sorted(diag.items()):
        logger.info(f"  {k} = {v}")

    # Evaluate
    pred_known = model.predict_known(X_val, t_val)
    acc_known = float((pred_known == y_val).mean())

    pred_blind, pred_t = model.predict_blind(X_val)
    acc_blind = float((pred_blind == y_val).mean())

    true_t_idx = np.array([model.task_to_idx[t] for t in t_val], dtype=np.int64)
    task_acc = float((pred_t == true_t_idx).mean())

    # Per-task accuracy under both routings
    per_known = {}
    per_blind = {}
    for t in tasks:
        mask = np.array([tt == t for tt in t_val])
        if mask.sum() == 0:
            continue
        per_known[t] = float((pred_known[mask] == y_val[mask]).mean())
        per_blind[t] = float((pred_blind[mask] == y_val[mask]).mean())

    record = {
        "epoch": 0,
        "acc_known": round(acc_known, 4),
        "acc_blind": round(acc_blind, 4),
        "task_acc": round(task_acc, 4),
        "per_task_known": {t: round(v, 4) for t, v in per_known.items()},
        "per_task_blind": {t: round(v, 4) for t, v in per_blind.items()},
        "fit_diagnostics": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in diag.items()},
        "config": {
            "dataset": args.dataset,
            "epsilon_t": args.epsilon_t,
            "class_noise_rho": args.class_noise_rho,
            "class_noise_rho_indep": args.class_noise_rho_indep,
            "noise_seed": args.noise_seed,
            "bls_m": args.bls_m, "bls_p": args.bls_p,
            "bls_l": args.bls_l, "bls_q": args.bls_q,
            "bls_C": args.bls_C, "bls_mu_kernel": args.bls_mu_kernel,
            "bls_delta": args.bls_delta,
            "bls_activation": args.bls_activation,
            "bls_score_mode": args.bls_score_mode,
            "bls_feature_normalize": bool(args.bls_feature_normalize),
            "bls_orth_enhancement": bool(args.bls_orth_enhancement),
            "bls_score_normalize": bool(args.bls_score_normalize),
        },
        "wallclock_s": round(time.time() - t0, 2),
    }
    history = [record]
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Per-sample predictions for paper-correct MCC-on-CoLA scoring
    preds_log = {
        "true_task": list(t_val),
        "true_label": [int(y) for y in y_val.tolist()],
        "pred_known": [int(y) for y in pred_known.tolist()],
        "pred_blind": [int(y) for y in pred_blind.tolist()],
    }
    with open(os.path.join(args.out_dir, "predictions.json"), "w") as f:
        json.dump(preds_log, f)

    logger.info(f"[IF-BLS] done - acc_known={acc_known:.4f} acc_blind={acc_blind:.4f} "
                f"task_acc={task_acc:.4f} total={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
