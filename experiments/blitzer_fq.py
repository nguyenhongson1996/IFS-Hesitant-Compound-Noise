import argparse
import json
import os
import time
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.common.seed_utils import set_seed, seeded_generator
from src.data_processors.blitzer_dataset import (
    BLITZER_TASKS, NoisyBlitzerDataset,
)
from src.models.fq_encoder import FQEncoderModel, TfidfMLPEncoder


def make_dataset(args, *, dataset_factory=None) -> NoisyBlitzerDataset:
    if dataset_factory is not None:
        return dataset_factory(args)
    return NoisyBlitzerDataset(
        epsilon_t=args.epsilon_t,
        class_noise_rho=args.class_noise_rho,
        class_noise_rho_indep=args.class_noise_rho_indep,
        seed=args.noise_seed,
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
    )


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epsilon_t", type=float, default=0.3)
    p.add_argument("--class_noise_rho", type=float, default=0.0)
    p.add_argument("--class_noise_rho_indep", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument("--num_epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_hes", type=float, default=1.0)
    p.add_argument("--max_features", type=int, default=2000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=5)
    p.add_argument("--encoder_hidden", nargs="+", type=int, default=[64])
    p.add_argument("--encoder_output", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--out_dir", default="results/blitzer_fq")
    return p


def train_fq(args, *, tasks: List[str] = BLITZER_TASKS,
             dataset_factory=None):
    set_seed(args.noise_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    train_ds = make_dataset(args, dataset_factory=dataset_factory)
    train_ds.set_mode(Mode.TRAIN)
    n_train = len(train_ds)
    train_ds.set_mode(Mode.VALIDATION)
    n_val = len(train_ds)
    train_ds.set_mode(Mode.TRAIN)
    logger.info(
        f"FQ {tasks=} input_dim={train_ds.input_dim} train={n_train} val={n_val} "
        f"setup={time.time() - t0:.1f}s eps={args.epsilon_t} rho={args.class_noise_rho} "
        f"rho_indep={args.class_noise_rho_indep}"
    )

    encoder = TfidfMLPEncoder(
        input_dim=train_ds.input_dim,
        hidden_dims=list(args.encoder_hidden),
        output_dim=args.encoder_output,
        dropout=args.dropout,
    )
    model = FQEncoderModel(
        encoder=encoder,
        hidden_dim=args.encoder_output,
        tasks=tasks,
        lambda_hes=args.lambda_hes,
    )
    if IS_CUDA_AVAILABLE:
        model.cuda()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    history: List[Dict[str, Any]] = []

    for epoch in range(args.num_epochs):
        train_ds.set_mode(Mode.TRAIN)
        model.train()
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=train_ds.collate_fn,
            generator=seeded_generator(args.noise_seed + epoch),
        )
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss, _ = model(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(1, n_batches)

        # Eval: three protocols (K, B, NOISY). Save per-sample preds on final epoch.
        is_final = (epoch == args.num_epochs - 1)
        results = eval_three_protocols(model, train_ds, args.batch_size,
                                       return_preds=is_final)
        preds_log = results.pop("preds_log", None)
        history.append({"epoch": epoch, "train_loss": round(avg_loss, 4), **results})
        logger.info(f"Epoch {epoch}/{args.num_epochs - 1} loss={avg_loss:.4f} {pformat(results)}")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    if preds_log is not None:
        with open(os.path.join(args.out_dir, "predictions.json"), "w") as f:
            json.dump(preds_log, f)
    logger.info(f"FQ Blitzer training complete. history.json + predictions.json written.")


@torch.no_grad()
def eval_three_protocols(model: FQEncoderModel, dataset, batch_size: int,
                         return_preds: bool = False) -> Dict[str, float]:
    # K + B share the same eval (Mode.VALIDATION = clean task names)
    dataset.set_mode(Mode.VALIDATION)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=dataset.collate_fn)

    correct_k, correct_b, correct_task, total = 0, 0, 0, 0
    preds_log = {"true_task": [], "true_label": [], "pred_known": [], "pred_blind": []} if return_preds else None
    for batch in loader:
        labels = batch[LABELS]
        task_names = batch[TASK_NAME]
        # K
        _, preds_k = model(batch, blind_override=False)
        pl_k = preds_k.argmax(dim=1).cpu()
        correct_k += (pl_k == labels).sum().item()
        # B
        _, preds_b = model(batch, blind_override=True)
        pl_b = preds_b.argmax(dim=1).cpu()
        correct_b += (pl_b == labels).sum().item()
        # task routing
        mu, pi, nu = model._compute_ifs_triples(batch)
        conf = (1 - pi) * torch.max(mu, nu)
        pred_tasks = torch.argmax(conf, dim=1).cpu().tolist()
        for pt, tn in zip(pred_tasks, task_names):
            if model.idx_to_task[pt] == tn:
                correct_task += 1
        total += len(labels)
        if preds_log is not None:
            for i, tn in enumerate(task_names):
                preds_log["true_task"].append(tn)
                preds_log["true_label"].append(int(labels[i].item()))
                preds_log["pred_known"].append(int(pl_k[i].item()))
                preds_log["pred_blind"].append(int(pl_b[i].item()))

    # FQ has no separate noisy-test split here; reuse observed-task eval.
    # Report acc_noisy_test = acc_known for parity with the current runner.
    model.train()
    acc_k = correct_k / total if total > 0 else 0.0
    acc_b = correct_b / total if total > 0 else 0.0
    task_acc = correct_task / total if total > 0 else 0.0
    out = {
        "acc_known": round(acc_k, 4),
        "acc_blind": round(acc_b, 4),
        "task_acc": round(task_acc, 4),
        "acc_noisy_test": round(acc_k, 4),  # Reuses observed-task eval.
    }
    if preds_log is not None:
        out["preds_log"] = preds_log
    return out


def main():
    args, unk_args = build_arg_parser().parse_known_args()
    train_fq(args)


if __name__ == "__main__":
    main()
