from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.trainer.bert_robust_loss_trainer import BertRobustLossTrainer


class BertNoiseAwareTrainer(BertRobustLossTrainer):

    def __init__(self, *args, mtlnl_anchor_frac: float = 0.02,
                 **kwargs):
        self._mtlnl_anchor_frac = mtlnl_anchor_frac
        super().__init__(*args, **kwargs)

    # Override train() to add end-of-epoch hook. Copy the parent train loop
    # body, inject self._end_of_epoch_hook(epoch) at the right spot.
    def train(self, num_epochs: int, batch_size: int, weight_decay: float,
              scheduler_type: str, warmup_ratio: float,
              max_gradient_clip: float, out_dir: str,
              lr: float = 1e-4, max_batches=None, **kwargs):
        import os
        from pathlib import Path
        from pprint import pformat
        from src.common.seed_utils import seeded_generator
        from src.models.multitask_cls import BaseMultitaskModel
        from src.utils import create_results_dataframe

        dataset = self.load_dataset()
        dataset.set_mode(Mode.TRAIN)
        dataset_length = len(dataset)
        total_training_steps = int(dataset_length * num_epochs / batch_size)
        optimizer, lr_scheduler = self.setup_optimizer(
            lr, total_training_steps, weight_decay, scheduler_type, warmup_ratio)

        eval_datasets = {task: self.load_dataset(task) for task in self.tasks}
        noisy_eval_datasets = {
            task: self.load_dataset(task, noisy=True) for task in self.tasks
        }

        # Remember train dataset for hook access
        self._train_dataset = dataset
        self._batch_size = batch_size
        self._task_to_idx = {t.value: i for i, t in enumerate(self.tasks)}

        max_score = 0.0
        max_epoch = 0
        for epoch in range(num_epochs):
            self._current_epoch = epoch
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            for model in self.model_dict.values():
                model.train()
            dataset.set_mode(Mode.TRAIN)
            train_loader = DataLoader(
                dataset=dataset, batch_size=batch_size,
                shuffle=True, collate_fn=dataset.collate_fn,
                generator=seeded_generator(self.noise_seed + epoch),
            )
            epoch_loss, train_score = self.train_epoch(
                train_dataloader=train_loader, optimizer=optimizer,
                lr_scheduler=lr_scheduler, max_gradient_clip=max_gradient_clip,
                max_batches=max_batches,
            )
            logger.info(f"Epoch loss: {epoch_loss:.4f}")
            logger.info(f"Train score:\n{pformat(train_score)}")

            # End-of-epoch hook (MTL-NL T estimation / ExcessMTL weight update)
            self._end_of_epoch_hook(epoch)

            eval_results, aggregated = self._run_eval_epoch(
                eval_datasets, noisy_eval_datasets, batch_size)

            model = self.model_dict[self.task_name]
            if isinstance(model, BaseMultitaskModel):
                model.save_eval_logs(out_dir, prefix="eval", epoch=epoch)
                model.log_eval_stats(verbose=True)
                model.reset_eval_logs()

            if aggregated > max_score:
                self.save(output_dir=out_dir)
                max_score = aggregated
                max_epoch = epoch

            logger.info(f"Best score (known): {max_score:.4f} at epoch {max_epoch}")
            logger.info(f"\nEval scores:\n{create_results_dataframe(eval_results)}")
            self.add_training_log(eval_results)

    @torch.no_grad()
    def _end_of_epoch_hook(self, epoch: int):
        model = self.model_dict[self.task_name]
        loss_type = model.loss_type

        if loss_type == "mtlnl":
            # Estimate T once after epoch 0; freeze for the rest of training.
            if model.mtlnl_T_frozen or epoch != 0:
                return
            self._estimate_mtlnl_T()
            model.mtlnl_T_frozen = True
        elif loss_type == "excessmtl":
            # Update task weights every epoch from val excess risk.
            self._update_excessmtl_weights()

    @torch.no_grad()
    def _estimate_mtlnl_T(self):
        model = self.model_dict[self.task_name]
        model.eval()
        dataset = self._train_dataset
        dataset.set_mode(Mode.TRAIN)
        loader = DataLoader(dataset=dataset, batch_size=self._batch_size,
                            shuffle=False, collate_fn=dataset.collate_fn)
        n_tasks = len(self.tasks)
        # by_task[t] = list of (P(y=0|x), P(y=1|x), observed_label)
        by_task = {t: [] for t in range(n_tasks)}
        for batch in loader:
            input_batch = {k: v for k, v in batch.items()
                           if k not in (TASK_NAME, LABELS, "true_task_name",
                                        "true_label", "sample_idx")}
            if IS_CUDA_AVAILABLE:
                input_batch = {k: v.cuda() for k, v in input_batch.items()}
            bert_out = model.bert(**input_batch)
            pooled = model.dropout(bert_out.pooler_output)
            tasks_in_batch = batch[TASK_NAME]
            labels = batch[LABELS].cpu().tolist()
            for i, t_name in enumerate(tasks_in_batch):
                logits = model.classifier_dict[t_name](pooled[i].unsqueeze(0))
                probs = nn.functional.softmax(logits, dim=-1).cpu().squeeze(0)
                if labels[i] == -1: continue  # skip eval rows
                t_idx = self._task_to_idx[t_name]
                by_task[t_idx].append((probs[0].item(), probs[1].item(), labels[i]))

        anchor_K = max(20, int(self._mtlnl_anchor_frac * sum(len(v) for v in by_task.values()) / max(n_tasks, 1)))
        T = torch.eye(2).unsqueeze(0).expand(n_tasks, 2, 2).clone()
        offdiag_logs = []
        for t in range(n_tasks):
            rows = by_task[t]
            if len(rows) < 2: continue
            # Class 0 anchors: top-K by P(y=0|x); class 1 anchors: top-K by P(y=1|x).
            rows0 = sorted(rows, key=lambda r: -r[0])[:anchor_K]
            rows1 = sorted(rows, key=lambda r: -r[1])[:anchor_K]
            for true_class, anchors in [(0, rows0), (1, rows1)]:
                if not anchors: continue
                n_obs0 = sum(1 for r in anchors if r[2] == 0)
                n_obs1 = sum(1 for r in anchors if r[2] == 1)
                total = n_obs0 + n_obs1
                if total > 0:
                    T[t, true_class, 0] = n_obs0 / total
                    T[t, true_class, 1] = n_obs1 / total
            offdiag = (T[t, 0, 1].item() + T[t, 1, 0].item()) / 2
            offdiag_logs.append(round(offdiag, 3))

        model.mtlnl_T.copy_(T.to(model.mtlnl_T.device))
        model.train()
        logger.info(f"MTL-NL T frozen after epoch 0: "
                    f"per-task off-diag means = {offdiag_logs} "
                    f"(anchor_K={anchor_K} per (task, class))")

    @torch.no_grad()
    def _update_excessmtl_weights(self):
        model = self.model_dict[self.task_name]
        model.eval()
        n_tasks = len(self.tasks)

        def per_task_loss(mode):
            self._train_dataset.set_mode(mode)
            loader = DataLoader(dataset=self._train_dataset, batch_size=self._batch_size,
                                shuffle=False, collate_fn=self._train_dataset.collate_fn)
            sums = [0.0] * n_tasks;
            counts = [0] * n_tasks
            for batch in loader:
                input_batch = {k: v for k, v in batch.items()
                               if k not in (TASK_NAME, LABELS, "true_task_name",
                                            "true_label", "sample_idx")}
                if IS_CUDA_AVAILABLE:
                    input_batch = {k: v.cuda() for k, v in input_batch.items()}
                bert_out = model.bert(**input_batch)
                pooled = model.dropout(bert_out.pooler_output)
                tasks_in_batch = batch[TASK_NAME]
                labels = batch[LABELS]
                if IS_CUDA_AVAILABLE: labels = labels.cuda()
                for i, t_name in enumerate(tasks_in_batch):
                    if labels[i].item() == -1: continue
                    logits = model.classifier_dict[t_name](pooled[i].unsqueeze(0))
                    ce = nn.functional.cross_entropy(logits, labels[i].unsqueeze(0)).item()
                    t_idx = self._task_to_idx[t_name]
                    sums[t_idx] += ce;
                    counts[t_idx] += 1
            return [s / max(c, 1) for s, c in zip(sums, counts)]

        train_loss_pt = per_task_loss(Mode.TRAIN)
        val_loss_pt = per_task_loss(Mode.VALIDATION)
        self._train_dataset.set_mode(Mode.TRAIN)
        # He et al. 2024 ICML "Robust Multi-Task Learning with Excess Risks":
        # min-max formulation - up-weight tasks with high excess risk so the
        # model focuses on the worst-performing task. weights scale with excess_risk.
        excess = [max(v - t, 1e-3) for v, t in zip(val_loss_pt, train_loss_pt)]
        mean_excess = sum(excess) / len(excess)
        weights = [e / mean_excess for e in excess]  # normalised: mean=1
        model.excessmtl_weights.copy_(torch.tensor(weights, dtype=torch.float32,
                                                   device=model.excessmtl_weights.device))
        model.train()
        logger.info(f"ExcessMTL weights: {[round(w, 3) for w in weights]} "
                    f"(train_loss={[round(t, 3) for t in train_loss_pt]}, "
                    f"val_loss={[round(v, 3) for v in val_loss_pt]})")
