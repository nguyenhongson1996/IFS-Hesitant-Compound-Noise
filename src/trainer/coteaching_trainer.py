from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler)

from src.common.consts import (IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME,
                               GlueTask)
from src.common.logger_utils import logger
from src.common.seed_utils import seeded_generator
from src.data_processors.noisy_ensemble_dataset import (
    NoisyTaskTestDataset, SAMPLE_IDX, TRUE_TASK_NAME)
from src.models.multitask_cls import (BaseMultitaskModel,
                                      MultitaskClassificationModel)
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer
from src.utils import create_results_dataframe


class CoteachingTrainer(MTDNNNoisyTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build a second BERT-backed network with an independent init.
        # Use the same BERT class but a fresh random initializer for the
        # per-task heads (BERT itself remains pretrained for both).
        self._model_b_key = self.task_name + "_B"
        # Re-init with a different seed so model B's classifier heads
        # diverge from model A's heads (cross-training relies on this).
        bert_b = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path).bert
        self.model_dict[self._model_b_key] = MultitaskClassificationModel(
            bert_b, self.tasks)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self._model_b_key].cuda()

    @staticmethod
    def _per_sample_ce(model: MultitaskClassificationModel,
                       batch: Dict[str, Any]) -> torch.Tensor:
        keys_skip = {TASK_NAME, LABELS, TRUE_TASK_NAME, SAMPLE_IDX,
                     "true_label"}
        input_batch = {k: v for k, v in batch.items() if k not in keys_skip}
        if IS_CUDA_AVAILABLE:
            input_batch = {k: v.cuda() for k, v in input_batch.items()}
        bert_out = model.bert(**input_batch)
        pooled = model.dropout(bert_out.pooler_output)
        labels = batch[LABELS]
        if IS_CUDA_AVAILABLE:
            labels = labels.cuda()
        task_names = batch[TASK_NAME]
        # Per-sample logits at the assigned head
        logits = []
        for i, t in enumerate(task_names):
            cls = model.classifier_dict[t]
            logits.append(cls(pooled[i].unsqueeze(0)))
        logits = torch.cat(logits, dim=0)  # [B, 2]
        return F.cross_entropy(logits, labels, reduction="none"), logits

    @staticmethod
    def _train_step(model: MultitaskClassificationModel,
                    batch: Dict[str, Any],
                    optimizer: AdamW,
                    max_gradient_clip: float) -> torch.Tensor:
        ce, _ = CoteachingTrainer._per_sample_ce(model, batch)
        loss = ce.mean()
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_gradient_clip)
        optimizer.step()
        return loss

    @staticmethod
    def _sub_batch(batch: Dict[str, Any], idx: torch.Tensor
                   ) -> Dict[str, Any]:
        idx_cpu = idx.cpu()
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v[idx_cpu]
            elif isinstance(v, list):
                out[k] = [v[i] for i in idx_cpu.tolist()]
            else:
                out[k] = v
        return out

    def train(self, num_epochs: int, batch_size: int, weight_decay: float,
              scheduler_type: str, warmup_ratio: float,
              max_gradient_clip: float, out_dir: str,
              lr: float = 1e-4, max_batches: Optional[int] = None,
              **kwargs):
        dataset = self.load_dataset()
        dataset.set_mode(Mode.TRAIN)
        logger.info(
            f"[Coteaching] epsilon_t={self.epsilon_t}, "
            f"rho={self.class_noise_rho}, "
            f"rho_indep={self.class_noise_rho_indep}, "
            f"warmup_epochs={self.warmup_epochs}"
        )

        total_steps = int(len(dataset) * num_epochs / batch_size)

        def _make_optimizer(model):
            bert_p, other_p = [], []
            for name, p in model.named_parameters():
                (bert_p if "bert" in name.lower() else other_p).append(p)
            opt = AdamW([{"params": bert_p, "lr": lr / 10},
                         {"params": other_p, "lr": lr}],
                        lr=lr, weight_decay=weight_decay)
            sched = get_scheduler(
                scheduler_type, optimizer=opt,
                num_warmup_steps=int(total_steps * warmup_ratio),
                num_training_steps=total_steps)
            return opt, sched

        model_a = self.model_dict[self.task_name]
        model_b = self.model_dict[self._model_b_key]
        opt_a, sched_a = _make_optimizer(model_a)
        opt_b, sched_b = _make_optimizer(model_b)
        eval_datasets = {t: self.load_dataset(t) for t in self.tasks}
        noisy_eval_datasets = {
            task: NoisyTaskTestDataset(
                self.load_dataset(task), task, self.tasks,
                self.epsilon_t, self.noise_seed)
            for task in self.tasks
        }

        keep_ratio_final = 1.0 - self.epsilon_t
        max_score = 0.0

        for epoch in range(num_epochs):
            self._current_epoch = epoch
            keep_ratio = (1.0 if epoch < self.warmup_epochs
                          else keep_ratio_final)
            phase = ("WARMUP" if epoch < self.warmup_epochs
                     else f"COTEACH(keep={keep_ratio:.0%})")
            logger.info(f"Epoch {epoch}/{num_epochs - 1} - Phase: {phase}")

            model_a.train();
            model_b.train()
            dataset.set_mode(Mode.TRAIN)
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                collate_fn=dataset.collate_fn,
                generator=seeded_generator(self.noise_seed + epoch))

            total_loss = 0.0
            n_batches = 0
            for i, batch in enumerate(tqdm(train_loader, leave=False)):
                if max_batches is not None and i >= max_batches:
                    break
                if epoch < self.warmup_epochs:
                    # Warm-up: each network trains on full batch with CE
                    self._train_step(model_a, batch, opt_a, max_gradient_clip)
                    loss_b = self._train_step(model_b, batch, opt_b,
                                              max_gradient_clip)
                    sched_a.step();
                    sched_b.step()
                    total_loss += loss_b.item()
                else:
                    # Co-teach: small-loss selection on each network's CE
                    with torch.no_grad():
                        ce_a, _ = self._per_sample_ce(model_a, batch)
                        ce_b, _ = self._per_sample_ce(model_b, batch)
                    n_keep = max(1, int(ce_a.size(0) * keep_ratio))
                    idx_a_keeps = torch.argsort(ce_a)[:n_keep]
                    idx_b_keeps = torch.argsort(ce_b)[:n_keep]
                    # A trains on what B selected
                    sub_for_a = self._sub_batch(batch, idx_b_keeps)
                    self._train_step(model_a, sub_for_a, opt_a,
                                     max_gradient_clip)
                    # B trains on what A selected
                    sub_for_b = self._sub_batch(batch, idx_a_keeps)
                    loss_b = self._train_step(model_b, sub_for_b, opt_b,
                                              max_gradient_clip)
                    sched_a.step();
                    sched_b.step()
                    total_loss += loss_b.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            logger.info(f"Epoch loss (B): {avg_loss:.4f}")

            # Eval (uses model A only - Han 2018 inference protocol)
            self.model_dict[self.task_name] = model_a
            eval_results, aggregated = self._run_eval_epoch(
                eval_datasets, noisy_eval_datasets, batch_size)
            if isinstance(model_a, BaseMultitaskModel):
                model_a.save_eval_logs(out_dir, prefix="eval", epoch=epoch)
                model_a.log_eval_stats(verbose=True)
                model_a.reset_eval_logs()

            if aggregated > max_score:
                self.save(output_dir=out_dir)
                max_score = aggregated

            logger.info(f"Best score (known): {max_score:.4f}")
            logger.info(f"\nEval scores:\n"
                        f"{create_results_dataframe(eval_results)}")
            self.add_training_log(eval_results)

    def add_training_log(self, eval_reports: Dict[str, Any]):
        pass
