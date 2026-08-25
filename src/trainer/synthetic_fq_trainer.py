import json
import os
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.common.seed_utils import set_seed, seeded_generator
from src.data_processors.synthetic_dataset import (
    NoisySyntheticDataset, SyntheticGaussianDataset,
    SyntheticNoisyTestDataset,
)
from src.models.encoder_ifs import MLPEncoder
from src.models.fq_encoder import FQEncoderModel


class SyntheticFQTrainer:

    def __init__(
            self,
            tasks: List[str],
            epsilon_t: float,
            noise_seed: int,
            input_dim: int = 2,
            hidden_dims: Optional[List[int]] = None,
            encoder_output_dim: int = 16,
            dropout: float = 0.0,
            lambda_hes: float = 1.0,
            class_noise_rho: float = 0.0,
            class_noise_rho_indep: float = 0.0,
    ):
        self.tasks = tasks
        self.epsilon_t = epsilon_t
        self.noise_seed = noise_seed
        self.class_noise_rho = class_noise_rho
        self.class_noise_rho_indep = class_noise_rho_indep

        set_seed(noise_seed)

        encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims if hidden_dims is not None else [32, 16],
            output_dim=encoder_output_dim,
            dropout=dropout,
        )
        self.model = FQEncoderModel(
            encoder=encoder,
            hidden_dim=encoder_output_dim,
            tasks=tasks,
            lambda_hes=lambda_hes,
        )
        if IS_CUDA_AVAILABLE:
            self.model.cuda()

    def train(self, num_epochs: int, batch_size: int, lr: float, out_dir: str,
              n_per_class: int = 500, sigma: float = 1.0,
              max_batches: Optional[int] = None):
        os.makedirs(out_dir, exist_ok=True)
        K = len(self.tasks)
        train_ds = NoisySyntheticDataset(
            epsilon_t=self.epsilon_t,
            class_noise_rho=self.class_noise_rho,
            class_noise_rho_indep=self.class_noise_rho_indep,
            n_per_class=n_per_class, sigma=sigma, seed=self.noise_seed,
            num_tasks=K,
        )
        val_ds = SyntheticGaussianDataset(
            n_per_class=n_per_class, sigma=sigma, seed=self.noise_seed, num_tasks=K)
        noisy_test_ds = SyntheticNoisyTestDataset(
            dataset=SyntheticGaussianDataset(
                n_per_class=n_per_class, sigma=sigma, seed=self.noise_seed, num_tasks=K),
            tasks=self.tasks, epsilon_t=self.epsilon_t, seed=self.noise_seed,
        )

        logger.info(
            f"FQ synthetic: epsilon_t={self.epsilon_t}, rho={self.class_noise_rho:.2f}, "
            f"actual task-noise rate={train_ds.noise_rate:.3f}, "
            f"class-flip rate={train_ds.class_flip_rate:.3f}, "
            f"lambda_hes={self.model.lambda_hes}"
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)
        history: List[Dict[str, Any]] = []
        best_blind_acc = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            train_ds.set_mode(Mode.TRAIN)
            self.model.train()
            loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                collate_fn=train_ds.collate_fn,
                generator=seeded_generator(self.noise_seed + epoch),
            )
            epoch_loss = self._train_epoch(loader, optimizer, max_batches)
            logger.info(f"  train loss: {epoch_loss:.4f}")

            results = self._eval_epoch(val_ds, noisy_test_ds, batch_size)
            logger.info(f"  eval: {pformat(results)}")
            history.append({"epoch": epoch, **results})

            if results.get("acc_blind", 0.0) > best_blind_acc:
                best_blind_acc = results["acc_blind"]

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"FQ training complete. Best acc_blind={best_blind_acc:.4f}")

    def _train_epoch(self, loader: DataLoader, optimizer: AdamW,
                     max_batches: Optional[int]) -> float:
        total_loss = 0.0
        i = -1
        for i, batch in enumerate(tqdm(loader, desc="train", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            optimizer.zero_grad()
            loss, _ = self.model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, i + 1)

    def _eval_epoch(self, val_ds, noisy_test_ds, batch_size: int) -> Dict[str, float]:
        acc_known, _ = self._eval_split(val_ds, batch_size, blind_override=False)
        acc_blind, task_acc = self._eval_split(val_ds, batch_size, blind_override=True)
        noisy_test_ds.set_mode(Mode.VALIDATION)
        acc_noisy = self._eval_noisy(noisy_test_ds, batch_size)
        return {
            "acc_known": round(acc_known, 4),
            "acc_blind": round(acc_blind, 4),
            "task_acc": round(task_acc, 4),
            "acc_noisy_test": round(acc_noisy, 4),
        }

    @torch.no_grad()
    def _eval_split(self, ds, batch_size: int, blind_override: bool) -> Tuple[float, float]:
        ds.set_mode(Mode.VALIDATION)
        self.model.eval()
        self.model.reset_eval_logs()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=ds.collate_fn)
        correct_label, correct_task, total = 0, 0, 0
        for batch in loader:
            labels = batch[LABELS]
            task_names = batch[TASK_NAME]
            _, preds = self.model(batch, blind_override=blind_override)
            pred_labels = preds.argmax(dim=1).cpu()
            correct_label += (pred_labels == labels).sum().item()
            if blind_override:
                # Routing accuracy: re-compute conf via the model's internal helper and compare its argmax task to the true task.
                mu, pi, nu = self.model._compute_ifs_triples(batch)
                conf = (1 - pi) * torch.max(mu, nu)
                pred_tasks = torch.argmax(conf, dim=1).cpu().tolist()
                for pt, tn in zip(pred_tasks, task_names):
                    if self.model.idx_to_task[pt] == tn:
                        correct_task += 1
            total += len(labels)
        self.model.train()
        return correct_label / total if total > 0 else 0.0, (
            correct_task / total if blind_override and total > 0 else 0.0
        )

    @torch.no_grad()
    def _eval_noisy(self, noisy_ds, batch_size: int) -> float:
        self.model.eval()
        self.model.reset_eval_logs()
        loader = DataLoader(noisy_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=noisy_ds.collate_fn)
        correct, total = 0, 0
        for batch in loader:
            labels = batch[LABELS]
            _, preds = self.model(batch, blind_override=False)
            pred_labels = preds.argmax(dim=1).cpu()
            correct += (pred_labels == labels).sum().item()
            total += len(labels)
        self.model.train()
        return correct / total if total > 0 else 0.0
