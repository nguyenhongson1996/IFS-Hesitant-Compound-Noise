from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.consts import GlueTask, IS_CUDA_AVAILABLE, LABELS, Mode, TASK_NAME
from src.common.logger_utils import logger
from src.data_processors.ensemble_dataset import EnsembleNLIDataset
from src.data_processors.noisy_ensemble_dataset import (
    NoisyEnsembleNLIDataset, NoisyTaskTestDataset, TRUE_TASK_NAME)
from src.models.multitask_cls import BaseMultitaskModel
from src.models.noisy_robust_ifs import NoisyRobustIFSModel
from src.common.seed_utils import set_seed, seeded_generator
from src.trainer.nli_trainer import NLIMultitaskIFSTrainer
from src.utils import create_results_dataframe
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NoisyMTLIFSTrainer(NLIMultitaskIFSTrainer):

    def __init__(self, tasks: List[GlueTask],
                 epsilon_t: float = 0.2,
                 noise_seed: int = 42,
                 class_noise_rho: float = 0.0,
                 class_noise_rho_indep: float = 0.0,
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
                 trust_no_gate: bool = False,
                 # Legacy kwargs accepted-but-stored-as-attrs for backward
                 # compatibility with baseline subclass trainers
                 # (CoteachingTrainer, KMeansReassignTrainer, etc.) that
                 # access self.warmup_epochs in their own train() loops.
                 warmup_epochs: Optional[int] = None,
                 post_warmup_loss: Optional[str] = None,
                 **kwargs):
        # Expose legacy attrs on self so subclass trainers can read them.
        self.warmup_epochs = warmup_epochs
        self.post_warmup_loss = post_warmup_loss
        # Compound task-class noise (paper Sec. 2)
        self.epsilon_t = epsilon_t
        self.noise_seed = noise_seed
        self.class_noise_rho = class_noise_rho
        self.class_noise_rho_indep = class_noise_rho_indep
        # IFS-Hesitant hyperparameters (paper Sec. 4.2, Algorithm 1)
        self.w_min = w_min
        self.weight_norm = weight_norm
        self.weight_norm_target = weight_norm_target
        self.lambda_other = lambda_other
        self.lambda_pi = lambda_pi
        self.trust_signal = trust_signal
        self.class_loss = class_loss
        self.l_pred_alpha = l_pred_alpha
        self.l_pred_barrier = l_pred_barrier
        self.head_type = head_type
        self.tnorm = tnorm
        self.bayes_trust = bayes_trust
        self.trust_no_gate = trust_no_gate
        self._current_epoch = 0
        self._eval_blind: Optional[bool] = None  # overrides model.do_blindly_decode during eval
        set_seed(noise_seed)  # fix all RNGs before model init
        super().__init__(tasks=tasks, **kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = NoisyRobustIFSModel(
            self.bert, self.tasks,
            do_blindly_decode=self.do_blindly_decode,
            w_min=self.w_min,
            weight_norm=self.weight_norm,
            weight_norm_target=self.weight_norm_target,
            lambda_other=self.lambda_other,
            lambda_pi=self.lambda_pi,
            trust_signal=self.trust_signal,
            class_loss=self.class_loss,
            l_pred_alpha=self.l_pred_alpha,
            l_pred_barrier=self.l_pred_barrier,
            head_type=self.head_type,
            tnorm=self.tnorm,
            bayes_trust=self.bayes_trust,
            trust_no_gate=self.trust_no_gate,
        )
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def load_dataset(self, task: GlueTask = None, **kwargs) -> EnsembleNLIDataset:
        if task is not None:
            # Per-task evaluation dataset - no noise
            return EnsembleNLIDataset(tasks=[task], tokenizer=self.tokenizer)
        return NoisyEnsembleNLIDataset(
            tasks=self.tasks, tokenizer=self.tokenizer,
            epsilon_t=self.epsilon_t,
            class_noise_rho=self.class_noise_rho,
            class_noise_rho_indep=self.class_noise_rho_indep,
            seed=self.noise_seed,
        )

    def forward_step(self, batch: Dict[str, Any], task=None) -> Tuple[Optional[torch.Tensor], Any]:
        return self.model_dict[self.task_name].forward(
            batch, blind_override=self._eval_blind)

    def train(self, num_epochs: int, batch_size: int, weight_decay: float,
              scheduler_type: str, warmup_ratio: float, max_gradient_clip: float,
              out_dir: str, lr: float = 1e-4, max_batches: Optional[int] = None, **kwargs):

        eval_results: Dict[str, Any] = {}

        dataset = self.load_dataset()  # NoisyEnsembleNLIDataset
        dataset.set_mode(Mode.TRAIN)
        class_flip_rate = getattr(dataset, "class_flip_rate", 0.0)
        ifs_model = self.model_dict[self.task_name]
        # IFS-Hesitant-specific attrs are read defensively so this train()
        # also serves the MoE-gate and other baseline subclasses (whose
        # model class does not have weight_norm_target / w_min).
        wn = getattr(ifs_model, "weight_norm_target", None)
        wmin = getattr(ifs_model, "w_min", None)
        ifs_tag = (f"IFS-Hesitant, head=3-logit decoupled, "
                   f"weight_norm_target={wn}, w_min={wmin}"
                   if wn is not None else f"{type(ifs_model).__name__}")
        logger.info(
            f"Compound task-class noise: epsilon_t={self.epsilon_t}, "
            f"rho={self.class_noise_rho}, rho_indep={self.class_noise_rho_indep}, "
            f"actual task rate={dataset.noise_rate:.3f}, "
            f"class-flip rate={class_flip_rate:.3f}. "
            f"Model: {ifs_tag}"
        )

        total_steps = int(len(dataset) * num_epochs / batch_size)
        optimizer, lr_scheduler = self.setup_optimizer(
            lr, total_steps, weight_decay, scheduler_type, warmup_ratio)

        # Per-task clean evaluation datasets (settings 1 & 2)
        eval_datasets = {task: self.load_dataset(task) for task in self.tasks}
        # Per-task noisy evaluation datasets (setting 3)
        noisy_eval_datasets = {
            task: NoisyTaskTestDataset(
                self.load_dataset(task), task, self.tasks, self.epsilon_t, self.noise_seed)
            for task in self.tasks
        }

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

    def add_training_log(self, eval_reports: Dict[str, Any]):
        # Base class expects plain task-name keys (e.g. "rte"); remap _known variants.
        known = {k[:-len("_known")]: v for k, v in eval_reports.items() if k.endswith("_known")}
        super().add_training_log(known)

    def _run_eval_epoch(self, eval_datasets: Dict, noisy_eval_datasets: Dict,
                        batch_size: int):
        all_results: Dict[str, Any] = {}

        # Setting 1: known task labels -> used for checkpoint selection
        self._eval_blind = False
        task_scores: List[float] = []
        for task, ds in eval_datasets.items():
            score = self.run_evaluation(ds, task, batch_size)
            all_results[f"{task.value}_known"] = score
            if task != GlueTask.MRPC:
                task_scores.append(sum(score["0.5"].values()) / len(score["0.5"]))
        aggregated = sum(task_scores)

        # Setting 2: blind (confidence-based task selection, task labels ignored)
        self._eval_blind = True
        for task, ds in eval_datasets.items():
            score = self.run_evaluation(ds, task, batch_size)
            all_results[f"{task.value}_blind"] = score

        # Setting 3: noisy test task labels (same epsilon_t as training noise)
        self._eval_blind = False
        for task, ds in noisy_eval_datasets.items():
            score = self.run_evaluation(ds, task, batch_size)
            all_results[f"{task.value}_noisy_test"] = score

        self._eval_blind = None  # restore default
        return all_results, aggregated

    @torch.no_grad()
    def _scan_training_data(self, dataset: NoisyEnsembleNLIDataset,
                            batch_size: int, max_batches: Optional[int] = None) -> Dict[str, Any]:
        model = self.model_dict[self.task_name]
        model.eval()
        dataset.set_mode(Mode.TRAIN)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=dataset.collate_fn)

        losses, embeddings, confidence, task_indices, true_tasks = [], [], [], [], []

        for i, batch in enumerate(tqdm(loader, desc="Scanning training set", leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            emb = model.get_batch_embeddings(batch)
            mu, pi, nu = model.memberships_from_embeddings(emb)

            labels = batch[LABELS]
            if IS_CUDA_AVAILABLE:
                labels = labels.cuda()

            t_names = batch[TASK_NAME]
            t_idx = [model.task_to_idx[t] for t in t_names]

            # Per-sample NLL on the (mu, nu) at the observed task. Used by
            # scan-based diagnostic subclasses (kmeans_reassign, dividemix,
            # high_loss_discard); CE-3soft training doesn't call this path.
            idx_b = torch.arange(mu.shape[0], device=mu.device)
            t_idx_t = torch.as_tensor(t_idx, device=mu.device, dtype=torch.long)
            mu_obs = mu[idx_b, t_idx_t]
            nu_obs = nu[idx_b, t_idx_t]
            per_loss = torch.where(labels > 0.5,
                                   -torch.log(mu_obs + 1e-8),
                                   -torch.log(nu_obs + 1e-8))
            conf = (1 - pi) * torch.max(mu, nu)  # [B, T]

            losses.extend(per_loss.cpu().tolist())
            embeddings.append(emb.cpu())
            confidence.append(conf.cpu())
            task_indices.extend(t_idx)
            true_tasks.extend(batch.get(TRUE_TASK_NAME, t_names))

        return {
            "losses": np.array(losses),
            "embeddings": torch.cat(embeddings, dim=0),
            "confidence": torch.cat(confidence, dim=0),
            "task_indices": task_indices,
            "true_tasks": true_tasks,
        }
