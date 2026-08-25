from bisect import bisect
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from src.common.consts import (ATTENTION_MASK, INPUT_IDS, LABEL, LABELS,
                               TOKEN_TYPE_IDS, TASK_NAME, GlueTask, Mode)
from src.data_processors.ensemble_dataset import EnsembleNLIDataset
from src.data_processors.nli_data_processor import pad_tensor_list

TRUE_TASK_NAME = "true_task_name"
TRUE_LABEL = "true_label"
SAMPLE_IDX = "sample_idx"


class NoisyEnsembleNLIDataset(EnsembleNLIDataset):

    def __init__(self, tasks: List[GlueTask], tokenizer: PreTrainedTokenizer,
                 epsilon_t: float = 0.2, class_noise_rho: float = 0.0,
                 class_noise_rho_indep: float = 0.0,
                 seed: int = 42):
        super().__init__(tasks, tokenizer)
        self.epsilon_t = epsilon_t
        self.class_noise_rho = class_noise_rho
        self.class_noise_rho_indep = class_noise_rho_indep
        self._build_noise_map(seed)

    def _build_noise_map(self, seed: int):
        train_size = self._length[Mode.TRAIN]
        rng_task = np.random.RandomState(seed)
        rng_class = np.random.RandomState(seed + 1)
        rng_indep = np.random.RandomState(seed + 2)

        task_mask = rng_task.random(train_size) < self.epsilon_t
        noisy: List[Optional[str]] = [None] * train_size

        for i in range(train_size):
            if task_mask[i]:
                task_idx = bisect(self._tasks_data_offset[Mode.TRAIN], i) - 1
                true_task = self._tasks[task_idx].value
                others = [t.value for t in self._tasks if t.value != true_task]
                noisy[i] = rng_task.choice(others)

        if self.class_noise_rho > 0:
            class_flip = task_mask & (rng_class.random(train_size) < self.class_noise_rho)
        else:
            class_flip = np.zeros(train_size, dtype=bool)

        # Independent class flip when the task identity is correct. Added to address reviewer concern that the original
        # noise model assumes Pr(y_tilde != y_star | t_tilde = t_star) = 0.
        if self.class_noise_rho_indep > 0:
            indep_flip = (~task_mask) & (
                    rng_indep.random(train_size) < self.class_noise_rho_indep)
            class_flip = class_flip | indep_flip

        self._noise_mask = task_mask
        self._noisy_task_names = noisy
        self._class_flip = class_flip

    @property
    def noise_rate(self) -> float:
        return float(self._noise_mask.mean())

    @property
    def class_flip_rate(self) -> float:
        return float(self._class_flip.mean())

    def __getitem__(self, index) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        sample[TRUE_TASK_NAME] = sample[TASK_NAME]  # always record ground truth.
        sample[TRUE_LABEL] = sample[LABEL]  # record pre-flip class label.
        sample[SAMPLE_IDX] = index  # stable position in training set.

        if self._mode == Mode.TRAIN:
            if self._noise_mask[index]:
                sample[TASK_NAME] = self._noisy_task_names[index]
            # Class flip is independent of the task-flip branch:
            # - compound flip (rho): only when task is noisy
            # - independent flip (rho_indep): only when task is correct
            # _build_noise_map combines both channels into _class_flip.
            if self._class_flip[index]:
                sample[LABEL] = 1 - sample[LABEL]

        return sample

    def collate_fn(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        padded = {
            field: pad_tensor_list(
                [ex[field][0] for ex in examples], self._tokenizer.pad_token_id
            )
            for field in [INPUT_IDS, ATTENTION_MASK, TOKEN_TYPE_IDS]
        }
        labels = torch.LongTensor([ex[LABEL] for ex in examples])
        task_names = [ex[TASK_NAME] for ex in examples]
        true_task_names = [ex.get(TRUE_TASK_NAME, tn)
                           for ex, tn in zip(examples, task_names)]
        true_labels = torch.LongTensor(
            [ex.get(TRUE_LABEL, ex[LABEL]) for ex in examples])
        sample_indices = torch.LongTensor([ex.get(SAMPLE_IDX, -1) for ex in examples])
        return {
            **padded,
            LABELS: labels,
            TASK_NAME: task_names,
            TRUE_TASK_NAME: true_task_names,
            TRUE_LABEL: true_labels,
            SAMPLE_IDX: sample_indices,
        }


class NoisyTaskTestDataset:

    def __init__(self, dataset: EnsembleNLIDataset, true_task: GlueTask,
                 tasks: List[GlueTask], epsilon_t: float, seed: int = 42):
        self.dataset = dataset
        self.true_task_value: str = true_task.value
        self.tasks = tasks
        self.epsilon_t = epsilon_t
        # Offset from training seed so test noise is independent.
        self._seed: int = seed + 9999
        self._noise_mask: Optional[np.ndarray] = None
        self._noisy_tasks: Optional[List[Optional[str]]] = None

    def set_mode(self, mode: Mode) -> None:
        self.dataset.set_mode(mode)
        self._prebuild_noise()

    def _prebuild_noise(self) -> None:
        size = len(self.dataset)
        rng = np.random.RandomState(self._seed)  # reset seed -> identical mask every epoch.
        mask = rng.random(size) < self.epsilon_t
        others = [t.value for t in self.tasks if t.value != self.true_task_value]
        self._noise_mask = mask
        self._noisy_tasks = [
            rng.choice(others) if (mask[i] and others) else None
            for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.dataset[idx])  # shallow copy - do not mutate the cache.
        if self._noise_mask is not None and self._noise_mask[idx]:
            item[TASK_NAME] = self._noisy_tasks[idx]
        return item

    @property
    def collate_fn(self) -> Callable:
        return self.dataset.collate_fn
