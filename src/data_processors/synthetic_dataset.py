from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.common.consts import LABELS, Mode, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import TRUE_TASK_NAME, SAMPLE_IDX

INPUT_DIM: int = 5  # 3 discriminative + 2 shared noise dims.
N_TASKS: int = 3
SYNTHETIC_TASKS: List[str] = ["syn_0", "syn_1", "syn_2"]


def tasks_for(num_tasks: int) -> List[str]:
    return [f"syn_{i}" for i in range(num_tasks)]


def input_dim_for(num_tasks: int) -> int:
    return num_tasks + 2


_DISC_DELTA: float = 2.0  # +/-class mean offset for the discriminative feature.
_NOISE_SIGMA: float = 1.0  # std for non-discriminative (noise) features.
_SHARED_DELTA: float = 2.0  # magnitude of class signal in the 2-D rotated subspace.


# Keep task signal comfortably above background noise so each head can separate its own task from off-task samples.


def _task_direction(t_idx: int, task_orthogonality: float,
                    num_tasks: int = N_TASKS) -> np.ndarray:
    angle = 2.0 * np.pi * float(t_idx) * float(task_orthogonality) / float(num_tasks)
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)


def _generate_samples(
        t_idx: int,
        label: int,
        n: int,
        sigma: float,
        rng: np.random.RandomState,
        task_orthogonality: Optional[float] = None,
        num_tasks: int = N_TASKS,
        input_dim: int = INPUT_DIM,
) -> np.ndarray:
    x = rng.randn(n, input_dim).astype(np.float32) * _NOISE_SIGMA
    sign = +1.0 if label == 1 else -1.0
    if task_orthogonality is not None:
        # Rotated-signal design: discriminative dim x_{t_idx} carries task identity (no class sign), shared dims (last
        # two) carry the rotated class signal.
        x[:, t_idx] = (_DISC_DELTA + rng.randn(n) * sigma).astype(np.float32)
        u = _task_direction(t_idx, task_orthogonality, num_tasks)
        x[:, num_tasks] = (sign * _SHARED_DELTA * u[0]
                           + rng.randn(n).astype(np.float32) * sigma)
        x[:, num_tasks + 1] = (sign * _SHARED_DELTA * u[1]
                               + rng.randn(n).astype(np.float32) * sigma)
    else:
        # Default: x_{t_idx} carries task identity and class sign.
        x[:, t_idx] = (sign * _DISC_DELTA + rng.randn(n) * sigma).astype(np.float32)
    return x


class SyntheticGaussianDataset(Dataset):

    def __init__(self, n_per_class: int = 500, sigma: float = 0.8,
                 val_ratio: float = 0.2, seed: int = 42,
                 task_orthogonality: Optional[float] = None,
                 num_tasks: int = N_TASKS):
        self.n_per_class = n_per_class
        self.sigma = sigma
        self.val_ratio = val_ratio
        self.seed = seed
        self.task_orthogonality = task_orthogonality
        self.num_tasks = num_tasks
        self.task_names = tasks_for(num_tasks)
        self.input_dim = input_dim_for(num_tasks)
        self._mode = Mode.TRAIN
        self._build_splits(np.random.RandomState(seed))

    def _build_splits(self, rng: np.random.RandomState):
        feats, labels, tasks = [], [], []
        for t_idx in range(self.num_tasks):
            for label in (1, 0):
                x = _generate_samples(t_idx, label, self.n_per_class,
                                      self.sigma, rng,
                                      task_orthogonality=self.task_orthogonality,
                                      num_tasks=self.num_tasks,
                                      input_dim=self.input_dim)
                feats.append(x)
                labels.extend([label] * self.n_per_class)
                tasks.extend([self.task_names[t_idx]] * self.n_per_class)

        feats = np.concatenate(feats, axis=0)
        labels = np.array(labels, dtype=np.int64)
        tasks = np.array(tasks)

        N = len(labels)
        perm = rng.permutation(N)
        n_val = int(N * self.val_ratio)
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        self._splits: Dict[Mode, tuple] = {
            Mode.TRAIN: (feats[train_idx], labels[train_idx], tasks[train_idx]),
            Mode.VALIDATION: (feats[val_idx], labels[val_idx], tasks[val_idx]),
            Mode.TEST: (feats[val_idx], labels[val_idx], tasks[val_idx]),
        }

    def set_mode(self, mode: Mode):
        self._mode = mode

    def __len__(self) -> int:
        return len(self._splits[self._mode][1])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feats, labels, tasks = self._splits[self._mode]
        return {
            "features": torch.from_numpy(feats[idx]),
            LABELS: int(labels[idx]),
            TASK_NAME: tasks[idx],
        }

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "features": torch.stack([e["features"] for e in examples]),
            LABELS: torch.tensor([e[LABELS] for e in examples], dtype=torch.long),
            TASK_NAME: [e[TASK_NAME] for e in examples],
        }

    def get_dataloader(self, batch_size: int, shuffle: bool = True,
                       generator=None, **kwargs) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.collate_fn,
                          generator=generator, **kwargs)


class NoisySyntheticDataset(SyntheticGaussianDataset):

    def __init__(self, epsilon_t: float = 0.2, class_noise_rho: float = 0.0,
                 class_noise_rho_indep: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.epsilon_t = epsilon_t
        self.class_noise_rho = class_noise_rho
        self.class_noise_rho_indep = class_noise_rho_indep
        self._build_noise_map(
            rng_task=np.random.RandomState(self.seed + 1),
            rng_class=np.random.RandomState(self.seed + 2),
            rng_indep=np.random.RandomState(self.seed + 3),
        )

    def _build_noise_map(self, rng_task: np.random.RandomState,
                         rng_class: np.random.RandomState,
                         rng_indep: np.random.RandomState):
        _, _, tasks = self._splits[Mode.TRAIN]
        N = len(tasks)
        task_mask = rng_task.random(N) < self.epsilon_t
        noisy_tasks: List[Optional[str]] = [None] * N
        for i in range(N):
            if task_mask[i]:
                others = [t for t in self.task_names if t != tasks[i]]
                noisy_tasks[i] = rng_task.choice(others)

        # Compound class flip: conditional on task being noisy.
        class_flip = np.zeros(N, dtype=bool)
        if self.class_noise_rho > 0:
            conditional = rng_class.random(N) < self.class_noise_rho
            class_flip = task_mask & conditional

        # Independent class flip: conditional on task being correct.
        if self.class_noise_rho_indep > 0:
            indep_conditional = rng_indep.random(N) < self.class_noise_rho_indep
            class_flip = class_flip | (~task_mask & indep_conditional)

        self._noise_mask = task_mask
        self._noisy_tasks = noisy_tasks
        self._class_flip = class_flip

    @property
    def noise_rate(self) -> float:
        return float(self._noise_mask.mean())

    @property
    def class_flip_rate(self) -> float:
        return float(self._class_flip.mean())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        sample[TRUE_TASK_NAME] = sample[TASK_NAME]
        sample["true_label"] = sample[LABELS]
        sample[SAMPLE_IDX] = idx
        if self._mode == Mode.TRAIN:
            if self._noise_mask[idx]:
                sample[TASK_NAME] = self._noisy_tasks[idx]
            # Class flip can occur whether or not the task is noisy:
            # - compound flip (rho): only when task is noisy
            # - independent flip (rho_indep): only when task is correct
            # _build_noise_map combines both into _class_flip.
            if self._class_flip[idx]:
                sample[LABELS] = 1 - sample[LABELS]
        return sample

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = SyntheticGaussianDataset.collate_fn(examples)
        batch[TRUE_TASK_NAME] = [e.get(TRUE_TASK_NAME, e[TASK_NAME]) for e in examples]
        batch[SAMPLE_IDX] = torch.tensor(
            [e.get(SAMPLE_IDX, -1) for e in examples], dtype=torch.long)
        batch["true_label"] = torch.tensor(
            [e.get("true_label", e[LABELS]) for e in examples], dtype=torch.long)
        return batch


class SyntheticNoisyTestDataset:

    def __init__(self, dataset: SyntheticGaussianDataset,
                 tasks: List[str], epsilon_t: float, seed: int = 42):
        self.dataset = dataset
        self.tasks = tasks
        self.epsilon_t = epsilon_t
        self._seed = seed + 9999
        self._noise_mask: Optional[np.ndarray] = None
        self._noisy_tasks: Optional[List[Optional[str]]] = None

    def set_mode(self, mode: Mode) -> None:
        self.dataset.set_mode(mode)
        self._prebuild_noise()

    def _prebuild_noise(self) -> None:
        N = len(self.dataset)
        rng = np.random.RandomState(self._seed)
        mask = rng.random(N) < self.epsilon_t
        self._noise_mask = mask
        _, _, tasks = self.dataset._splits[self.dataset._mode]
        self._noisy_tasks = []
        for i in range(N):
            if mask[i]:
                others = [t for t in self.tasks if t != tasks[i]]
                self._noisy_tasks.append(rng.choice(others) if others else None)
            else:
                self._noisy_tasks.append(None)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.dataset[idx])
        if self._noise_mask is not None and self._noise_mask[idx]:
            item[TASK_NAME] = self._noisy_tasks[idx]
        return item

    @property
    def collate_fn(self) -> Callable:
        return self.dataset.collate_fn
