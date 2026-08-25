import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common.consts import LABELS, Mode, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import TRUE_TASK_NAME, SAMPLE_IDX

BLITZER_TASKS: List[str] = ["books", "dvd", "electronics", "kitchen"]
N_TASKS: int = len(BLITZER_TASKS)

# Folder name -> canonical task name; "kitchen_&_housewares" is the original Blitzer folder name.
_DOMAIN_TO_TASK = {
    "books": "books",
    "dvd": "dvd",
    "electronics": "electronics",
    "kitchen_&_housewares": "kitchen",
}

_DEFAULT_RAW_DIR = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "blitzer" / "sorted_data_acl"
)

_REVIEW_BLOCK_RE = re.compile(r"<review>(.*?)</review>", re.DOTALL)
_REVIEW_TEXT_RE = re.compile(r"<review_text>(.*?)</review_text>", re.DOTALL)


def parse_review_file(path: Path) -> List[str]:
    text = path.read_text(errors="ignore")
    out: List[str] = []
    for block in _REVIEW_BLOCK_RE.findall(text):
        m = _REVIEW_TEXT_RE.search(block)
        if m:
            body = m.group(1).strip()
            if body:
                out.append(body)
    return out


def load_blitzer_corpus(raw_dir: Path = _DEFAULT_RAW_DIR
                        ) -> Tuple[List[str], List[int], List[str]]:
    texts: List[str] = []
    labels: List[int] = []
    tasks: List[str] = []
    for folder, task in _DOMAIN_TO_TASK.items():
        for polarity, lbl in [("positive", 1), ("negative", 0)]:
            path = raw_dir / folder / f"{polarity}.review"
            if not path.exists():
                raise FileNotFoundError(f"Missing Blitzer file: {path}. Download and extract the Multi-Domain "
                                        f"Sentiment Dataset from the following URL: "
                                        f"https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html, "
                                        f"then organize the extracted files into the expected directory structure "
                                        f"under {raw_dir.parent}.")
            for body in parse_review_file(path):
                texts.append(body)
                labels.append(lbl)
                tasks.append(task)
    return texts, labels, tasks


def stratified_split_indices(
        tasks: List[str], labels: List[int], test_frac: float = 0.2,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for task in BLITZER_TASKS:
        for lbl in (0, 1):
            cell = [i for i, (t, y) in enumerate(zip(tasks, labels))
                    if t == task and y == lbl]
            cell = np.array(cell)
            rng.shuffle(cell)
            n_test = int(round(test_frac * len(cell)))
            test_idx.extend(cell[:n_test].tolist())
            train_idx.extend(cell[n_test:].tolist())
    return (np.array(sorted(train_idx)), np.array(sorted(test_idx)))


def build_tfidf_features(
        texts_train: List[str], texts_test: List[str],
        max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,  # 1 + log(tf), better for review-length variation.
    )
    X_train = vec.fit_transform(texts_train).astype(np.float32).toarray()
    X_test = vec.transform(texts_test).astype(np.float32).toarray()
    return X_train, X_test, vec


class BlitzerDataset(Dataset):

    def __init__(
            self,
            raw_dir: Path = _DEFAULT_RAW_DIR,
            max_features: int = 10000,
            ngram_range: Tuple[int, int] = (1, 2),
            min_df: int = 5,
            test_frac: float = 0.2,
            split_seed: int = 42,
    ) -> None:
        super().__init__()
        texts, labels, tasks = load_blitzer_corpus(raw_dir)
        train_idx, test_idx = stratified_split_indices(
            tasks, labels, test_frac=test_frac, seed=split_seed,
        )
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        X_train, X_test, vec = build_tfidf_features(
            texts_train, texts_test,
            max_features=max_features,
            ngram_range=ngram_range, min_df=min_df,
        )
        self._vectorizer = vec
        self._features = {
            Mode.TRAIN: torch.from_numpy(X_train),
            Mode.VALIDATION: torch.from_numpy(X_test),
        }
        self._labels = {
            Mode.TRAIN: np.array([labels[i] for i in train_idx], dtype=np.int64),
            Mode.VALIDATION: np.array([labels[i] for i in test_idx], dtype=np.int64),
        }
        self._tasks = {
            Mode.TRAIN: [tasks[i] for i in train_idx],
            Mode.VALIDATION: [tasks[i] for i in test_idx],
        }
        self.task_names = BLITZER_TASKS
        self.input_dim = X_train.shape[1]
        self._mode: Mode = Mode.TRAIN

    def set_mode(self, mode: Mode) -> None:
        self._mode = mode

    def __len__(self) -> int:
        return len(self._labels[self._mode])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "input_features": self._features[self._mode][idx],
            LABELS: torch.tensor(int(self._labels[self._mode][idx]), dtype=torch.long),
            TASK_NAME: self._tasks[self._mode][idx],
            TRUE_TASK_NAME: self._tasks[self._mode][idx],
            "true_label": int(self._labels[self._mode][idx]),
            SAMPLE_IDX: idx,
        }

    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "input_features": torch.stack([ex["input_features"] for ex in examples]),
            LABELS: torch.stack([ex[LABELS] for ex in examples]),
            TASK_NAME: [ex[TASK_NAME] for ex in examples],
            TRUE_TASK_NAME: [ex[TRUE_TASK_NAME] for ex in examples],
            "true_label": torch.tensor([ex["true_label"] for ex in examples], dtype=torch.long),
            SAMPLE_IDX: torch.tensor([ex[SAMPLE_IDX] for ex in examples], dtype=torch.long),
        }


class NoisyBlitzerDataset(BlitzerDataset):

    def __init__(
            self,
            epsilon_t: float = 0.2,
            class_noise_rho: float = 0.0,
            class_noise_rho_indep: float = 0.0,
            seed: int = 42,
            **kwargs: Any,
    ) -> None:
        super().__init__(split_seed=seed, **kwargs)
        self.epsilon_t = epsilon_t
        self.class_noise_rho = class_noise_rho
        self.class_noise_rho_indep = class_noise_rho_indep
        self.noise_seed = seed
        self._inject_train_noise()

    def _inject_train_noise(self) -> None:
        rng_task = np.random.default_rng(self.noise_seed + 1)
        rng_compound = np.random.default_rng(self.noise_seed + 2)
        rng_indep = np.random.default_rng(self.noise_seed + 3)

        clean_tasks = list(self._tasks[Mode.TRAIN])
        clean_labels = self._labels[Mode.TRAIN].copy()

        train_n = len(clean_tasks)
        flip_task = rng_task.random(train_n) < self.epsilon_t
        compound_flip = rng_compound.random(train_n) < self.class_noise_rho
        indep_flip = rng_indep.random(train_n) < self.class_noise_rho_indep

        noisy_tasks: List[str] = list(clean_tasks)
        noisy_labels = clean_labels.copy()

        for i in range(train_n):
            true_task = clean_tasks[i]
            if flip_task[i]:
                others = [t for t in self.task_names if t != true_task]
                # Uniform over alternatives, deterministic per-i.
                pick = rng_task.integers(0, len(others))
                noisy_tasks[i] = others[pick]
                if compound_flip[i]:
                    noisy_labels[i] = 1 - noisy_labels[i]
            else:
                if indep_flip[i]:
                    noisy_labels[i] = 1 - noisy_labels[i]

        self._tasks[Mode.TRAIN] = noisy_tasks
        self._labels[Mode.TRAIN] = noisy_labels
        # Keep the clean copy for diagnostics.
        self._clean_train_labels = clean_labels
        self._clean_train_tasks = clean_tasks

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._mode == Mode.TRAIN:
            return {
                "input_features": self._features[Mode.TRAIN][idx],
                LABELS: torch.tensor(int(self._labels[Mode.TRAIN][idx]), dtype=torch.long),
                TASK_NAME: self._tasks[Mode.TRAIN][idx],
                TRUE_TASK_NAME: self._clean_train_tasks[idx],
                "true_label": int(self._clean_train_labels[idx]),
                SAMPLE_IDX: idx,
            }
        # validation/test - clean, identical to base class.
        return super().__getitem__(idx)
