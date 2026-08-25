import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.common.consts import IS_CUDA_AVAILABLE, MODEL_NAME_OR_PATH, WEIGHT_DIR, GlueTask
from src.common.logger_utils import logger


class BaseTrainer(ABC):
    def __init__(self, model_name_or_path: Optional[str] = None, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.task_name: Optional[str] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.base_model: Optional[PreTrainedModel] = None
        self.model_dict: Dict[GlueTask, PreTrainedModel] = {}
        self.training_log: List[Dict[str, Any]] = []

    @abstractmethod
    def setup(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self, **kwargs) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def setup_optimizer(self, **kwargs) -> Tuple[Optimizer, LRScheduler]:
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Any]:
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self, train_dataloader: DataLoader, optimizer: Optimizer, lr_scheduler: LRScheduler,
                    max_gradient_clip: float, **kwargs) -> Tuple[float, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, test_dataloader: DataLoader) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, num_epochs: int, batch_size: int, lr: float = 0.001, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, output_dir: str, task_name: GlueTask):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, output_dir: str, task_name: GlueTask):
        raise NotImplementedError

    def save(self, output_dir: str):
        # Create a folder to save the trained model if it is not exists.
        output_dir = Path(WEIGHT_DIR) / output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Save trainer configuration.
        _trainer_cfg = {
            MODEL_NAME_OR_PATH: self.model_name_or_path
        }
        if os.path.exists(Path(output_dir) / "trainer_config.json"):
            with open(Path(output_dir) / "trainer_config.json", "r") as fh:
                trainer_cfg = json.load(fh)
            if _trainer_cfg != trainer_cfg:
                raise ValueError("Trainer config is conflicted.")
        else:
            with open(Path(output_dir) / "trainer_config.json", "w") as fh:
                json.dump(_trainer_cfg, fh)
        # Save model.
        for task_name in self.model_dict.keys():
            self.save_model(output_dir=output_dir, task_name=task_name)
        with open(Path(output_dir) / "train_log.log", "w") as fh:
            json.dump(self.training_log, fh, ensure_ascii=False)
        logger.info(f"Saved model to {output_dir}.")

    @classmethod
    def load_from_disk(cls, output_dir: str, tasks: List[GlueTask],
                       keep_base_model: bool = True, **kwargs) -> "BaseTrainer":
        # Create a folder to save the trained model if it is not exists.
        output_dir = Path(WEIGHT_DIR) / output_dir
        # Initialize trainer.
        if not os.path.exists(Path(output_dir) / "trainer_config.json"):
            raise ValueError(f"Can not find trainer_config in {output_dir}.")
        with open(Path(output_dir) / "trainer_config.json", "r") as f:
            trainer_cfg = json.load(f)
        trainer = cls(**trainer_cfg, tasks=tasks, **kwargs)
        # Load model weights for each task.
        for task in tasks:
            trainer.load_model(output_dir=output_dir, task_name=task)
            logger.info(f"Loaded model for task {task.value}")
            trainer.task_name = task
        # Set up device.
        if IS_CUDA_AVAILABLE:
            trainer.base_model.cuda()
            for task in tasks:
                trainer.model_dict[task].cuda()
        # Optionally discard base_model to save memory.
        if not keep_base_model:
            trainer.base_model = None
        logger.info(f"Loaded from {output_dir}")
        return trainer
