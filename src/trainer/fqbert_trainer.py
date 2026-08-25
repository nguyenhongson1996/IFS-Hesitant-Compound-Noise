from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.fqbert_noisy import FQBertNoisyModel
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer


class FQBertNoisyTrainer(MTDNNNoisyTrainer):

    def __init__(self, *args, lambda_hes: float = 1.0, **kwargs):
        self._lambda_hes = lambda_hes
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = FQBertNoisyModel(self.bert, self.tasks)
        # Inject lambda_hes onto the model so the loss reads it at forward time.
        self.model_dict[self.task_name].lambda_hes = self._lambda_hes
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None) -> Tuple[Optional[torch.Tensor], Any]:
        # Strip helper keys (noise injection metadata) - not model inputs.
        clean_batch = {k: v for k, v in batch.items()
                       if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}
        model = self.model_dict[self.task_name]
        # Propagate K/B mode for the eval slice currently being processed.
        # - K (oracle): self._eval_blind is False
        # - B (blind):  self._eval_blind is True
        # - Training:   self._eval_blind is None
        model._eval_blind = getattr(self, "_eval_blind", None)
        return model.forward(clean_batch)

    def add_training_log(self, eval_reports: Dict[str, Any]):
        # Match MTDNNNoisyTrainer - no extra logging needed.
        pass
