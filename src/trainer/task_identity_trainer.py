from typing import Any, Dict, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import IS_CUDA_AVAILABLE
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.multitask_cls import MultitaskTaskIdentificationModel
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer


class MTDNNTaskIdentityTrainer(MTDNNNoisyTrainer):

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskTaskIdentificationModel(
            self.bert, self.tasks)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None) -> Tuple[Optional[Any], Any]:
        # Strip helper keys (true labels, sample idx) - they're not model inputs
        clean_batch = {k: v for k, v in batch.items()
                       if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}
        # Propagate the trainer's eval-blind state to the model so its forward
        # picks the correct routing for the saved prediction:
        # - K mode (oracle task ID): self._eval_blind is False
        # - B mode (gate-routed):    self._eval_blind is True
        # Default to True at training time (model uses gate routing internally).
        model = self.model_dict[self.task_name]
        model._eval_blind = getattr(self, "_eval_blind", True)
        return model.forward(clean_batch)
