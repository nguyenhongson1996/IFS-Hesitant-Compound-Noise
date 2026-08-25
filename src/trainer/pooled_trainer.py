from typing import Any, Dict, Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import IS_CUDA_AVAILABLE
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.multitask_cls import PooledMultitaskModel
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer


class PooledTrainer(MTDNNNoisyTrainer):

    def __init__(self, *args, loss_type: str = "ce",
                 bootstrap_beta: float = 0.95,
                 gce_q: float = 0.7,
                 discard_ratio: float = 0.3, **kwargs):
        self._loss_type = loss_type
        self._bootstrap_beta = bootstrap_beta
        self._gce_q = gce_q
        self._discard_ratio = discard_ratio
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = PooledMultitaskModel(
            self.bert, self.tasks,
            loss_type=self._loss_type,
            bootstrap_beta=self._bootstrap_beta,
            gce_q=self._gce_q,
            discard_ratio=self._discard_ratio,
        )
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None
                     ) -> Tuple[Optional[Any], Any]:
        clean = {k: v for k, v in batch.items()
                 if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}
        return self.model_dict[self.task_name].forward(clean)
