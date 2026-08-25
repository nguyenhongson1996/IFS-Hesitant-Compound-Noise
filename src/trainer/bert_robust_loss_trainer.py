from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.multitask_cls import MultitaskRobustLossModel
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer


class BertRobustLossTrainer(MTDNNNoisyTrainer):

    def __init__(self, *args, loss_type: str = "ce",
                 gce_q: float = 0.7,
                 bootstrap_beta: float = 0.95, **kwargs):
        self._loss_type = loss_type
        self._gce_q = gce_q
        self._bootstrap_beta = bootstrap_beta
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskRobustLossModel(
            self.bert, self.tasks,
            loss_type=self._loss_type,
            gce_q=self._gce_q,
            bootstrap_beta=self._bootstrap_beta,
        )
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None
                     ) -> Tuple[Optional[Any], Any]:
        clean = {k: v for k, v in batch.items()
                 if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}

        if self._eval_blind:
            # Blind: run all heads, pick the one with highest max-softmax
            # confidence - same routing as MTDNNNoisyTrainer.forward_step.
            model = self.model_dict[self.task_name]
            input_batch = {k: v for k, v in clean.items()
                           if k not in (TASK_NAME, LABELS)}
            if IS_CUDA_AVAILABLE:
                input_batch = {k: v.cuda() for k, v in input_batch.items()}
            bert_out = model.bert(**input_batch)
            pooled = model.dropout(bert_out.pooler_output)
            B = pooled.shape[0]
            all_logits = torch.stack(
                [model.classifier_dict[t.value](pooled)
                 for t in self.tasks], dim=1)  # [B, T, 2]
            confidence = torch.softmax(all_logits, dim=-1).max(
                dim=-1).values  # [B, T]
            best_idx = confidence.argmax(dim=1)  # [B]
            predictions = all_logits[torch.arange(B), best_idx]  # [B, 2]

            if not model.training:
                task_names = clean[TASK_NAME]
                for ii in range(B):
                    pred_task = self.tasks[best_idx[ii].item()].value
                    pred_label = torch.argmax(predictions[ii]).item()
                    model.all_task_names.append(task_names[ii])
                    model.all_evaluated_labels.append(
                        clean[LABELS][ii].item())
                    model.all_membership_scores.append({
                        "predicted_task": pred_task,
                        "predicted_label": pred_label,
                    })
            return None, predictions

        return self.model_dict[self.task_name].forward(clean)
