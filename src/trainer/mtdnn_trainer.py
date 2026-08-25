from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import GlueTask, IS_CUDA_AVAILABLE, LABELS, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.multitask_cls import MultitaskClassificationModel
from src.trainer.noisy_mtl_trainer import NoisyMTLIFSTrainer


class MTDNNNoisyTrainer(NoisyMTLIFSTrainer):

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskClassificationModel(
            self.bert, self.tasks)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None) -> Tuple[Optional[torch.Tensor], Any]:
        clean_batch = {k: v for k, v in batch.items()
                       if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}

        if self._eval_blind:
            # Blind: run all heads, pick the one with highest max-softmax confidence.
            model = self.model_dict[self.task_name]
            input_batch = {k: v for k, v in clean_batch.items() if k not in (TASK_NAME, LABELS)}
            if IS_CUDA_AVAILABLE:
                input_batch = {k: v.cuda() for k, v in input_batch.items()}
            bert_out = model.bert(**input_batch)
            pooled = model.dropout(bert_out.pooler_output)
            B = pooled.shape[0]
            all_logits = torch.stack(
                [model.classifier_dict[t.value](pooled) for t in self.tasks], dim=1)  # [B, T, 2]
            confidence = torch.softmax(all_logits, dim=-1).max(dim=-1).values  # [B, T]
            best_idx = confidence.argmax(dim=1)  # [B]
            predictions = all_logits[torch.arange(B), best_idx]  # [B, 2]

            if not model.training:
                task_names = clean_batch[TASK_NAME]
                for ii in range(B):
                    pred_task = self.tasks[best_idx[ii].item()].value
                    pred_label = torch.argmax(predictions[ii]).item()
                    model.all_task_names.append(task_names[ii])
                    model.all_evaluated_labels.append(clean_batch[LABELS][ii].item())
                    model.all_membership_scores.append({
                        "predicted_task": pred_task,
                        "predicted_label": pred_label,
                    })

            return None, predictions

        return self.model_dict[self.task_name].forward(clean_batch)

    def add_training_log(self, eval_reports: Dict[str, Any]):
        pass
