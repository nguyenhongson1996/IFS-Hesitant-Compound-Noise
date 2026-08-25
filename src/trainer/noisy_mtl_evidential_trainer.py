from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, GlueTask
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME
from src.models.evidential_binary import (
    evidential_predict,
    evidential_route_blind,
)
from src.models.noisy_robust_evidential_binary import (
    NoisyRobustEvidentialBinaryModel,
)
from src.trainer.mtdnn_trainer import MTDNNNoisyTrainer


class NoisyMTLEvidentialTrainer(MTDNNNoisyTrainer):

    def __init__(self, *args, kl_anneal_epochs: int = 10, **kwargs):
        self._kl_anneal_epochs = kl_anneal_epochs
        super().__init__(*args, **kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = NoisyRobustEvidentialBinaryModel(
            self.bert, self.tasks,
            kl_anneal_epochs=self._kl_anneal_epochs,
        )
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def forward_step(self, batch: Dict[str, Any], task=None
                     ) -> Tuple[Optional[Any], Any]:
        # Anneal the KL weight as a function of current epoch
        epoch = getattr(self, "_current_epoch", 0)
        anneal = self._kl_anneal_epochs
        kl_lambda = (min(1.0, epoch / max(1, anneal))
                     if anneal > 0 else 0.0)
        model = self.model_dict[self.task_name]
        model.kl_lambda = kl_lambda

        clean = {k: v for k, v in batch.items()
                 if k not in (SAMPLE_IDX, TRUE_TASK_NAME, "true_label")}

        if self._eval_blind:
            # Blind: compute alpha for all heads, route by max confidence
            input_batch = {k: v for k, v in clean.items()
                           if k not in (TASK_NAME, LABELS)}
            if IS_CUDA_AVAILABLE:
                input_batch = {k: v.cuda() for k, v in input_batch.items()}
            bert_out = model.bert(**input_batch)
            pooled = model.dropout(bert_out.pooler_output)
            alpha = model.head(pooled)  # [B, T, 2]

            hat_t = evidential_route_blind(alpha)  # [B]
            p_all = evidential_predict(alpha)  # [B, T, 2]
            B = alpha.size(0)
            predictions = p_all[torch.arange(B), hat_t]  # [B, 2]

            if not model.training:
                task_names = clean[TASK_NAME]
                for ii in range(B):
                    pt = hat_t[ii].item()
                    pred_label = int(p_all[ii, pt].argmax().item())
                    model.all_task_names.append(task_names[ii])
                    model.all_evaluated_labels.append(
                        clean[LABELS][ii].item())
                    model.all_membership_scores.append({
                        "alpha": alpha[ii].detach().cpu().tolist(),
                        "predicted_task": model.idx_to_task[pt],
                        "predicted_label": pred_label,
                    })
            return None, predictions

        return model.forward(clean)
