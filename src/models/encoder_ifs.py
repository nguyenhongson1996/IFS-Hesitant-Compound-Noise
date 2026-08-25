from typing import Any, Dict, List

import torch
import torch.nn as nn

TRUST_SIGNALS = ("ifs", "entropy", "margin", "loss")

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME
from src.data_processors.noisy_ensemble_dataset import SAMPLE_IDX, TRUE_TASK_NAME


class MLPEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = batch["features"]
        if IS_CUDA_AVAILABLE:
            x = x.cuda()
        return self.net(x)
