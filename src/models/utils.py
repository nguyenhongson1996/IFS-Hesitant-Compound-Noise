from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.common.consts import WEIGHT_DIR
from src.common.logger_utils import logger


def load_pretrained_classifier(output_dir: str, classification_dict: nn.ModuleDict):
    output_dir = Path(WEIGHT_DIR) / output_dir
    for task, classifier in classification_dict.items():
        checkpoint_path = Path(output_dir) / f"{task}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        classifier_state_dict = {
            "weight": checkpoint["classifier.weight"],
            "bias": checkpoint["classifier.bias"]
        }
        logger.info(f"Load trained weight for classification layer for task {task}")
        classifier.load_state_dict(classifier_state_dict)
