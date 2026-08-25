import logging
import os
import pickle
import time
from typing import Any, Dict, List, Union
import zipfile

import csv
import numpy as np
import torch
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
from torch import Tensor, nn

from src.common.consts import GLUE_SUBMISSION_CONFIG, GlueTask

log = logging.getLogger(__name__)


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def get_logits(inputs: Tensor, classifier) -> Tensor:
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def create_results_dataframe(reports: Dict[str, Any]) -> pd.DataFrame:
    # Collect all data for DataFrame.
    rows: List[Dict[str, Any]] = []

    datasets = sorted(list(reports.keys()))

    for dataset_name in datasets:
        thresholds = reports[dataset_name]
        for threshold, metrics in thresholds.items():
            for metric_name, value in metrics.items():
                rows.append({
                    'threshold': float(threshold),
                    'dataset': dataset_name.upper(),
                    'metric': metric_name.upper(),
                    'value': value
                })

    df = pd.DataFrame(rows)

    pivot_df = df.pivot_table(
        index='threshold',
        columns=['dataset', 'metric'],
        values='value',
        fill_value=None
    )

    return pivot_df
