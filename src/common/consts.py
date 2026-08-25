import os
import sys
from enum import Enum
from pathlib import Path

import torch
import numpy as np

np.random.seed(2025)
torch.manual_seed(2025)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)

GLUE = "glue"
TASK_NAME = "task_name"


class GlueTask(Enum):
    MRPC = "mrpc"
    RTE = "rte"
    WNLI = "wnli"
    COLA = "cola"
    SST2 = "sst2"
    QNLI = "qnli"
    MNLI_M = "mnli-m"
    MNLI_MM = "mnli-mm"
    STSB = "stsb"
    QQP = "qqp"
    AX = "ax"
    SNLI = "snli"
    SCITAIL = "scitail"
    # Non-GLUE binary sentiment tasks. All single-sentence binary positive/negative; loaded from HuggingFace Hub.
    IMDB = "imdb"
    YELP = "yelp_polarity"
    AMAZON = "amazon_polarity"


HEADER = ("index", "prediction")
LABEL_MAP = {"nli_tasks": {0: "entailment", 1: "not_entailment"},
             "multi_nli_tasks": {0: "entailment", 1: "neutral", 2: "contradiction"}}
NUM_TEST_SAMPLES = {GlueTask.AX: 1104, GlueTask.COLA: 1063, GlueTask.MNLI_MM: 9847, GlueTask.MNLI_M: 9796,
                    GlueTask.MRPC: 1725, GlueTask.QNLI: 5463, GlueTask.QQP: 390965, GlueTask.RTE: 3000,
                    GlueTask.SST2: 1821, GlueTask.STSB: 1379, GlueTask.WNLI: 146,
                    }
GLUE_SUBMISSION_CONFIG = {
    GlueTask.MRPC: {"filename": "MRPC.tsv", "header": HEADER, "label_map": None},
    GlueTask.RTE: {"filename": "RTE.tsv", "header": HEADER, "label_map": LABEL_MAP["nli_tasks"]},
    GlueTask.WNLI: {"filename": "WNLI.tsv", "header": HEADER, "label_map": None},
    GlueTask.COLA: {"filename": "CoLA.tsv", "header": HEADER, "label_map": None},
    GlueTask.SST2: {"filename": "SST-2.tsv", "header": HEADER, "label_map": None},
    GlueTask.QNLI: {"filename": "QNLI.tsv", "header": HEADER, "label_map": LABEL_MAP["nli_tasks"]},
    GlueTask.MNLI_M: {"filename": "MNLI-m.tsv", "header": HEADER, "label_map": LABEL_MAP["multi_nli_tasks"]},
    GlueTask.MNLI_MM: {"filename": "MNLI-mm.tsv", "header": HEADER, "label_map": LABEL_MAP["multi_nli_tasks"]},
    GlueTask.STSB: {"filename": "STS-B.tsv", "header": HEADER, "label_map": None},
    GlueTask.QQP: {"filename": "QQP.tsv", "header": HEADER, "label_map": None},
    GlueTask.AX: {"filename": "AX.tsv", "header": HEADER, "label_map": LABEL_MAP["multi_nli_tasks"]},
}


class Mode(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# Tasks that have the same dataset format.
MRPC_TASKS = {GlueTask.MRPC, GlueTask.RTE, GlueTask.WNLI}
MLNI_TASKS = {GlueTask.MNLI_M, GlueTask.MNLI_MM}  # Not support.
COLA_TASKS = {GlueTask.COLA, GlueTask.SST2}
QNLI_TASKS = {GlueTask.QNLI}
SNLI_TASKS = {GlueTask.SNLI}
SCITAIL_TASKS = {GlueTask.SCITAIL}
QQP_TASKS = {GlueTask.QQP}
STSB_TASKS = {GlueTask.STSB}
# Single-sentence binary sentiment tasks loaded from HuggingFace Hub (non-GLUE). Used for the multi-domain
# Amazon/sentiment study (B19).
SENTIMENT_HF_TASKS = {GlueTask.IMDB, GlueTask.YELP, GlueTask.AMAZON}

# Binary classification tasks only (truly binary, not multi-class converted to binary).
# Excluded: MNLI (3-class), SNLI (3-class), SCITAIL (3-class), STS-B (regression).
BINARY_TASKS = {GlueTask.MRPC, GlueTask.RTE, GlueTask.WNLI, GlueTask.COLA,
                GlueTask.SST2, GlueTask.QNLI, GlueTask.QQP,
                GlueTask.IMDB, GlueTask.YELP, GlueTask.AMAZON}

# Dataset sampling ratios for training set.
TRAIN_SAMPLE_RATIOS = {
    GlueTask.SST2: 0.1,  # 10%
    GlueTask.QNLI: 0.1,  # 10%
    GlueTask.QQP: 0.02,  # 2%
    # Multi-domain sentiment subsampling: keep training-set sizes comparable to GLUE Config A (~3-5k per task) for fair
    # MTL.
    GlueTask.IMDB: 0.2,  # 25k -> 5k
    GlueTask.YELP: 0.01,  # 560k -> 5.6k
    GlueTask.AMAZON: 0.002,  # 3.6M -> 7.2k
}

# Dataset sampling ratios for validation set.
VALIDATION_SAMPLE_RATIOS = {
    GlueTask.QQP: 0.25,  # 25% for QQP validation/dev set.
}

# Dataset sampling ratios for test set.
TEST_SAMPLE_RATIOS = {
    GlueTask.QQP: 0.02,  # 2% for QQP test set.
}


def override_train_sample_ratio(task: GlueTask, ratio: float) -> None:
    if ratio is None or ratio >= 1.0:
        TRAIN_SAMPLE_RATIOS.pop(task, None)
    else:
        TRAIN_SAMPLE_RATIOS[task] = float(ratio)


IS_CUDA_AVAILABLE = torch.cuda.is_available()

# Transformers input fields.
LABEL = "label"
LABELS = "labels"
INPUT_IDS = "input_ids"
TOKEN_TYPE_IDS = "token_type_ids"
ATTENTION_MASK = "attention_mask"
MODEL_NAME_OR_PATH = "model_name_or_path"

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
SRC_DIR = PROJECT_DIR / "src"
CONFIG_DIR = PROJECT_DIR / "config"
CACHE_DIR = PROJECT_DIR / "cache"
TEMPLATE_DIR = PROJECT_DIR / "templates"
WEIGHT_DIR = Path(os.environ.get("IFS_WEIGHT_DIR", str(PROJECT_DIR / "weights")))
PREDICTION_DIR = PROJECT_DIR / "predictions"

BERT_SINGLE_TASK_DIR = WEIGHT_DIR / "bert_single_task"
sys.path.append(str(PROJECT_DIR))
