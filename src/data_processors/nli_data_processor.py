from functools import partial
from typing import Any, Dict, List
import torch

from src.common.consts import (COLA_TASKS, GLUE, MRPC_TASKS, QNLI_TASKS, GlueTask,
                               QQP_TASKS, BINARY_TASKS, SENTIMENT_HF_TASKS,
                               TRAIN_SAMPLE_RATIOS, VALIDATION_SAMPLE_RATIOS, TEST_SAMPLE_RATIOS)
from datasets import Dataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding


def tokenize(examples: Dict[str, Any], text_fields: List[str],
             tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    kwargs = {"padding": False, "return_tensors": "pt"}
    args = [examples[text_field] for text_field in text_fields]
    return tokenizer(*args, **kwargs)


def pad_tensor_list(tensor_list: List[Any], pad_value: int = 0, max_length: int = 512) -> torch.Tensor:
    # Convert all elements to tensors if they are not already.
    tensor_list = [torch.tensor(ts) if not isinstance(ts, torch.Tensor) else ts for ts in tensor_list]
    # If the list is empty, raise an error.
    if not tensor_list:
        raise ValueError("Input tensor list is empty.")

    tensor_list = [torch.cat([tensor[:max_length - 1], tensor[-1:]]) if tensor.size(0) > max_length else tensor
                   for tensor in tensor_list]
    max_length = max(tensor.size(0) for tensor in tensor_list)
    padded_tensor = torch.full((len(tensor_list), max_length), pad_value, dtype=tensor_list[0].dtype)
    for idx, tensor in enumerate(tensor_list):
        padded_tensor[idx, :tensor.size(0)] = tensor

    return padded_tensor


def load_glue_dataset(task_name: GlueTask, tokenizer: PreTrainedTokenizer) -> Dataset:
    # Only process binary classification tasks
    if task_name not in BINARY_TASKS:
        raise ValueError(f"Task {task_name} is not a binary classification task.")

    # Determine text fields based on task
    if task_name in MRPC_TASKS:
        text_fields = ["sentence1", "sentence2"]
        dataset = load_dataset(GLUE, task_name.value)
    elif task_name in COLA_TASKS:
        text_fields = ["sentence"]
        dataset = load_dataset(GLUE, task_name.value)
    elif task_name in QNLI_TASKS:
        text_fields = ["question", "sentence"]
        dataset = load_dataset(GLUE, task_name.value)
    elif task_name in QQP_TASKS:
        text_fields = ["question1", "question2"]
        dataset = load_dataset(GLUE, task_name.value)
    elif task_name in SENTIMENT_HF_TASKS:
        # Single-sentence binary sentiment, loaded from HuggingFace Hub. Field names differ per dataset:
        if task_name == GlueTask.IMDB:
            text_fields = ["text"]
            dataset = load_dataset("imdb")
        elif task_name == GlueTask.YELP:
            text_fields = ["text"]
            dataset = load_dataset("yelp_polarity")
        elif task_name == GlueTask.AMAZON:
            # amazon_polarity has fields "title", "content"; use content.
            text_fields = ["content"]
            dataset = load_dataset("amazon_polarity")
        # These HF datasets have no GLUE-style "validation" split. Build one by carving the trailing 10% off train,
        # leaving 90% for train, the original test as the held-out test split.
        if "validation" not in dataset:
            train_n = len(dataset["train"])
            split_at = int(0.9 * train_n)
            full_train = dataset["train"]
            dataset["train"] = full_train.select(range(split_at))
            dataset["validation"] = full_train.select(range(split_at, train_n))
    else:
        raise ValueError(f"Task {task_name} is not supported or is not a binary classification task.")

    # Apply sampling to training set.
    if task_name in TRAIN_SAMPLE_RATIOS:
        ratio = TRAIN_SAMPLE_RATIOS[task_name]
        if "train" in dataset:
            train_size = len(dataset["train"])
            sample_size = int(train_size * ratio)
            dataset["train"] = dataset["train"].select(range(sample_size))

    # Apply sampling to validation set (only for QQP).
    if task_name in VALIDATION_SAMPLE_RATIOS:
        ratio = VALIDATION_SAMPLE_RATIOS[task_name]
        if "validation" in dataset:
            split_size = len(dataset["validation"])
            sample_size = int(split_size * ratio)
            dataset["validation"] = dataset["validation"].select(range(sample_size))

    # Apply sampling to test set (only for QQP).
    if task_name in TEST_SAMPLE_RATIOS:
        ratio = TEST_SAMPLE_RATIOS[task_name]
        if "test" in dataset:
            split_size = len(dataset["test"])
            sample_size = int(split_size * ratio)
            dataset["test"] = dataset["test"].select(range(sample_size))

    # Tokenize dataset.
    tokenize_fn = partial(tokenize, text_fields=text_fields, tokenizer=tokenizer)
    dataset = dataset.map(tokenize_fn, batched=False, remove_columns=text_fields)

    return dataset
