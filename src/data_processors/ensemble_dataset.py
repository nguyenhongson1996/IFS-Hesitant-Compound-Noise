from bisect import bisect
from typing import Any, Dict, List, Tuple

from immutabledict import immutabledict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from src.common.consts import (ATTENTION_MASK, INPUT_IDS, LABEL, LABELS, TOKEN_TYPE_IDS, TASK_NAME, GlueTask,
                               Mode)
from src.data_processors.nli_data_processor import load_glue_dataset, pad_tensor_list


class EnsembleNLIDataset(Dataset):

    def __init__(self, tasks: List[GlueTask], tokenizer: PreTrainedTokenizer):
        # Mapping between the task name and the dataset corresponding to the task.
        self._tasks: Tuple[GlueTask] = tuple(tasks)
        self._dataset_dict = {task: load_glue_dataset(task, tokenizer) for task in self._tasks}
        self._tokenizer = tokenizer
        # Set up the dataset.
        self._setup()
        self._mode: Mode = Mode.TRAIN

    def _setup(self):
        # Offset of the task data in the ensemble data.
        tasks_data_offset: Dict[Mode, List[int]] = {Mode.TRAIN: [], Mode.VALIDATION: [], Mode.TEST: []}
        train_offset = 0
        validation_offset = 0
        test_offset = 0
        for task in self._tasks:
            task_train_data_length = len(self._dataset_dict[task][Mode.TRAIN.value])
            task_validation_data_length = len(self._dataset_dict[task][Mode.VALIDATION.value])
            task_test_data_length = len(self._dataset_dict[task][Mode.TEST.value])
            tasks_data_offset[Mode.TRAIN].append(train_offset)
            tasks_data_offset[Mode.VALIDATION].append(validation_offset)
            tasks_data_offset[Mode.TEST].append(test_offset)
            train_offset += task_train_data_length
            validation_offset += task_validation_data_length
            test_offset += task_test_data_length

        self._tasks_data_offset = immutabledict(tasks_data_offset)
        self._length = immutabledict({Mode.TRAIN: train_offset, Mode.VALIDATION: validation_offset,
                                      Mode.TEST: test_offset})

    def set_mode(self, mode: Mode):
        self._mode = mode

    def __len__(self):
        return self._length[self._mode]

    def __getitem__(self, index) -> Dict[str, Any]:
        task_idx = bisect(self._tasks_data_offset[self._mode], index) - 1
        offset = self._tasks_data_offset[self._mode][task_idx]
        task_name: GlueTask = self._tasks[task_idx]
        task_data_idx = index - offset
        sample = self._dataset_dict[task_name][self._mode.value][task_data_idx]
        sample[TASK_NAME] = task_name.value
        return sample

    def collate_fn(self, examples: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
        # Pad the input features to the longest sequence in the batch.
        padded = {field: pad_tensor_list([example[field][0] for example in examples], self._tokenizer.pad_token_id)
                  for field in [INPUT_IDS, ATTENTION_MASK, TOKEN_TYPE_IDS]}
        labels = torch.LongTensor([example[LABEL] for example in examples])
        task_names = [example[TASK_NAME] for example in examples]
        collected_input = {**padded, LABELS: labels, TASK_NAME: task_names}
        return collected_input

    def get_dataloader(self, batch_size=8, shuffle=True, **kwargs):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn, **kwargs)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from src.common.logger_utils import logger

    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = EnsembleNLIDataset(tasks=[GlueTask.COLA, GlueTask.MRPC], tokenizer=bert_tokenizer)
    dataset.set_mode(Mode.TRAIN)
    logger.info(f"Training dataset length: {len(dataset)}")
    logger.info(f"10th training sample {dataset[10]}")
    logger.info(f"10000th training sample {dataset[10000]}")

    dataloader = dataset.get_dataloader(batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    logger.info(f"Batch labels: {batch[LABELS]}")
    logger.info(f"Batch task names: {batch[TASK_NAME]}")
    logger.info(f"Input_ids lengths in batch: {[len(ids) for ids in batch[INPUT_IDS]]}")
    logger.info(f"Attention_mask lengths in batch: {[len(mask) for mask in batch[ATTENTION_MASK]]}")
    logger.info(f"Token_type_ids lengths in batch: {[len(ids) for ids in batch[TOKEN_TYPE_IDS]]}")

    logger.info(f"Batch: {batch}")

    dataset.set_mode(Mode.VALIDATION)
    logger.info(f"Validation dataset length: {len(dataset)}")
    logger.info(f"1st validation sample's label: {dataset[0][LABEL]}")

    dataset.set_mode(Mode.TEST)
    logger.info(f"Testing dataset length: {len(dataset)}")
    logger.info(f"1st testing sample's label: {dataset[0][LABEL]}")
