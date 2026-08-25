import copy
import json
import os
import pickle
from pprint import pformat
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from transformers.models.bert import BertModel
from tqdm import tqdm

from src.common.consts import IS_CUDA_AVAILABLE, LABELS, TASK_NAME, WEIGHT_DIR, GlueTask, Mode
from src.common.logger_utils import logger
from src.data_processors.ensemble_dataset import EnsembleNLIDataset
from src.models.multitask_cls import (BaseMultitaskModel, MultitaskBinaryTaskIdentificationModel,
                                      MultitaskClassificationModel, MultitaskDiffMembershipsFQBModel,
                                      MultitaskFuzzyQueryBasedModel, MultitaskTaskIdentificationModel,
                                      MultitaskTaskMaximumHeadModel)
from src.models.multitask_ifs import MultitaskIFSFuzzyQueryBasedModel
from src.trainer.base import BaseTrainer
from src.utils import create_results_dataframe


class NLISingleTaskTrainer(BaseTrainer):

    def __init__(self, model_name_or_path: Optional[str] = None,
                 task_name: Optional[Union[GlueTask, str]] = None, **kwargs):
        super(NLISingleTaskTrainer, self).__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.task_name = task_name
        self.setup(**kwargs)
        pass

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path,
                                                                             num_labels=2)
        if self.task_name is not None:
            self.model_dict[self.task_name] = copy.deepcopy(self.base_model)
            if IS_CUDA_AVAILABLE:
                self.model_dict[self.task_name].cuda()
            # Free the base model after loading the pre-trained model for fine-tuning.
            self.base_model = None

    def load_dataset(self, task: GlueTask = None, **kwargs) -> EnsembleNLIDataset:
        task = task or self.task_name
        return EnsembleNLIDataset(tasks=[task], tokenizer=self.tokenizer)

    def setup_optimizer(self, lr: float, num_training_steps: int, weight_decay: float, scheduler_type: str,
                        warmup_ratio: float, **kwargs) -> Tuple[Optimizer, LRScheduler]:
        optimizer = AdamW(self.model_dict[self.task_name].parameters(), lr=lr, weight_decay=weight_decay)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        lr_scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps)
        return optimizer, lr_scheduler

    def forward_step(self, batch: Dict[str, Any], task: GlueTask = None) -> Tuple[Optional[torch.Tensor], Any]:
        task = task or self.task_name
        if IS_CUDA_AVAILABLE:
            batch = {key: value.cuda() for key, value in batch.items() if key != TASK_NAME}

        outputs = self.model_dict[task](**batch)
        return outputs.loss, outputs.logits

    def train_epoch(self, train_dataloader: DataLoader, optimizer: Optimizer, lr_scheduler: LRScheduler,
                    max_gradient_clip: float, max_batches: Optional[int] = None, **kwargs) -> Tuple[float, Any]:
        logger.info("Training ...")
        total_loss = 0
        gts: List[int] = []
        probs: List[torch.Tensor] = []
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            if max_batches is not None and i >= max_batches:
                break
            loss, output = self.forward_step(batch)
            loss.backward()
            for model in self.model_dict.values():
                clip_grad_norm_(model.parameters(), max_gradient_clip)
            optimizer.step()
            lr_scheduler.step()
            for model in self.model_dict.values():
                model.zero_grad()
            pbar.set_description(f"Step loss: {loss.item()}")
            total_loss += loss.item()
            gts.extend(batch[LABELS].detach().cpu())
            actual_probs = torch.softmax(output, dim=1)
            probs.append(actual_probs.detach().cpu())
        return total_loss, self.compute_metrics(gts=gts, probs=torch.cat(probs, dim=0), threshold_list=[0.5])

    def evaluate(self, test_dataloader: DataLoader, task: Optional[GlueTask] = None) -> Dict[str, Any]:
        task = task or self.task_name
        logger.info("Evaluating ...")
        pbar = tqdm(test_dataloader)
        gts: List[int] = []
        probs: List[torch.Tensor] = []
        for batch in pbar:
            _, output = self.forward_step(batch)
            gts.extend(batch[LABELS].detach().cpu())
            actual_probs = torch.softmax(output, dim=1)
            probs.append(actual_probs.detach().cpu())
        return self.compute_metrics(gts=gts, probs=torch.cat(probs, dim=0), task=task, threshold_list=[0.5])

    def compute_metrics(self, gts: List[int], probs: torch.Tensor, task: GlueTask = None,
                        threshold_list: Optional[List[float]] = None) -> Dict[str, Any]:
        task = task or self.task_name
        metrics: Dict[str, Dict[str, Any]] = {}
        if not threshold_list:
            threshold_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        for threshold in threshold_list:
            preds = (probs[:, 1] > threshold).long().tolist()
            if task == GlueTask.MRPC:
                metrics[f"{threshold}"] = {"f1": f1_score(y_true=gts, y_pred=preds),
                                           "acc": accuracy_score(y_true=gts, y_pred=preds)}
            elif task == GlueTask.COLA:
                metrics[f"{threshold}"] = {"mcc": matthews_corrcoef(y_true=gts, y_pred=preds)}
            else:
                metrics[f"{threshold}"] = {"acc": accuracy_score(y_true=gts, y_pred=preds)}
        return metrics

    def train(self, num_epochs: int, batch_size: int, weight_decay: float, scheduler_type: str,
              warmup_ratio: float, max_gradient_clip: float, out_dir: str, lr: float = 1e-5, **kwargs):
        dataset = self.load_dataset()
        dataset.set_mode(Mode.TRAIN)
        dataset_length = len(dataset)
        total_training_steps = int(dataset_length * num_epochs / batch_size)
        optimizer, lr_scheduler = self.setup_optimizer(lr, total_training_steps, weight_decay,
                                                       scheduler_type, warmup_ratio)
        max_score = 0
        for epoch in range(num_epochs):
            self.model_dict[self.task_name].train()
            dataset.set_mode(Mode.TRAIN)

            logger.info(f"Training epoch {epoch}")
            train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=dataset.collate_fn)
            epoch_loss, train_score = self.train_epoch(train_dataloader=train_dataloader, optimizer=optimizer,
                                                       lr_scheduler=lr_scheduler, max_gradient_clip=max_gradient_clip)
            logger.info(f"Epoch loss: {epoch_loss}")

            evaluate_score = self.run_evaluation(dataset, self.task_name, batch_size)

            logger.info(f"\nTrain GLUE score:\n{pformat(train_score)}")
            logger.info(f"\nEvaluate GLUE score:\n{pformat(evaluate_score)}")

            if (max_score < sum(evaluate_score["0.5"].values())) and (sum(train_score["0.5"].values())
                                                                      >= sum(evaluate_score["0.5"].values())):
                self.save(output_dir=out_dir)
                max_score = sum(evaluate_score["0.5"].values())

    def run_evaluation(self, dataset: EnsembleNLIDataset, task: Optional[GlueTask], batch_size: int) -> Dict[str, Any]:
        for model in self.model_dict.values():
            model.eval()

        task = task or self.task_name

        if task == GlueTask.MRPC:
            dataset.set_mode(Mode.TEST)
        else:
            dataset.set_mode(Mode.VALIDATION)

        test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=dataset.collate_fn)
        evaluate_score = self.evaluate(test_dataloader, task=task)
        return evaluate_score

    def save_model(self, output_dir: str, task_name: GlueTask):
        torch.save(self.model_dict[task_name].state_dict(), Path(output_dir) / f"{task_name.value}.pt")

    def load_model(self, output_dir: str, task_name: GlueTask):
        if not os.path.exists(Path(output_dir) / f"{task_name.value}.pt"):
            raise ValueError(f"Task {task_name} not found in {output_dir}.")
        self.model_dict[task_name] = copy.deepcopy(self.base_model)
        self.model_dict[task_name].load_state_dict(torch.load(Path(output_dir) / f"{task_name.value}.pt",
                                                              map_location="cpu"))


class NLIBinaryTasksTrainer(NLISingleTaskTrainer):

    def __init__(self, tasks: List[GlueTask], **kwargs):
        super(NLIBinaryTasksTrainer, self).__init__(tasks=tasks, **kwargs)
        self.tasks = tasks
        # Metadata to save.
        self.metadata: List[Dict[str, Any]] = []

    def load_dataset(self, task: GlueTask = None, **kwargs) -> EnsembleNLIDataset:
        if task:
            return EnsembleNLIDataset(tasks=[task], tokenizer=self.tokenizer)
        return EnsembleNLIDataset(tasks=self.tasks, tokenizer=self.tokenizer)

    def add_training_log(self, eval_reports: Dict[str, Any]):
        pass

    def train(self, num_epochs: int, batch_size: int, weight_decay: float, scheduler_type: str,
              warmup_ratio: float, max_gradient_clip: float, out_dir: str, lr: float = 1e-5, **kwargs):
        eval_results: Dict[str, Dict[str, Any]] = {}
        dataset = self.load_dataset()
        dataset.set_mode(Mode.TRAIN)
        dataset_length = len(dataset)
        total_training_steps = int(dataset_length * num_epochs / batch_size)
        optimizer, lr_scheduler = self.setup_optimizer(lr, total_training_steps, weight_decay,
                                                       scheduler_type, warmup_ratio)
        testing_data_dict = {task: self.load_dataset(task) for task in self.tasks}
        max_score = 0
        max_epoch = 0
        for epoch in range(num_epochs):
            for model in self.model_dict.values():
                model.train()
            dataset.set_mode(Mode.TRAIN)

            logger.info(f"Training epoch {epoch}")
            train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=dataset.collate_fn)
            epoch_loss, train_score = self.train_epoch(train_dataloader=train_dataloader, optimizer=optimizer,
                                                       lr_scheduler=lr_scheduler, max_gradient_clip=max_gradient_clip)

            logger.info(f"Epoch loss: {epoch_loss}")
            logger.info(f"\nTrain GLUE score:\n{pformat(train_score)}")

            task_scores = []
            for task_name, test_dataset in testing_data_dict.items():
                evaluate_score = self.run_evaluation(test_dataset, task_name, batch_size)
                eval_results[task_name.value] = evaluate_score
                if task_name != GlueTask.MRPC:
                    task_scores.append(sum(evaluate_score["0.5"].values()) / len(evaluate_score["0.5"].values()))
            aggregated_score = sum(task_scores)
            if isinstance(self.model_dict[self.task_name], BaseMultitaskModel):
                if self.model_dict[self.task_name].save_eval_logs is not None:
                    self.model_dict[self.task_name].save_eval_logs(out_dir, prefix="eval", epoch=epoch)
                    self.model_dict[self.task_name].log_eval_stats(verbose=True)
                    self.model_dict[self.task_name].reset_eval_logs()
            if max_score < aggregated_score:
                self.save(output_dir=out_dir)
                max_score = aggregated_score
                max_epoch = epoch

            logger.info(f"Max score is {max_score} at epoch {max_epoch}")
            eval_report_df = create_results_dataframe(eval_results)
            logger.info(f"\nEvaluate GLUE score:\n{eval_report_df}")
            self.add_training_log(eval_results)

    def save_model(self, output_dir: str, task_name: str):
        torch.save(self.model_dict[task_name].state_dict(), Path(output_dir) / f"{task_name}.pt")

    def load_model(self, output_dir: Path, task_name: Optional[GlueTask] = None):
        if not os.path.exists(Path(output_dir) / f"{self.task_name}.pt"):
            raise ValueError(f"Task {task_name} not found in {output_dir}.")
        self.model_dict[self.task_name].load_state_dict(torch.load(Path(output_dir) / f"{self.task_name}.pt",
                                                                   map_location="cpu"))

    @classmethod
    def load_from_disk(cls, output_dir: str, tasks: List[GlueTask], task_name: str,
                       keep_base_model: bool = True, **kwargs) -> "BaseTrainer":
        # Create a folder to save the trained model if it is not exists.
        output_dir = Path(WEIGHT_DIR) / output_dir
        # Initialize trainer.
        if not os.path.exists(Path(output_dir) / "trainer_config.json"):
            raise ValueError(f"Can not find trainer_config in {output_dir}.")
        with open(Path(output_dir) / "trainer_config.json", "r") as f:
            trainer_cfg = json.load(f)
        trainer = cls(task_name=task_name, tasks=tasks, **trainer_cfg, **kwargs)
        trainer.load_model(output_dir=output_dir, task_name=task_name)
        # Set up device.
        if IS_CUDA_AVAILABLE:
            for task in trainer.model_dict:
                trainer.model_dict[task].cuda()
        logger.info(f"Loaded model from {output_dir}")
        return trainer


class NLIDifferentClsLayerTrainer(NLIBinaryTasksTrainer):

    def __init__(self, tasks: List[GlueTask], load_trained_classifier: bool = False, freeze_cls: bool = False,
                 **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.load_trained_classifier = load_trained_classifier
        self.freeze_cls = freeze_cls
        super(NLIDifferentClsLayerTrainer, self).__init__(tasks, **kwargs)
        print(self.model_dict[self.task_name])

    def setup(self, load_trained_classifier: bool = False, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskClassificationModel(
            self.bert, self.tasks, load_trained_classifier=self.load_trained_classifier)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def setup_optimizer(self, lr: float, num_training_steps: int, weight_decay: float, scheduler_type: str,
                        warmup_ratio: float, **kwargs) -> Tuple[Optimizer, LRScheduler]:
        params_to_optimize = []
        # Add shared BERT encoder parameters (only once).
        params_to_optimize.extend(self.bert.parameters())
        # Add classification head parameters for each task.
        if not self.freeze_cls:
            for classifier in self.model_dict[self.task_name].classifier_dict.values():
                params_to_optimize.extend(classifier.parameters())
        else:
            for classifier in self.model_dict[self.task_name].classifier_dict.values():
                for param in classifier.parameters():
                    param.requires_grad = False
        optimizer = AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        lr_scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps)
        return optimizer, lr_scheduler

    def forward_step(self, batch: Dict[str, Any], task: GlueTask = None) -> Tuple[Optional[torch.Tensor], Any]:
        return self.model_dict[self.task_name].forward(batch)


class NLIMultitaskFuzzyTrainer(NLIBinaryTasksTrainer):
    def __init__(self, tasks: List[GlueTask], do_blindly_decode: bool = True, is_max_opposite=False, **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.do_blindly_decode = do_blindly_decode
        self.is_max_opposite = is_max_opposite
        super(NLIMultitaskFuzzyTrainer, self).__init__(tasks=tasks, **kwargs)
        print(self.model_dict[self.task_name])
        for name, param in self.model_dict[self.task_name].named_parameters():
            if "bert" not in name:  # Skip BERT parameters for clarity.
                print(f"Custom parameter: {name} - {param.shape}")
        logger.info(f"Do blindly decode: {do_blindly_decode}")

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskFuzzyQueryBasedModel(self.bert, self.tasks,
                                                                        do_blindly_decode=self.do_blindly_decode,
                                                                        is_max_opposite=self.is_max_opposite)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def setup_optimizer(self, lr: float, num_training_steps: int, weight_decay: float, scheduler_type: str,
                        warmup_ratio: float, **kwargs) -> Tuple[Optimizer, LRScheduler]:
        model = self.model_dict[self.task_name]

        # Separate BERT parameters from other parameters.
        bert_params = []
        other_params = []

        for name, param in model.named_parameters():
            if "bert" in name.lower():
                bert_params.append(param)
            else:
                other_params.append(param)

        logger.info(f"Number of BERT params: {len(bert_params)}")
        logger.info(f"Number of other params: {len(other_params)}")
        # Create parameter groups with different learning rates.
        param_groups = [
            {"params": bert_params, "lr": lr / 10},  # BERT gets 1/10 of the learning rate.
            {"params": other_params, "lr": lr}  # Other parameters get full learning rate.
        ]
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        lr_scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps)
        return optimizer, lr_scheduler

    def forward_step(self, batch: Dict[str, Any], task: GlueTask = None) -> Tuple[Optional[torch.Tensor], Any]:
        return self.model_dict[self.task_name].forward(batch)

    def add_training_log(self, eval_reports: Dict[str, Any]):

        reports: Dict[str, Any] = {}
        datasets = sorted(list(eval_reports.keys()))

        for dataset_name in datasets:
            if dataset_name == GlueTask.MRPC.value:
                metric = "f1"
            elif dataset_name == GlueTask.COLA.value:
                metric = "mcc"
            else:
                metric = "acc"
            score = eval_reports[dataset_name]["0.5"][metric]
            reports[dataset_name] = score

        reports["fuzzy_centers"] = self.model_dict[self.task_name].fuzzy_centers.tolist()
        reports["fuzzy_std"] = self.model_dict[self.task_name].fuzzy_std.tolist()


class NLITaskIdentificationTrainer(NLIDifferentClsLayerTrainer):

    def __init__(self, tasks: List[GlueTask], load_trained_classifier: bool = False, freeze_cls: bool = False,
                 **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.load_trained_classifier = load_trained_classifier
        self.freeze_cls = freeze_cls
        super(NLITaskIdentificationTrainer, self).__init__(tasks, **kwargs)
        print(self.model_dict[self.task_name])

    def setup(self, load_trained_classifier: bool = False, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskTaskIdentificationModel(
            self.bert, self.tasks, load_trained_classifier=self.load_trained_classifier)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()


class NLIMultitaskDiffMembershipFuzzyTrainer(NLIMultitaskFuzzyTrainer):
    def __init__(self, membership_type: str = "gaussian", **kwargs):
        self.membership_type = membership_type
        self.global_step = 0  # Track global training step.
        super(NLIMultitaskDiffMembershipFuzzyTrainer, self).__init__(**kwargs)

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskDiffMembershipsFQBModel(bert=self.bert, tasks=self.tasks,
                                                                           do_blindly_decode=self.do_blindly_decode,
                                                                           is_max_opposite=self.is_max_opposite,
                                                                           membership_type=self.membership_type)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def train_epoch(self, train_dataloader: DataLoader, optimizer: Optimizer, lr_scheduler: LRScheduler,
                    max_gradient_clip: float, max_batches: Optional[int] = None, **kwargs) -> Tuple[float, Any]:
        logger.info("Training ...")
        model = self.model_dict[self.task_name]

        total_loss = 0
        gts: List[int] = []
        probs: List[torch.Tensor] = []
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            if max_batches is not None and i >= max_batches:
                break
            self.global_step += 1
            loss, output = self.forward_step(batch)
            loss.backward()

            grad_norm_fuzzy = model.current_grad.get("fuzzy_params", 0.0)
            grad_norm_query = model.current_grad.get("query_vectors", 0.0)
            clip_grad_norm_(model.parameters(), max_gradient_clip)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

            model.training_logs["steps"].append(self.global_step)
            model.training_logs["train_loss"].append(loss.item())
            model.training_logs["unknown_loss"].append(model.current_loss_components.get("unknown_loss", 0.0))
            model.training_logs["opposite_loss"].append(model.current_loss_components.get("opposite_loss", 0.0))
            model.training_logs["correct_task_loss"].append(model.current_loss_components.get("correct_task_loss", 0.0))
            model.training_logs["fuzzy_params_grad_norm"].append(grad_norm_fuzzy)
            model.training_logs["query_vectors_grad_norm"].append(grad_norm_query)
            model.training_logs["fuzzy_memberships"].append(model.current_fuzzy_memberships)

            pbar.set_description(f"Step loss: {loss.item()}")
            total_loss += loss.item()
            gts.extend(batch[LABELS].detach().cpu())
            actual_probs = torch.softmax(output, dim=1)
            probs.append(actual_probs.detach().cpu())
        return total_loss, self.compute_metrics(gts=gts, probs=torch.cat(probs, dim=0), threshold_list=[0.5])

    def save_training_logs(self, output_dir: str, filename: str):
        model = self.model_dict[self.task_name]
        logs_path = Path(output_dir) / filename
        if not logs_path.parent.exists():
            os.makedirs(logs_path.parent, exist_ok=True)
        with open(logs_path, "wb") as f:
            pickle.dump(model.training_logs, f)
        logger.info(f"Training logs saved to {logs_path}")

    def train(self, num_epochs: int, batch_size: int, weight_decay: float, scheduler_type: str,
              warmup_ratio: float, max_gradient_clip: float, out_dir: str, log_file: str, lr: float = 1e-5, **kwargs):
        super().train(num_epochs=num_epochs, batch_size=batch_size, weight_decay=weight_decay,
                      scheduler_type=scheduler_type, warmup_ratio=warmup_ratio,
                      max_gradient_clip=max_gradient_clip, out_dir=out_dir, lr=lr)
        # Save training logs after all epochs are done.
        self.save_training_logs(output_dir=out_dir, filename=log_file)
        logger.info(f"Training completed. Logs saved to {out_dir}/training_logs.json")

    def add_training_log(self, eval_reports: Dict[str, Any]):

        reports: Dict[str, Any] = {}
        datasets = sorted(list(eval_reports.keys()))

        for dataset_name in datasets:
            if dataset_name == GlueTask.MRPC.value:
                metric = "f1"
            elif dataset_name == GlueTask.COLA.value:
                metric = "mcc"
            else:
                metric = "acc"
            score = eval_reports[dataset_name]["0.5"][metric]
            reports[dataset_name] = score

        reports["fuzzy_params"] = self.model_dict[self.task_name].fuzzy_params.tolist()


class NLIMaximumTaskTrainer(NLITaskIdentificationTrainer):

    def __init__(self, tasks: List[GlueTask], load_trained_classifier: bool = False, freeze_cls: bool = False,
                 **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.load_trained_classifier = load_trained_classifier
        self.freeze_cls = freeze_cls
        super(NLIMaximumTaskTrainer, self).__init__(tasks, **kwargs)
        print(self.model_dict[self.task_name])

    def setup(self, load_trained_classifier: bool = False, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskTaskMaximumHeadModel(
            self.bert, self.tasks, load_trained_classifier=self.load_trained_classifier)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()


class NLIBinaryTaskIdentificationTrainer(NLIDifferentClsLayerTrainer):

    def __init__(self, tasks: List[GlueTask], load_trained_classifier: bool = False, freeze_cls: bool = False,
                 **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.load_trained_classifier = load_trained_classifier
        self.freeze_cls = freeze_cls
        super(NLIBinaryTaskIdentificationTrainer, self).__init__(tasks, **kwargs)
        print(self.model_dict[self.task_name])

    def setup(self, load_trained_classifier: bool = False, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskBinaryTaskIdentificationModel(
            self.bert, self.tasks, load_trained_classifier=self.load_trained_classifier)
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()


class NLIMultitaskIFSTrainer(NLIBinaryTasksTrainer):

    def __init__(self, tasks: List[GlueTask], do_blindly_decode: bool = True, lambda_other: float = 1.0, **kwargs):
        self.bert: Optional[BertModel] = None
        self.tasks = tasks
        self.do_blindly_decode = do_blindly_decode
        self.lambda_other = lambda_other
        super(NLIMultitaskIFSTrainer, self).__init__(tasks=tasks, **kwargs)
        print(self.model_dict[self.task_name])
        for name, param in self.model_dict[self.task_name].named_parameters():
            if "bert" not in name:  # Skip BERT parameters for clarity.
                print(f"Custom parameter: {name} - {param.shape}")
        logger.info(f"Do blindly decode: {do_blindly_decode}")
        logger.info(f"Lambda other: {lambda_other}")

    def setup(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.base_model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.bert = self.base_model.bert
        self.model_dict[self.task_name] = MultitaskIFSFuzzyQueryBasedModel(
            self.bert, self.tasks,
            do_blindly_decode=self.do_blindly_decode,
            lambda_other=self.lambda_other
        )
        if IS_CUDA_AVAILABLE:
            self.model_dict[self.task_name].cuda()

    def setup_optimizer(self, lr: float, num_training_steps: int, weight_decay: float, scheduler_type: str,
                        warmup_ratio: float, **kwargs) -> Tuple[Optimizer, LRScheduler]:
        model = self.model_dict[self.task_name]

        # Separate BERT parameters from other parameters.
        bert_params = []
        other_params = []

        for name, param in model.named_parameters():
            if "bert" in name.lower():
                bert_params.append(param)
            else:
                other_params.append(param)

        logger.info(f"Number of BERT params: {len(bert_params)}")
        logger.info(f"Number of other params: {len(other_params)}")
        # Create parameter groups with different learning rates.
        param_groups = [
            {"params": bert_params, "lr": lr / 10},  # BERT gets 1/10 of the learning rate.
            {"params": other_params, "lr": lr}  # Other parameters get full learning rate.
        ]
        optimizer = AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        lr_scheduler = get_scheduler(scheduler_type, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_training_steps)
        return optimizer, lr_scheduler

    def forward_step(self, batch: Dict[str, Any], task: GlueTask = None) -> Tuple[Optional[torch.Tensor], Any]:
        return self.model_dict[self.task_name].forward(batch)

    def add_training_log(self, eval_reports: Dict[str, Any]):
        reports: Dict[str, Any] = {}
        datasets = sorted(list(eval_reports.keys()))

        for dataset_name in datasets:
            if dataset_name == GlueTask.MRPC.value:
                metric = "f1"
            elif dataset_name == GlueTask.COLA.value:
                metric = "mcc"
            else:
                metric = "acc"
            score = eval_reports[dataset_name]["0.5"][metric]
            reports[dataset_name] = score

        # Log IFS-specific parameters.
        model = self.model_dict[self.task_name]
        reports["fuzzy_centers"] = model.fuzzy_centers.detach().cpu().tolist()
        reports["fuzzy_std"] = model.fuzzy_std.detach().cpu().tolist()
        reports["query_vectors"] = model.query_vectors.detach().cpu().tolist()
