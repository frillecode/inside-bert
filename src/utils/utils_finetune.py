import numpy as np
from typing import Callable
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Trainer
from configs.experiment_setup import ExperimentInfo
import evaluate
from torch.utils.data import random_split, dataset
import torch


# Load and transform logits
def load_and_transform_logits(
    experiment_info: ExperimentInfo, transform_method: Callable
):
    """Load logits and transform them using the provided method."""

    pass


# Convert a subset to a Dataset object
def subset_to_dataset(subset: dataset.Subset):
    data = [subset[i] for i in range(len(subset))]

    # Convert list of dicts to dict of lists
    data = {k: [dic[k] for dic in data] for k in data[0]}

    # Create a new Dataset object
    dataset = Dataset.from_dict(data)

    return dataset


# Load and preprocess SST-2 dataset
def load_and_preprocess_sst2(
    experiment_info: ExperimentInfo, only_subset: bool = False, val_split: float = 0.1
):
    # load the SST-2 dataset
    dataset = load_dataset("glue", "sst2")

    # load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_info.model, use_fast=True)

    # tokenize the dataset and return tensors
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
    )

    # select a small subset of the dataset if only_subset is True
    if only_subset:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    else:
        train_dataset = tokenized_datasets["train"]

    # Split the train dataset into train and validation
    train_size = int((1.0 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataset = subset_to_dataset(train_dataset)
    eval_dataset = subset_to_dataset(eval_dataset)

    # check number of labels
    n_labels = len(set(train_dataset["label"]))

    return train_dataset, eval_dataset, n_labels


# Load and preprocess MRPC dataset
def load_and_preprocess_mrpc(
    experiment_info: ExperimentInfo, only_subset: bool = False
):
    # load the MRPC dataset
    dataset = load_dataset("glue", "mrpc")

    # load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_info.model, use_fast=True)

    # tokenize the dataset and return tensors
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
    )

    # select a small subset of the dataset if only_subset is True
    if only_subset:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        eval_dataset = (
            tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))
        )
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

    # check number of labels
    n_labels = len(set(train_dataset["label"]))

    return train_dataset, eval_dataset, n_labels


# Load and preprocess MNLI dataset
def load_and_preprocess_mnli(
    experiment_info: ExperimentInfo, only_subset: bool = False, val_split: float = 0.1
):
    # load the MNLI dataset
    dataset = load_dataset("glue", "mnli")

    # load a pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(experiment_info.model, use_fast=True)

    # tokenize the dataset and return tensors
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ),
        batched=True,
    )

    # select a small subset of the dataset if only_subset is True
    if only_subset:
        train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    else:
        train_dataset = tokenized_datasets["train"]

    # Split the train dataset into train and validation
    train_size = int((1.0 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, eval_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataset = subset_to_dataset(train_dataset)
    eval_dataset = subset_to_dataset(eval_dataset)

    # check number of labels
    n_labels = len(set(train_dataset["label"]))

    return train_dataset, eval_dataset, n_labels


# Load and preprocess data - general function
def load_and_preprocess_data(
    experiment_info: ExperimentInfo, only_subset: bool = False
):
    if experiment_info.task == "SST-2":
        return load_and_preprocess_sst2(experiment_info, only_subset)
    elif experiment_info.task == "MRPC":
        return load_and_preprocess_mrpc(experiment_info, only_subset)
    elif experiment_info.task == "MNLI":
        return load_and_preprocess_mnli(experiment_info, only_subset)
    else:
        raise ValueError("Task not supported.")


# Custom Trainer class
class CustomTrainer(Trainer):
    """
    Custom Trainer class to save the logits at each training step.
    """

    def __init__(
        self, model, args, train_dataset, eval_dataset, compute_metrics, output_dir
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        self.output_dir = output_dir

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        else:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = outputs.logits
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Save the logits
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        np.save(
            f"{self.output_dir}/logits/logits_step_{self.state.global_step}.npy",
            logits.detach().cpu().numpy(),
        )

        return loss.detach()


# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)
