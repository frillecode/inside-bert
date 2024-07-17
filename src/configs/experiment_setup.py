import os
from transformers import (
    TrainingArguments,
)
from datetime import datetime


class ExperimentInfo:

    experiment_name: str = "blablabla"
    task: str = "MNLI"
    model: str = "distilbert-base-uncased"
    only_subset: bool = False
    val_split: float = 0.1

    output_dir: str = f"/Users/au617011/Documents/Thesis-results/{experiment_name}"

    # define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,  # default is 8
        per_device_eval_batch_size=8,  # default is 8
        num_train_epochs=3,  # default is 3
        logging_dir="logs",
        logging_steps=50,  # default is 500
        evaluation_strategy="steps",  # default is "no"
        save_steps=50,
        output_dir=f"{output_dir}/checkpoints",
        use_cpu=True,
        # no_cuda=True,  ####################################################################
    )

    # initialize experiment
    def __init__(self):
        self.experiment_name = self.experiment_name
        self.task = self.task
        self.model = self.model
        self.training_args = self.training_args
        self.output_dir = self.output_dir
        self.only_subset = self.only_subset
        self.val_split = self.val_split
        self.experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=False)

        # create timestamped subfolder within experiment folder (indicating each run of the experiment)
        self.current_run_dir = os.path.join(self.output_dir, self.experiment_timestamp)
        os.makedirs(self.current_run_dir, exist_ok=False)

        # create subfolders within the timestamped subfolder
        subfolders = ["checkpoints", "logits", "finetuned-model"]
        for subfolder in subfolders:
            os.makedirs(os.path.join(self.current_run_dir, subfolder), exist_ok=False)

        # save the experiment info to a .txt file in the timestamped subfolder
        with open(f"{self.current_run_dir}/experiment-variables.txt", "w") as f:
            f.write(
                f"experiment_name: {self.experiment_name}\n"
                f"experiment_timestamp: {self.experiment_timestamp}\n"
                f"model: {self.model}\n"
                f"task: {self.task}\n"
                f"experiment_folder_path: {self.output_dir}\n"
                f"only_subset: {self.only_subset}\n"
                f"val_split: {self.val_split}\n"
                f"training_args: {self.training_args}\n"
            )
