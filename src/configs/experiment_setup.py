import os
from transformers import (
    TrainingArguments,
)


class ExperimentInfo:

    experiment_name: str = "distilbert-base-uncased-mnli"
    task: str = "MNLI"
    model: str = "distilbert-base-uncased"

    output_dir: str = f"/Users/au617011/Documents/Thesis-results/{experiment_name}"

    # define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=12,
        save_steps=10,
        output_dir=f"{output_dir}/checkpoints",
        no_cuda=True,
    )

    # initialize experiment
    def __init__(self):
        self.experiment_name = self.experiment_name
        self.task = self.task
        self.model = self.model
        self.training_args = self.training_args
        self.output_dir = self.output_dir

        # check if necessary directores exists, otherwise raise an error
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(
                f"Project directory {self.output_dir} does not exist."
            )
        else:
            # create experiment folder
            folders = ["", "checkpoints", "logits", "finetuned-model"]

            for folder in folders:
                os.makedirs(os.path.join(self.output_dir, folder), exist_ok=True)

            # save the experiment info to a .txt file in the experiment folder
            with open(f"{self.output_dir}/experiment-variables.txt", "w") as f:
                f.write(
                    f"experiment_name: {self.experiment_name}\n"
                    f"model: {self.model}\n"
                    f"task: {self.task}\n"
                    f"experiment_folder_path: {self.output_dir}\n"
                    f"training_args: {self.training_args}\n"
                )