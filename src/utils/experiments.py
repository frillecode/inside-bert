import os
from transformers import (
    TrainingArguments,
)
from datetime import datetime
from typing import Optional


class Experiment:
    def __init__(
        self,
        task: str,
        model: str,
        output_dir: str,
        training_args: dict,
        only_subset: bool = False,
        timestamp: Optional[
            str
        ] = None,  # used to load a specific experiment or create one with a specific timestamp
    ):
        self.experiment_name = f"{model}-{task}"
        self.task = task
        self.model = model
        self.output_dir = output_dir
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.only_subset = only_subset

        # Define training arguments
        self.training_args = TrainingArguments(
            **training_args, output_dir=self.output_dir
        )

        # Ensure that the output directory exists
        self.check_output_dir()

        # Create the experiment directory if it does not exist
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir, exist_ok=False)

        # Define current run directory
        if timestamp is None:
            self.create_run_dir_with_current_timestamp()
        else:
            self.load_run_dir_with_provided_timestamp(timestamp)

    def check_output_dir(self):
        # Ensure that the output directory exists
        if not os.path.exists(self.output_dir):
            # otherwise raise an error
            raise FileNotFoundError(
                f"Output directory {self.output_dir} does not exist. Please define an existing path."
            )

    def create_run_dir_with_current_timestamp(self):
        """
        If timestamp is not provided, create a new run directory with the current timestamp
        """
        self.experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_run_dir = os.path.join(
            self.experiment_dir, str(self.experiment_timestamp)
        )
        print(
            f"[INFO] No timestamp provided. Creating a new run of experiment {self.experiment_name} with timestamp {self.experiment_timestamp}..."
        )
        self.create_run_dir_folders()

    def load_run_dir_with_provided_timestamp(self, timestamp: str):
        """
        If timestamp is provided, load the existing run directory with the provided timestamp. If it does not exist, raise an error.
        """
        self.experiment_timestamp = datetime.strptime(
            timestamp, "%Y-%m-%d_%H-%M-%S"
        ).strftime("%Y-%m-%d_%H-%M-%S")
        self.current_run_dir = os.path.join(
            self.experiment_dir, str(self.experiment_timestamp)
        )

        # either load the existing run directory
        if os.path.exists(self.current_run_dir):
            load_dir = input(
                f"[INFO] Loading previous run of experiment {self.experiment_name} with the provided timestamp {self.experiment_timestamp}. ONLY use this for loading results, not training models (as this will overwrite the old results). Do you want to proceed? (y/n) "
            )
            if load_dir.lower() != "y":
                raise FileNotFoundError(f"[INFO] Stopping the program.")
        # or ask user if they want to create a new one
        else:
            raise FileNotFoundError(
                f"[INFO] Previous run of experiment {self.experiment_name} with the provided timestamp {self.experiment_timestamp} does not exist. Stopping the program."
            )

    def create_run_dir_folders(self):
        # create timestamped subfolder within experiment folder (indicating each run of the experiment)
        os.makedirs(self.current_run_dir, exist_ok=False)

        # create subfolders within the current run dir directory
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
                f"current_run_dir: {self.current_run_dir}\n"
                f"only_subset: {self.only_subset}\n"
                f"training_args: {self.training_args}\n"
            )
