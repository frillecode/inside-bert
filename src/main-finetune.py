import os

import torch
from utils.experiments import Experiment
from utils.utils_finetune import (
    load_and_preprocess_data,
    CustomTrainer,
    compute_metrics,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainerCallback,
)
import yaml


# Define function to load and preprocess data and model and train the model
def main(experiment: Experiment, only_subset: bool = False):

    print("[INFO] Setting up experiment...")

    # Load and preprocess data
    print("[INFO] Loading and preprocessing data...")
    train_dataset, eval_dataset, n_labels = load_and_preprocess_data(
        experiment_info=experiment, only_subset=only_subset
    )

    # Load pretrained bert model
    print("[INFO] Loading pretrained model...")
    device = torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        experiment.model,
        num_labels=n_labels,
    ).to(device)

    # Define the trainer
    print("[INFO] Defining the trainer...")
    trainer = CustomTrainer(
        model=model,
        args=experiment.training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        output_dir=experiment.current_run_dir,
    )

    # Train the model
    print("[INFO] Training the model...")
    trainer.train()

    # Save the model
    print("[INFO] Saving the model...")
    trainer.save_model(f"{experiment.current_run_dir}/finetuned-model")

    print("[INFO] Done!")

    return trainer.model


if __name__ == "__main__":
    # Load experiment info from config file
    with open(
        os.path.join(
            "src",
            "finetune_configs",
            "distilbert-base-uncased-MNLI_finetune_config.yaml",
        ),
        "r",
    ) as file:
        experiment_config = yaml.safe_load(file)

    # Initialize the experiment
    experiment = Experiment(**experiment_config)

    # Run the main function
    trained_model = main(experiment, only_subset=experiment.only_subset)

    # Save the model (unsure if this is necessary)
    torch.save(
        trained_model,
        f"{experiment.current_run_dir}/model.pth",
    )
