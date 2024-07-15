import os

import torch
from configs.experiment_setup import ExperimentInfo
from utils.utils_finetuning import (
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


# Define function to load and preprocess data and model and train the model
def main(experiment: ExperimentInfo, subset: bool = False):

    print("[INFO] Setting up experiment...")

    # Load and preprocess data
    print("[INFO] Loading and preprocessing data...")
    train_dataset, eval_dataset, n_labels = load_and_preprocess_data(
        experiment_info=experiment, subset=subset
    )

    # Load pretrained bert model
    print("[INFO] Loading pretrained model...")
    device = torch.device("mps")
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
        output_dir=experiment.output_dir,
    )

    # Train the model
    print("[INFO] Training the model...")
    trainer.train()

    # Save the model
    print("[INFO] Saving the model...")
    trainer.save_model(f"{experiment.output_dir}/finetuned-model")

    print("[INFO] Done!")

    return trainer.model


if __name__ == "__main__":
    # Define experiment
    experiment = ExperimentInfo()

    # Run the main function
    trained_model = main(experiment, subset=True)  # subset=True

    # Save the model
    torch.save(trained_model, f"{experiment.output_dir}/model.pth")