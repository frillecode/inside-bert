import os
import yaml
from utils.experiments import Experiment
import matplotlib.pyplot as plt
import json

# Load specific run of the experiment
title = "bert-base-multilingual-uncased-MRPC"

with open(
    os.path.join(
        "configs",
        "infodynamics_configs",  # Important that this is infodynamics_configs
        f"{title}_infodynamics_config.yaml",  # Has to contain a timestamp!
    ),
    "r",
) as file:
    experiment_config = yaml.safe_load(file)

experiment = Experiment(**experiment_config)

# Load model checkpoint 
model_checkpoint = os.path.join(experiment.current_run_dir, "checkpoints", "checkpoint-5000", "trainer_state.json")

# Load trainer_state file
with open(model_checkpoint, "r") as file:
    res = json.load(file)

# Create log_history
merged_log_history = []
for i in range(0, len(res["log_history"]), 2):
    merged_dict = {**res["log_history"][i], **res["log_history"][i+1]}
    merged_log_history.append(merged_dict)

# # Save log_history
# with open(os.path.join(experiment.current_run_dir, "log_history.json"), "w") as file:
#     json.dump(merged_log_history, file)

merged_log_history = [x for x in merged_log_history if x["step"] <= experiment.step_cutoff]


# Plot log_history
plt.plot([x["step"] for x in merged_log_history], [x["eval_loss"] for x in merged_log_history], label="eval_loss")
plt.plot([x["step"] for x in merged_log_history], [x["loss"] for x in merged_log_history], label="loss")
plt.plot([x["step"] for x in merged_log_history], [x["eval_accuracy"] for x in merged_log_history], label="eval_accuracy")
plt.legend()
plt.xlabel("Step")
plt.title(title)
plt.grid()
plt.savefig(os.path.join(experiment.current_run_dir, "loss_plot.png"))
plt.show()
