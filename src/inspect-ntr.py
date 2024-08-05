import os
import yaml
import ndjson
from src.utils.experiments import Experiment
import matplotlib.pyplot as plt

# Load specific run of the experiment
with open(
    os.path.join(
        "src",
        "configs",
        "infodynamics_configs",  # Important that this is infodynamics_configs
        "distilbert-base-uncased-MNLI-infodynamcis_config.yaml",  # Has to contain a timestamp!
    ),
    "r",
) as file:
    experiment_config = yaml.safe_load(file)

experiment = Experiment(**experiment_config)

# Load NTR results
with open(f"{experiment.current_run_dir}/NTR_results.json", "r") as fin:
    ntr_results = ndjson.load(fin)

# Load resonance
resonance = ntr_results[0]["resonance"]

# Load novelty
novelty = ntr_results[0]["novelty"]

# Load transience
transience = ntr_results[0]["transience"]


# plot novelty over time 
plt.plot(novelty)
plt.xlabel("Steps")
plt.ylabel("Novelty")
