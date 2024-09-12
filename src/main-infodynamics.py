import numpy as np
import os
import yaml
from utils.experiments import Experiment
from utils.utils_infodynamics import (
    calc_ntr,
    calculate_resonance_novelty_slope,
    curb_incomplete_signal,
    load_and_reshape_logits_from_dir,
)
import json


def main(experiment: Experiment): 

    # Load logits from the directory
    logits = load_and_reshape_logits_from_dir(
        path = os.path.join(experiment.current_run_dir, "logits"),
        step_cutoff=experiment.step_cutoff
    )

    # Apply softmax to logits
    logits = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    # Calculate NTR
    window = experiment.window
    model = calc_ntr(logits, window=window) # visualize=True

    novelty = curb_incomplete_signal(model.nsignal, window=window)
    transience = curb_incomplete_signal(model.tsignal, window=window)
    resonance = curb_incomplete_signal(model.rsignal, window=window)

    results = {
        "novelty": novelty.tolist(),
        "transience": transience.tolist(),
        "resonance": resonance.tolist(),
    }

    with open(f"{experiment.current_run_dir}/NTR-results_window-{experiment.window}_step_cutoff-{experiment.step_cutoff}.json", "w") as fout:
        json.dump(results, fout)


if __name__ == "__main__":
    # Load experiment info from config file
    with open(
        os.path.join(
            "src",
            "configs",
            "infodynamics_configs",  # Important that this is infodynamics_configs
            "distilbert-base-uncased-MRPC_infodynamics_config.yaml",  # Has to contain a timestamp!
        ),
        "r",
    ) as file:
        experiment_config = yaml.safe_load(file)

    # Load specific run of the experiment
    experiment = Experiment(**experiment_config)

    main(experiment)
