"""
Experiment directory management utilities for the TD3 training pipeline.

Provides helpers for creating timestamped experiment directories with
standardized subdirectories for models, weights, and plots.

Author: Peter Likavec
"""

import os
from datetime import datetime
from enum import Enum

EXPERIMENT_PREFIX = "run"
BASE_SIMULATION_DIR = "simulations"
MODEL_DIR = "models"
WEIGHTS_DIR = "weights"
PLOT_DIR = "plots"


class RunType(str, Enum):
    """Enum distinguishing between training and evaluation runs."""

    TRAIN = "train"
    TEST = "test"


def create_experiment_directories(
    simulation_dir: str = BASE_SIMULATION_DIR,
    prefix: str = EXPERIMENT_PREFIX,
    run_type: RunType = RunType.TRAIN,
):
    """
    Create a timestamped experiment directory with standard subdirectories.

    Creates plots/, models/, and weights/ subdirectories inside a new
    timestamped folder under simulations/train or simulations/test.
    Also initializes an empty notes.md file in the experiment root.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(simulation_dir, exist_ok=True)

    # Create train and test simulations_directories if they don't exist
    os.makedirs(os.path.join(simulation_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(simulation_dir, "test"), exist_ok=True)

    simulation_dir = os.path.join(simulation_dir, "train" if run_type == RunType.TRAIN else "test")

    # Create a new experiment directory with timestamp
    experiment_dir = os.path.join(simulation_dir, f"{prefix}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=False)

    plots_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir, exist_ok=False)

    models_dir = os.path.join(experiment_dir, "models")
    os.makedirs(models_dir, exist_ok=False)

    weights_dir = os.path.join(experiment_dir, "weights")
    os.makedirs(weights_dir, exist_ok=False)

    with open(os.path.join(experiment_dir, "notes.md"), "w") as f:
        f.write(f"# {prefix}_{timestamp}")

    return experiment_dir, plots_dir, models_dir, weights_dir


def create_directory(path):
    """Create a directory at the given path if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    return path
