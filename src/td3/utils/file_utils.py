"""
Experiment directory management utilities.

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
    Creates plots/, models/, and weights/ subdirectories inside a new timestamped folder under
    simulations/train or simulations/test.

    Args:
        simulation_dir: Base directory for simulations (default: "simulations").
        prefix: Prefix for the experiment directory name (default: "run").
        run_type: Type of run, either RunType.TRAIN or RunType.TEST (default: RunType.TRAIN).
    Returns:
        A tuple containing the paths to the experiment directory, plots directory,
        models directory, and weights directory.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_type_dir = "train" if run_type == RunType.TRAIN else "test"
    experiment_dir = os.path.join(simulation_dir, run_type_dir, f"{prefix}_{timestamp}")

    sub_dirs = ["plots", "models", "weights"]
    plots_dir, models_dir, weights_dir = [
        os.makedirs(os.path.join(experiment_dir, d), exist_ok=True)
        or os.path.join(experiment_dir, d)
        for d in sub_dirs
    ]

    with open(os.path.join(experiment_dir, "notes.md"), "w") as f:
        f.write(f"# {prefix}_{timestamp}")

    return experiment_dir, plots_dir, models_dir, weights_dir


def create_directory(path: str) -> str:
    """
    Creates a directory at the specified path if it does not already exist.

    Args:
        path: The directory path to create.

    Returns:
        The path that was created or already exists.
    """
    os.makedirs(path, exist_ok=True)
    return path