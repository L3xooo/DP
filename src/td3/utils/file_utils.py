import os
from datetime import datetime

SIMULATION_PREFIX = "run"
BASE_SIMULATION_DIR = "simulations"
MODEL_DIR = "models"
WEIGHTS_DIR = "weights"
PLOT_DIR = "plots"


def create_run_directories():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create base simulations directory if it doesn't exist
    os.makedirs(BASE_SIMULATION_DIR, exist_ok=True)

    # Create a new run directory with timestamp
    run_dir = os.path.join(BASE_SIMULATION_DIR, f"{SIMULATION_PREFIX}_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)

    # Define subdirectory paths
    model_dir = os.path.join(run_dir, "model")
    weights_dir = os.path.join(run_dir, "weights")
    plots_dir = os.path.join(run_dir, "plots")

    # Create subdirectories
    os.makedirs(model_dir, exist_ok=False)
    os.makedirs(weights_dir, exist_ok=False)
    os.makedirs(plots_dir, exist_ok=False)

    return run_dir, model_dir, weights_dir, plots_dir
