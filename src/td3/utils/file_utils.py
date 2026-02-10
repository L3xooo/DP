import os
from datetime import datetime

EXPERIMENT_PREFIX = "experiment"
SIMULATION_PREFIX = "run"
BASE_SIMULATION_DIR = "simulations"
MODEL_DIR = "models"
WEIGHTS_DIR = "weights"
PLOT_DIR = "plots"

def create_experiment_directories(simulation_dir=BASE_SIMULATION_DIR):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create base simulations directory if it doesn't exist
    os.makedirs(simulation_dir, exist_ok=True)

    # Create a new experiment directory with timestamp
    experiment_dir = os.path.join(simulation_dir, f"{EXPERIMENT_PREFIX}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=False)

    plots_dir = os.path.join(experiment_dir, "plots")
    models_dir = os.path.join(experiment_dir, "models")
    weights_dir = os.path.join(experiment_dir, "weights")

    os.makedirs(plots_dir, exist_ok=False)
    os.makedirs(models_dir, exist_ok=False)
    os.makedirs(weights_dir, exist_ok=False)

    return experiment_dir, plots_dir, models_dir, weights_dir

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    return path

def create_run_directories(simulation_dir=BASE_SIMULATION_DIR):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create base simulations directory if it doesn't exist
    os.makedirs(simulation_dir, exist_ok=True)

    # Create a new run directory with timestamp
    run_dir = os.path.join(simulation_dir, f"{SIMULATION_PREFIX}_{timestamp}")
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
