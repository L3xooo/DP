"""
Test execution pipeline for trained TD3 portfolio agents.

Loads a saved training configuration and model checkpoints from an experiment
directory, runs each model against a held-out test period, and exports
portfolio weight trajectories alongside performance plots.

Author: Peter Likavec
"""

import sys
sys.path.insert(0, 'src')

import os
from typing import List, Tuple

from td3.config.app_config import AppConfig
from td3.data.data_processor import DataProcessor
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import ExperimentMetrics
from td3.models.td3 import TD3
from td3.providers.graph_provider import provide_test_graphs
from td3.utils.file_utils import create_experiment_directories, RunType
from td3.utils.graph_utils import plot_price_history
from td3.utils.logs.logger import LoggerFactory
from td3.utils.torch_utils import count_params

EXPERIMENT_DIR = "/data/teodora_portfolio/train/run_2026-06-10_18-22-46/"
logger = LoggerFactory.create_logger(__name__)

def get_model_paths() -> Tuple[List[str], List[str]]:
    """
    Load all model paths from the "models" directory within the experiment directory.

    Returns:
        Tuple of the paths and names of the models.
    """

    names = []
    paths = []
    for filename in os.listdir(EXPERIMENT_DIR + "models"):
        if filename.endswith(".pth"):
            names.append(filename)
            paths.append(os.path.join(EXPERIMENT_DIR + "models", filename))
    return names, paths

def export_weights(
    experiment_metrics: "ExperimentMetrics",
    model_names: list,
    tickers: list,
    weights_dir: str,
    dates: List[str],
) -> None:
    """
    Create a CSV file for each model containing the weights of the portfolio at each step of
    the episode, along with the corresponding dates.

    Args:
        experiment_metrics: Metrics object containing the runs and episodes with their respective weights.
        model_names: List of model names corresponding to the runs in experiment_metrics.
        tickers: List of tickers corresponding to the weights, including "Cash" as the first entry.
        weights_dir: Directory where the CSV files will be saved.
        dates: List of dates corresponding to each step in the episode.
    """

    for run, name in zip(experiment_metrics.runs, model_names):
        data_series = [step.weights for step in run.episodes[:-2]]
        episode_dates = dates[: len(data_series)]
        save_path = os.path.join(weights_dir, f"weights_{name}.csv")
        with open(save_path, "w") as f:
            f.write("Date," + ",".join(tickers) + "\n")

            for date, weights in zip(episode_dates, data_series):
                f.write(date + "," + ",".join(map(str, weights)) + "\n")


def main() -> None:
    """
    Main function for the TD3 training pipeline.
    """
    app_config = AppConfig.load_from_train_config(
        path=f"{EXPERIMENT_DIR}/config.json", start_date="2019-06-1", end_date="2021-12-30")
    model_names, model_paths = get_model_paths()

    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir, _ = create_experiment_directories(
        run_type=RunType.TEST
    )

    data_3d_features, data_3d_prices, tickers, features, all_dates = DataProcessor(data_dir=app_config.data_dir, parquet_dir=app_config.parquet_dir).load_data(app_config)
    plot_price_history(data_3d_prices, tickers, all_dates, save_path=plots_dir + "/price_history.png")
    steps, _, _= data_3d_features.shape

    logger.info("Loaded data with features of shape %s and prices of shape %s", data_3d_features.shape, data_3d_prices.shape)
    logger.info("Data contains %s steps", steps)

    for model_path in model_paths:
        env = PortfolioEnv(
            features=data_3d_features, prices=data_3d_prices, tickers=tickers, app_config=app_config
        )

        td3_agent = TD3(
            hidden_size=app_config.hidden_size,
            device=app_config.device,
            state_dim=int(env.observation_space.shape[0]),
            action_dim=env.action_space.shape[0],
            noise_anneal_episodes=app_config.number_of_episodes,
            lr=app_config.lr,
            dropout_rate=app_config.dropout_rate,
            normalization=app_config.normalization,
        )

        td3_agent.load_model(model_path)
        td3_agent.actor.eval()
        td3_agent.critic1.eval()
        td3_agent.critic2.eval()
        logger.info("Model parameters: %s", count_params(td3_agent.actor))

        state = env.reset()
        done = False
        run_metrics = experiment_metrics.start_run(model_path)
        while True:
            step_metrics = run_metrics.start_step()
            if done:
                break

            # logger.info("Executing action for on date %s step %s", all_dates[env.current_step], env.current_step)
            action, noisy_logits = td3_agent.select_action(state, None, None, False)
            logger.debug("Action: %s", action)
            logger.debug("Noisy logits: %s", noisy_logits)
            new_state, reward_val, done, _, _ = env.step(action)
            if new_state is not None:
                state = new_state
                total_reward = (
                    run_metrics.episodes[-2].reward if len(run_metrics.episodes) > 1 else 0.0
                ) + float(reward_val)

                step_metrics.set_basic(total_reward, env.portfolio_value.curr, action)

    provide_test_graphs(
        plots_dir,
        weights_dir,
        experiment_metrics,
        model_names,
        app_config.ticker_config.tickers_with_cash,
    )
    export_weights(experiment_metrics, model_names, ["Cash"] + tickers, weights_dir, all_dates)


if __name__ == "__main__":
    main()
