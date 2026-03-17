import os
from typing import List

from td3.config.app_config import AppConfig
from td3.data.data_processor import DataProcessor
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import ExperimentMetrics
from td3.models.td3 import TD3
from td3.providers.graph_provider import provide_test_graphs
from td3.utils.date_utils import check_if_later_date
from td3.utils.file_utils import create_experiment_directories, RunType
from td3.utils.logger import log_stock_value, LoggerFactory

EXPERIMENT_DIR = "simulations/train/run_2026-03-10_07-07-56/"
logger = LoggerFactory.create_logger(__name__)

def load_config() -> AppConfig:
    config_path = os.path.join(EXPERIMENT_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return AppConfig.from_json(config_path)


def get_model_paths():
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
    """Export all episode weights for each run directly to CSV."""

    for run, name in zip(experiment_metrics.runs, model_names):
        # Each episode's weights
        data_series = [step.weights for step in run.episodes[:-1]]

        # Align dates with episodes
        episode_dates = dates[:len(data_series)]

        save_path = os.path.join(weights_dir, f"weights_{name}.csv")
        with open(save_path, "w") as f:
            # Header: Date + tickers as-is
            f.write("Date," + ",".join(tickers) + "\n")

            # Write each episode
            for date, weights in zip(episode_dates, data_series):
                f.write(date + "," + ",".join(map(str, weights)) + "\n")

def main():
    train_config = load_config()
    model_names, model_paths = get_model_paths()

    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir = create_experiment_directories(
        run_type=RunType.TEST
    )
    app_config = AppConfig(
        start_date="2019-01-21",
        end_date="2023-01-01",
        number_of_episodes=1,
    )

    check_if_later_date(app_config.start_date, train_config.end_date)
    dp = DataProcessor(data_dir=app_config.data_dir)
    data_3d_features, data_3d_prices, tickers, features, all_dates = dp.load_data(app_config)

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
        )
        td3_agent.load_model(model_path)

        state = env.reset()
        done = False
        run_metrics = experiment_metrics.start_run(model_path)
        counter = 0
        while True:
            print(all_dates[counter])
            step_metrics = run_metrics.start_step()
            if done:
                break

            action, noisy_logits = td3_agent.select_action(state, None, None, False)

            new_state, reward_val, done, trunc, info = env.step(action)
            state = new_state
            total_reward = (
                run_metrics.episodes[-2].reward if len(run_metrics.episodes) > 1 else 0.0
            ) + float(reward_val)
            step_metrics.set_basic(total_reward, env.portfolio_value.curr, action)
            counter += 1

    provide_test_graphs(plots_dir, weights_dir, experiment_metrics, model_names, app_config.ticker_config.tickers_with_cash)
    export_weights(experiment_metrics, model_names, ["Cash"] + tickers, weights_dir, all_dates)

if __name__ == "__main__":
    main()
