import argparse
import json
from dataclasses import replace
from pathlib import Path

import optuna

from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import EpisodeMetrics, ExperimentMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_experiment_directories, create_directory
from td3.utils.logger import LoggerFactory, log_stock_value

logger = LoggerFactory.create_logger(__name__)


def build_app_config(base_config: AppConfig, trial: optuna.Trial, episodes: int) -> AppConfig:
    noise_init = trial.suggest_float("noise_init", 0.1, 0.5)
    noise_final_max = max(0.01, min(0.1, noise_init))
    noise_final = trial.suggest_float("noise_final", 0.01, noise_final_max)

    return replace(
        base_config,
        iterations=1,
        number_of_episodes=episodes,
        hidden_size=trial.suggest_categorical("hidden_size", [128, 256, 512]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        noise_init=noise_init,
        noise_final=noise_final,
        noise_anneal_episodes=trial.suggest_int(
            "noise_anneal_episodes", 10, max(10, episodes * 2), step=10
        ),
    )


def compute_objective(run_metrics: ExperimentMetrics, window: int) -> float:
    if not run_metrics.runs:
        return 0.0

    episodes = [ep for ep in run_metrics.runs[-1].episodes if isinstance(ep, EpisodeMetrics)]
    if not episodes:
        return 0.0

    window = max(1, min(window, len(episodes)))
    recent = episodes[-window:]
    values = [ep.final_portfolio_value for ep in recent]
    return float(sum(values) / len(values))


def run_training(
    app_config: AppConfig,
    data_3d_features,
    data_3d_prices,
    tickers,
    log_actions: bool,
) -> tuple[float, str]:
    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir = create_experiment_directories(prefix="hpo")
    app_config.to_json(experiment_dir)

    for iteration in range(app_config.iterations):
        create_directory(f"{weights_dir}/run_{iteration}")

        env = PortfolioEnv(
            features=data_3d_features,
            prices=data_3d_prices,
            tickers=tickers,
            app_config=app_config,
        )

        td3_agent = TD3(
            hidden_size=app_config.hidden_size,
            device=app_config.device,
            state_dim=int(env.observation_space.shape[0]),
            action_dim=env.action_space.shape[0],
            lr=app_config.lr,
            noise_init=app_config.noise_init,
            noise_final=app_config.noise_final,
            noise_anneal_episodes=app_config.noise_anneal_episodes,
        )

        run_metrics = experiment_metrics.start_run(run_id=str(iteration))
        for episode in range(app_config.number_of_episodes):
            episode_metrics = run_metrics.start_episode()
            state = env.reset(options={"episode_number": episode})
            done = False

            while True:
                if done:
                    episode_metrics.aggregate()
                    break

                td3_agent.set_episode(episode)
                action, _ = td3_agent.select_action(state)

                if log_actions:
                    log_stock_value(
                        logger,
                        app_config.ticker_config.tickers_with_cash,
                        action,
                        "Action Weights",
                        decimals=4,
                        use_color=True,
                    )

                new_state, reward_val, done, trunc, info = env.step(action)
                env.replay_buffer.add(state, action, reward_val, done, new_state)
                state = new_state
                episode_metrics.update(
                    td3_agent.update(
                        env.replay_buffer,
                        batch_size=app_config.batch_size,
                    ).set_basic(
                        float(reward_val), float(env.portfolio_value.curr), env.weights.curr
                    )
                )

        td3_agent.save_model(models_dir, filename=f"td3_model_run_{iteration}.pth")

    return experiment_metrics, experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--metric-window", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--study-name", type=str, default="td3_hpo")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--best-params-path", type=str, default="hpo_best_params.json")
    parser.add_argument("--log-actions", action="store_true")
    args = parser.parse_args()

    base_config = AppConfig()
    dp = DataProcessor(data_dir=base_config.data_dir)
    df = dp.load_panel(
        tickers=base_config.ticker_config.tickers,
        start=base_config.start_date,
        end=base_config.end_date,
    )

    df_features = df.loc[:, (slice(None), base_config.filter_in)]
    df_prices = df.loc[:, (slice(None), ["close"])]

    data_3d_features, _, tickers, _ = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

    sampler = optuna.samplers.TPESampler(seed=args.seed) if args.seed is not None else None
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        app_config = build_app_config(base_config, trial, episodes=args.episodes)
        experiment_metrics, experiment_dir = run_training(
            app_config,
            data_3d_features,
            data_3d_prices,
            tickers,
            log_actions=args.log_actions,
        )
        trial.set_user_attr("experiment_dir", experiment_dir)
        return compute_objective(experiment_metrics, window=args.metric_window)

    study.optimize(objective, n_trials=args.n_trials)

    best_payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": study.study_name,
    }
    Path(args.best_params_path).write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()
