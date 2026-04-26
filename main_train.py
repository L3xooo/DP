"""
Main training loop for the TD3 agent in the portfolio management environment.

This script initializes the environment, loads data, and runs multiple iterations of training,
logging metrics and saving models and plots for analysis.

Author: Peter Likavec
"""

from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import ExperimentMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.providers.graph_provider import provide_train_graphs
from td3.utils.file_utils import create_experiment_directories, create_directory
from td3.utils.graph_utils import plot_episode_weights, plot_price_history
from td3.utils.logs.logger import LoggerFactory

logger = LoggerFactory.create_logger(__name__)


def main() -> None:
    """
    Main function to run the TD3 training loop for portfolio management.
    """
    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir = create_experiment_directories()
    app_config = AppConfig()
    app_config.to_json(experiment_dir)

    dp = DataProcessor(data_dir=app_config.data_dir)
    data_3d_features, data_3d_prices, tickers, features, dates = dp.load_data(app_config)
    plot_price_history(data_3d_prices, tickers, dates, save_path=plots_dir + "/price_history.png")

    for iteration in range(app_config.iterations):
        logger.info("Starting iteration %d", iteration)
        create_directory(f"{weights_dir}/run_{iteration}")

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

        run_metrics = experiment_metrics.start_run(run_id=str(iteration))
        for episode in range(app_config.number_of_episodes):
            logger.info("Starting episode %d / %d", episode + 1, app_config.number_of_episodes)

            episode_metrics = run_metrics.start_episode()
            state = env.reset(options={"episode_number": episode})
            done = False

            while True:
                if done:
                    plot_episode_weights(
                        episode_metrics.final_weights,
                        app_config.ticker_config.tickers_with_cash,
                        episode=episode,
                        save_dir=weights_dir + "/run_" + str(iteration),
                    )
                    episode_metrics.aggregate()
                    break

                td3_agent.set_episode(episode)
                action, noisy_logits = td3_agent.select_action(state)

                new_state, reward_val, done, _, _ = env.step(action)
                if new_state is not None:
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

        td3_agent.save_model(models_dir, filename=f'td3_model_run_{iteration}.pth')

    provide_train_graphs(plots_dir, experiment_metrics)


if __name__ == "__main__":
    main()
