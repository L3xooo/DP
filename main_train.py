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
from td3.utils.graph_utils import plot_episode_weights, plot_price_history, plot_multi_line_chart
from td3.utils.logs.logger import LoggerFactory
from td3.utils.torch_utils import count_params
from itertools import accumulate

logger = LoggerFactory.create_logger(__name__)


def main() -> None:
    """
    Main function to run the TD3 training loop for portfolio management.
    """
    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir, rewards_dir = create_experiment_directories()
    app_config = AppConfig()
    app_config.to_json(experiment_dir)

    dp = DataProcessor(data_dir=app_config.data_dir)
    data_3d_features, data_3d_prices, tickers, features, dates = dp.load_data(app_config)
    plot_price_history(data_3d_prices, tickers, dates, save_path=plots_dir + "/price_history.png")

    logger.info("Dates included: %s", len(dates))
    logger.info("Tickers included: %s", tickers)
    logger.info("Features included: %s", features)

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
            lr=app_config.lr,
            dropout_rate=app_config.dropout_rate,
            normalization=app_config.normalization,
        )
        logger.info("Model parameters: %s", count_params(td3_agent.actor))

        run_metrics = experiment_metrics.start_run(run_id=str(iteration))
        for episode in range(app_config.number_of_episodes):

            if episode % 20 == 0 and episode != 0:
                logger.debug("Saving model in episode %d", episode)
                td3_agent.save_model(models_dir, filename=f'td3_model_run_{iteration}_{episode}.pth')

            # logger.info("Starting episode %d / %d", episode + 1, app_config.number_of_episodes)
            episode_metrics = run_metrics.start_episode()
            state = env.reset()
            td3_agent.set_episode_and_noise(episode)
            episode_metrics.exploration_noise = float(td3_agent.policy_noise)
            done = False

            while True:
                if done:
                    cumulative_rewards = list(accumulate(episode_metrics.rewards))
                    plot_multi_line_chart(
                        data_series=[cumulative_rewards],
                        labels=["Cumulative Reward"],
                        title=f"Cumulative Reward - Episode {episode}",
                        image_name=f"episode_{episode}_cumulative_reward.jpg",
                        save_dir=rewards_dir + "/run_" + str(iteration),
                        x_label="Step",
                        y_label="Cumulative Reward",
                    )
                    plot_episode_weights(
                        episode_metrics.final_weights,
                        app_config.ticker_config.tickers_with_cash,
                        episode=episode,
                        save_dir=weights_dir + "/run_" + str(iteration),
                    )
                    episode_metrics.aggregate()
                    logger.debug("Total reward: %s", episode_metrics.total_reward)
                    break
                logger.debug("State: %s", state)
                action, _ = td3_agent.select_action(state)
                new_state, reward_val, done, _, _ = env.step(action)
                if new_state is not None:
                    logger.debug("Reward: %s", reward_val)
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
        logger.info(
            "Train stats | Iteration: %d | Total reward: %.6f | "
            "Portfolio value: %.2f",
            iteration,
            run_metrics.episodes[-1].total_reward,
            run_metrics.episodes[-1].final_portfolio_value,
        )

    provide_train_graphs(plots_dir, experiment_metrics)

if __name__ == "__main__":
    main()
