from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import StepMetrics, RunMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_run_directories
from td3.utils.graph_utils import plot_line_chart, plot_episode_weights
from td3.utils.logger import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

def main():
    app_config = AppConfig()
    run_metrics = RunMetrics()
    run_dir, model_dir, weights_dir, plots_dir = create_run_directories()
    app_config.to_json(run_dir)

    dp = DataProcessor(data_dir=app_config.data_dir)
    df = dp.load_panel(
        tickers=app_config.ticker_config.tickers,
        start=app_config.start_date,
        end=app_config.end_date,
    )

    df_features = df.drop(columns=app_config.filter_out, level=1)
    df_prices = df.loc[:, (slice(None), ['close'])]
    data_3d_features, _, tickers, features = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

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
        noise_anneal_episodes=app_config.number_of_episodes,
        learning_starts=app_config.learning_start_episode,
    )

    for episode in range(app_config.number_of_episodes):
        print(f"Starting episode {episode} / {app_config.number_of_episodes}")
        episode_metrics = run_metrics.start_episode()
        state = env.reset(options={"episode_number": episode})
        done = False

        while True:
            if done:
                episode_metrics.aggregate()
                break

            td3_agent.set_episode(episode)
            action, noisy_logits = td3_agent.select_action(state)

            new_state, reward_val, done, trunc, info = env.step(action)
            env.replay_buffer.add(state, action, reward_val, done, new_state)
            state = new_state
            episode_metrics.update(td3_agent.update(env.replay_buffer, batch_size=app_config.batch_size)
                                   .set_basic(float(reward_val), float(env.portfolio_value.curr)))

    td3_agent.save_model(model_dir)

    plot_line_chart(
        [m.total_reward for m in run_metrics.episodes],
        "Reward",
        "Reward per Episode",
        save_dir=plots_dir,
        image_name="rewards.png",
    )
    plot_line_chart(
        [m.final_portfolio_value for m in run_metrics.episodes],
        "Portfolio Value",
        "Portfolio Value per Episode",
        save_dir=plots_dir,
        image_name="portfolio_value.png",
    )
    # plot_line_chart(
    #     run_metrics.actor_losses,
    #     "Actor Loss",
    #     "Average Actor Loss per Episode",
    #     save_dir=plots_dir,
    #     image_name="actor_loss_per_episode.png",
    # )
    #
    # plot_line_chart(
    #     run_metrics.critic_values["c1"],
    #     "Critic 1 Loss",
    #     "Average Critic 1 Loss per Episode",
    #     save_dir=plots_dir,
    #     image_name="critic1_loss_per_episode.png",
    # )
    #
    # plot_line_chart(
    #     run_metrics.critic_values["c2"],
    #     "Critic 2 Loss",
    #     "Average Critic 2 Loss per Episode",
    #     save_dir=plots_dir,
    #     image_name="critic2_loss_per_episode.png",
    # )


if __name__ == "__main__":
    main()