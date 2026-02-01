from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_run_directories
from td3.utils.graph_utils import plot_episode_weights
from td3.utils.logger import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

if __name__ == "__main__":
    app_config = AppConfig()
    run_dir, model_dir, weights_dir, plots_dir = create_run_directories()
    app_config.to_json(run_dir)

    dp = DataProcessor(data_dir=app_config.data_dir)
    df = dp.load_panel(tickers=app_config.ticker_config.tickers, start=app_config.start_date, end=app_config.end_date)

    df_features = df.drop(columns=app_config.filter_out, level=1)
    df_prices = df.loc[:, (slice(None), ['close'])]
    data_3d_features, _, tickers, features = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

    env = PortfolioEnv(features=data_3d_features, prices=data_3d_prices, tickers=tickers, app_config=app_config)

    action_dim = env.action_space.shape[0]
    state_dim = int(env.observation_space.shape[0])

    td3_agent = TD3(state_dim=state_dim,
                    action_dim=action_dim,
                    noise_anneal_episodes = app_config.number_of_episodes,
                    learning_starts=app_config.learning_start_episode)

    for episode in range(app_config.number_of_episodes):
        c = 0
        logger.info(f"-------------------- Starting Episode {episode} -------------------")
        episode_reward = 0.0
        state = env.reset(options={"episode_number": episode})
        done = False
        episode_weights = []

        while True:
            if done:
                logger.info(f"Episode {episode} finished, total reward {episode_reward:.2f} portfolio value {env.portfolio_value.curr}")
                # all_rewards.append(episode_reward)
                # all_portfolio_values.append(float(env.portfolio_value.curr))
                plot_episode_weights(episode_weights, app_config.ticker_config.tickers_with_cash, episode, save_dir=weights_dir)
                break

            td3_agent.set_episode(episode)
            action, noisy_logits = td3_agent.select_action(state, 0.75, None, None)

            new_state, reward_val, done, trunc, info = env.step(action)
            env.replay_buffer.add(state, action, reward_val, done, new_state)
            state = new_state

            c1, c2 = td3_agent.update(env.replay_buffer, batch_size=128, temperature=0.75)

            episode_reward += float(reward_val)

            if c % 3 == 0:
                episode_weights.append(env.weights.prev)
            c += 1

    td3_agent.save_model(model_dir)
    #
    # plot_line_chart(all_rewards, "Reward", "Reward per Episode", image_name="rewards.png", save_dir=plots_dir)
    # plot_line_chart(all_portfolio_values, "Portfolio Value", "Portfolio Value per Episode", save_dir=plots_dir, image_name="portfolio_value.png")