from config.ticker_config import get_ticker_config
from environment.portfolio import PortfolioEnv
from models.td3 import TD3
from utils.file_utils import create_run_directories
from utils.graph_utils import plot_episode_weights, plot_line_chart
from utils.logger import LoggerFactory
from data.data_processor import DataProcessor

logger = LoggerFactory.create_logger(__name__)

NUMBER_OF_EPISODES = 1000
LEARNING_START_EPISODE = 100

tickers = get_ticker_config("10_TICKERS").tickers
cash_tickers = ["Cash"] + tickers.copy()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    run_dir, model_dir, weights_dir, plots_dir = create_run_directories()

    dp = DataProcessor(data_dir="../indicators")
    df = dp.load_panel(tickers=tickers, start="2016-05-01", end="2019-01-18",)
    filter_out = ['obv', 'volume_base', 'open', 'high', 'low', 'unix']
    df_features = df.drop(columns=filter_out, level=1)
    df_prices = df.loc[:, (slice(None), ['close'])]
    data_3d_features, _, tickers, features = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

    env = PortfolioEnv(features=data_3d_features, prices=data_3d_prices, tickers=tickers)

    action_dim = env.action_space.shape[0]
    state_dim = int(env.observation_space.shape[0])

    logger.info(f"state_dim: {state_dim}")
    logger.info(f"action_dim: {action_dim}")

    td3_agent = TD3(state_dim=state_dim, action_dim=action_dim, noise_anneal_episodes = NUMBER_OF_EPISODES,
                    learning_starts=LEARNING_START_EPISODE)
    all_rewards = []
    all_portfolio_values = []
    all_critic_values = {
        "c1": [],
        "c2": []
    }
    all_weights = []

    for episode in range(NUMBER_OF_EPISODES):
            c = 0
            logger.info(f"-------------------- Starting Episode {episode} -------------------")
            episode_reward = 0.0
            state = env.reset(options={"episode_number": episode})
            done = False

            prev_state = None
            prev_action = None
            episode_weights = []
            while True:
                # step_bar.update(1)
                if done:
                    logger.info(f"Episode {episode} finished, total reward {episode_reward:.2f} portfolio value {env.portfolio_value.curr}")
                    all_rewards.append(episode_reward)
                    all_portfolio_values.append(float(env.portfolio_value.curr))
                    plot_episode_weights(episode_weights, cash_tickers, episode, save_dir=weights_dir)
                    break

                # logger.info("-------------------- New Iteration Step -------------------")
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

    print(all_rewards)
    print(all_portfolio_values)
    td3_agent.save_model(model_dir)

    plot_line_chart(all_rewards, "Reward", "Reward per Episode", image_name="rewards.png", save_dir=plots_dir)
    plot_line_chart(all_portfolio_values, "Portfolio Value", "Portfolio Value per Episode", save_dir=plots_dir, image_name="portfolio_value.png")