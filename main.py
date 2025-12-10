import numpy as np

from models.td3 import TD3
from environment.portfolio import PortfolioEnv
from utils.logger import LoggerFactory, log_numpy, log_stock_value
from data.data_processor import DataProcessor
import matplotlib.pyplot as plt

# Pt - total portfolio value = Pt-1 + pt^T*kt - previous total + prices * change in shares
# pt - prices vector
# mn,t - number of shares held
# kn,t - number of shares buying/selling/holding

logger = LoggerFactory.create_logger(__name__)

NUMBER_OF_EPISODES = 1

# tickers = ["AAPL", "MSFT"]

tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet (Google)
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "JPM",   # JPMorgan Chase
    "JNJ",   # Johnson & Johnson
    "XOM"    # Exxon Mobil
]

# tickers = [
#     # Technology
#     "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "IBM",
#
#     # Consumer / Retail
#     "AMZN", "HD", "MCD", "NKE", "SBUX", "COST",
#
#     # Financials
#     "JPM", "BAC", "WFC", "GS", "MS",
#
#     # Healthcare
#     "JNJ", "PFE", "MRK", "ABBV", "UNH",
#
#     # Energy
#     "XOM", "CVX", "COP",
#
#     # Industrials
#     "CAT", "BA", "GE",
#
#     # Communications
#     "VZ", "T"
# ]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":

    dp = DataProcessor(data_dir="../indicators")
    df = dp.load_panel(
        tickers=tickers,
        start="2018-01-01",
        end="2019-01-01",
        # filter_cols=None  # netreba písať, default je None → načíta všetko z CSV
    )
    filter_out = [
        'obv', 'volume_base', 'open', 'high', 'low', 'unix'
    ]
    df_features = df.drop(columns=filter_out, level=1)
    df_prices = df.loc[:, (slice(None), ['close'])]
    data_3d_features, _, _, features = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

    # print(len(features))
    # print(data_3d_features.shape)
    # print(data_3d_prices.shape)

    env = PortfolioEnv(features=data_3d_features, prices=data_3d_prices, num_assets=len(tickers), tickers=tickers)

    action_dim = env.action_space.shape[0]
    state_dim = int(env.observation_space.shape[0])
    td3_agent = TD3(state_dim=state_dim, action_dim=action_dim)
    all_rewards = []

    logger.info(f"Actor parameters: {count_params(td3_agent.actor)}")
    logger.info(f"Critic1 parameters: {count_params(td3_agent.critic1)}")
    logger.info(f"Critic2 parameters: {count_params(td3_agent.critic2)}")

    for episode in range(NUMBER_OF_EPISODES):
        logger.info(f"-------------------- Starting Episode {episode} -------------------")
        episode_reward = 0.0
        state = env.reset(options={"episode_number": episode})
        done = False

        while True:
            if done:
                logger.info(f"Episode {episode} finished, total reward {episode_reward:.2f}")
                all_rewards.append(episode_reward)
                # print(episode_reward)
                break
            logger.info("-------------------- New Iteration Step -------------------")
            # get current prices via env helper (close is feature index 0)
            prices = env._get_prices_current()
            action = td3_agent.select_action(state)
            # Logging for debugging
            log_numpy(logger, prices, "Prices")
            log_stock_value(logger, tickers, action, "Stock Weight")

            new_state, reward_val, done, trunc, info = env.step(action)

            env.replay_buffer.add(state, action, reward_val, done, new_state)
            td3_agent.update(env.replay_buffer, batch_size=64)

            episode_reward += float(reward_val)
            state = new_state

        # env.print_stats()
    #
    # plt.plot(all_rewards)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("Training Reward Over Time")
    # plt.show()