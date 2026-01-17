from models.td3 import TD3
from environment.portfolio import PortfolioEnv
from utils.graph_utils import plot_episode_weights
from utils.logger import LoggerFactory, log_stock_value, log_values_with_color
from data.data_processor import DataProcessor
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import torch

logger = LoggerFactory.create_logger(__name__)

NUMBER_OF_EPISODES = 500
LEARNING_START_EPISODE = 1

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
        start="2017-01-01",
        end="2020-01-01",
        # filter_cols=None  # netreba písať, default je None → načíta všetko z CSV
    )
    filter_out = [
        'obv', 'volume_base', 'open', 'high', 'low', 'unix'
    ]
    df_features = df.drop(columns=filter_out, level=1)
    df_prices = df.loc[:, (slice(None), ['close'])]
    data_3d_features, _, tickers, features = dp.to_3d(df_features)
    data_3d_prices, _, _, _ = dp.to_3d(df_prices)

    logger.info(len(features))
    logger.info(tickers)
    logger.info(features)
    logger.info(data_3d_features.shape)
    logger.info(data_3d_prices.shape)

    env = PortfolioEnv(features=data_3d_features, prices=data_3d_prices, num_assets=len(tickers), tickers=tickers)

    action_dim = env.action_space.shape[0]
    state_dim = int(env.observation_space.shape[0])
    td3_agent = TD3(state_dim=state_dim, action_dim=action_dim, noise_anneal_episodes = NUMBER_OF_EPISODES, learning_starts=LEARNING_START_EPISODE)
    all_rewards = []
    all_portfolio_values = []
    all_critic_values = {
        "c1": [],
        "c2": []
    }
    all_weights = []

    log_values_with_color(logger, {"Actor" : count_params(td3_agent.actor), "Critic1":
        count_params(td3_agent.critic1), "Critic2": count_params(td3_agent.critic2)}, use_color=False, level="info",)

    # with logging_redirect_tqdm():
    #     for episode in tqdm(range(NUMBER_OF_EPISODES), desc="Training episodes"):
    for episode in range(NUMBER_OF_EPISODES):
            c = 0
            logger.info(f"-------------------- Starting Episode {episode} -------------------")
            episode_reward = 0.0
            state = env.reset(options={"episode_number": episode})
            done = False

            prev_state = None
            prev_action = None
            episode_weights = []
            # step_bar = tqdm(desc=f"Episode {episode} steps", leave=False)
            while True:
                c += 1
                # step_bar.update(1)
                if done:
                    logger.info(f"Episode {episode} finished, total reward {episode_reward:.2f}")
                    all_rewards.append(episode_reward)
                    all_portfolio_values.append(env.new_portfolio_value)
                    # print(episode_reward)
                    all_weights.append(env.new_shares_weights)
                    plot_episode_weights(episode_weights, tickers, episode)

                    break
                logger.info("-------------------- New Iteration Step -------------------")
                # get current prices via env helper (close is feature index 0)

                # log_stock_value(logger, tickers, env.get_prices_previous(), "P Prices")
                # log_stock_value(logger, tickers, env.get_prices_current(), "C Prices")
                # log_stock_value(logger, tickers, env.get_prices_current() - env.get_prices_previous(), "D Prices", use_color=True)

                td3_agent.set_episode(episode)
                action, noisy_logits = td3_agent.select_action(state, 1, None, None)

                new_state, reward_val, done, trunc, info = env.step(action)
                if prev_state is not None:
                    env.replay_buffer.add(prev_state, prev_action, reward_val, done, state)

                prev_state = state
                prev_action = action
                state = new_state

                c1, c2 = td3_agent.update(env.replay_buffer, batch_size=128)
                if c1 is not None and c2 is not None:
                    all_critic_values["c1"].append(c1)
                    all_critic_values["c2"].append(c2)
                    # logger.info(f"Critic1 Value: {c1.item():.4f}, Critic2 Value: {c2.item():.4f}")

                episode_reward += float(reward_val)
                if c % 3 == 0:
                    episode_weights.append(env.new_shares_weights)
                # print(episode_weights)
            # step_bar.close()

        # env.print_stats()
    #
    # logger.info(env.replay_buffer.size())
    # print(all_rewards)
    # print(env.new_portfolio_weights_history)

    td3_agent.save_model("td3_v2")

    plt.figure(figsize=(10, 5))

    plt.plot(all_critic_values["c1"], label="Critic 1")
    plt.plot(all_critic_values["c2"], label="Critic 2")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Metrics Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.axvline(x=LEARNING_START_EPISODE, color='r', linestyle='--',
                label=f'Start Learning at Episode {LEARNING_START_EPISODE}')
    plt.ylabel("Reward")
    plt.title("Training Reward Over Time")
    plt.show()

    plt.plot(all_portfolio_values)
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Portfolio Value Over Time")
    plt.show()

    #
    # plt.figure(figsize=(10, 6))
    # plt.stackplot(range(len(all_weights)), labels=[f' {i}' for i in tickers],
    #               alpha=0.7)
    # plt.title("Portfolio Weights Over Time")
    # plt.xlabel("Time Step")
    # plt.ylabel("Portfolio Weight")
    # plt.legend(title="Actions", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.show()