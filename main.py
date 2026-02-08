from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.run_metrics import RunMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_run_directories
from td3.utils.graph_utils import plot_episode_weights, plot_line_chart
from td3.utils.logger import LoggerFactory
import numpy as np

logger = LoggerFactory.create_logger(__name__)

if __name__ == "__main__":
    app_config = AppConfig()
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

    action_dim = env.action_space.shape[0]
    state_dim = int(env.observation_space.shape[0])

    td3_agent = TD3(
        device=app_config.device,
        state_dim=state_dim,
        action_dim=action_dim,
        noise_anneal_episodes=app_config.number_of_episodes,
        learning_starts=app_config.learning_start_episode,
    )

    run_metrics = RunMetrics()
    all_rewards = []
    for episode in range(app_config.number_of_episodes):
        c = 0
        print(f"Episode {episode} starting...")
        logger.info(f"-------------------- Starting Episode {episode} -------------------")
        episode_reward = 0.0
        episode_actor_loss = 0.0
        episode_c1 = 0.0
        episode_c2 = 0.0
        state = env.reset(options={"episode_number": episode})
        done = False
        episode_weights = []

        while True:
            if done:
                logger.info(
                    f"Episode {episode} finished | Reward:  {episode_reward:.2f} portfolio value {env.portfolio_value.curr}"
                )

                run_metrics.rewards.append(episode_reward)
                run_metrics.portfolio_values.append(env.portfolio_value.curr)
                run_metrics.actor_losses.append(episode_actor_loss / c)
                run_metrics.critic_values["c1"].append(episode_c1 / c)
                run_metrics.critic_values["c2"].append(episode_c2 / c)
                plot_episode_weights(
                    episode_weights,
                    app_config.ticker_config.tickers_with_cash,
                    episode,
                    save_dir=weights_dir,
                )
                break

            td3_agent.set_episode(episode)
            action, noisy_logits = td3_agent.select_action(state, 0.75, None, None)

            new_state, reward_val, done, trunc, info = env.step(action)
            env.replay_buffer.add(state, action, reward_val, done, new_state)
            state = new_state

            metrics = td3_agent.update(
                env.replay_buffer, batch_size=app_config.batch_size, temperature=0.75
            )

            all_rewards.append(reward_val)
            episode_reward += float(reward_val)

            if metrics is not None:
                episode_c1 += metrics["critic1_loss"]
                episode_c2 += metrics["critic2_loss"]
                if metrics["actor_loss"] is not None:
                    episode_actor_loss += metrics["actor_loss"]

            if c % 3 == 0:
                episode_weights.append(env.weights.prev)
            c += 1

    r = np.array(all_rewards, dtype=float)
    pos = np.sum(r > 0)
    neg = np.sum(r < 0)
    zero = np.sum(r == 0)

    print(f"positive={pos}, negative={neg}, zero={zero}, total={r.size}")
    print(f"pos%={pos / r.size:.2%}, neg%={neg / r.size:.2%}")

    td3_agent.save_model(model_dir)

    plot_line_chart(
        run_metrics.rewards,
        "Reward",
        "Reward per Episode",
        save_dir=plots_dir,
        image_name="rewards.png",
    )
    plot_line_chart(
        run_metrics.portfolio_values,
        "Portfolio Value",
        "Portfolio Value per Episode",
        save_dir=plots_dir,
        image_name="portfolio_value.png",
    )
    plot_line_chart(
        run_metrics.actor_losses,
        "Actor Loss",
        "Average Actor Loss per Episode",
        save_dir=plots_dir,
        image_name="actor_loss_per_episode.png",
    )

    plot_line_chart(
        run_metrics.critic_values["c1"],
        "Critic 1 Loss",
        "Average Critic 1 Loss per Episode",
        save_dir=plots_dir,
        image_name="critic1_loss_per_episode.png",
    )

    plot_line_chart(
        run_metrics.critic_values["c2"],
        "Critic 2 Loss",
        "Average Critic 2 Loss per Episode",
        save_dir=plots_dir,
        image_name="critic2_loss_per_episode.png",
    )
