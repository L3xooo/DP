from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import ExperimentMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.utils.file_utils import create_experiment_directories, create_directory
from td3.utils.graph_utils import plot_multi_line_chart, plot_episode_weights
from td3.utils.logger import LoggerFactory

logger = LoggerFactory.create_logger(__name__)

def main():
    experiment_metrics = ExperimentMetrics()
    app_config = AppConfig()
    experiment_dir, plots_dir, models_dir, weights_dir = create_experiment_directories()
    app_config.to_json(experiment_dir)

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

    for iteration in range(app_config.iterations):
        print(f"Starting iteration {iteration + 1} / {app_config.iterations}")
        create_directory(f"{weights_dir}/run_{iteration}")

        env = PortfolioEnv(features=data_3d_features, prices=data_3d_prices,
            tickers=tickers, app_config=app_config)

        td3_agent = TD3(hidden_size=app_config.hidden_size, device=app_config.device,
            state_dim=int(env.observation_space.shape[0]), action_dim=env.action_space.shape[0],
            noise_anneal_episodes=app_config.number_of_episodes, learning_starts=app_config.learning_start_episode)

        run_metrics = experiment_metrics.start_run(run_id=str(iteration))
        for episode in range(app_config.number_of_episodes):
            print(f"Starting episode {episode + 1} / {app_config.number_of_episodes}")

            episode_metrics = run_metrics.start_episode()
            state = env.reset(options={"episode_number": episode})
            done = False

            while True:
                if done:
                    plot_episode_weights(episode_metrics.final_weights, app_config.ticker_config.tickers_with_cash, episode, weights_dir + "/run_" + str(iteration))
                    episode_metrics.aggregate()
                    break

                td3_agent.set_episode(episode)
                action, noisy_logits = td3_agent.select_action(state)

                new_state, reward_val, done, trunc, info = env.step(action)
                env.replay_buffer.add(state, action, reward_val, done, new_state)
                state = new_state
                episode_metrics.update(td3_agent.update(env.replay_buffer, batch_size=app_config.batch_size)
                                       .set_basic(float(reward_val), float(env.portfolio_value.curr), env.weights.curr))

        td3_agent.save_model(models_dir, filename=f'td3_model_run_{iteration}.pth')

    plot_multi_line_chart(
        data_series=[[ep.total_reward for ep in run.episodes]
        for run in experiment_metrics.runs],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Episode Total Reward per Run",
        image_name="episode_rewards.png",
        save_dir=plots_dir,
        y_label="Total Reward")

    plot_multi_line_chart(
        data_series=[[ep.final_portfolio_value for ep in run.episodes]
        for run in experiment_metrics.runs],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Portfolio Value per Run",
        image_name="portfolio_value.png",
        save_dir=plots_dir,
        y_label="Total Portfolio Value")

if __name__ == "__main__":
    main()