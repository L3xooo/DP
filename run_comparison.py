"""
Run TD3 training across multiple regime configurations for comparison.

Usage:
    python run_comparison.py                          # all 4 scenarios, default episodes/iterations
    python run_comparison.py --episodes 20 --iterations 2   # quick test
    python run_comparison.py --regime --hmm           # run only the "both" scenario
"""

import sys
import os
import argparse
import time
sys.path.insert(0, 'src')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines'))

from td3.config.app_config import AppConfig
from td3.environment.portfolio import PortfolioEnv
from td3.metrics.metrics import ExperimentMetrics, StepMetrics
from td3.models.td3 import TD3
from td3.data.data_processor import DataProcessor
from td3.providers.graph_provider import provide_train_graphs
from td3.utils.file_utils import create_experiment_directories, create_directory
from td3.utils.graph_utils import plot_episode_weights, plot_price_history, plot_multi_line_chart
from td3.utils.logs.logger import LoggerFactory
from td3.utils.torch_utils import count_params
from itertools import accumulate
import numpy as np
from baselines.metrics import compute_all_metrics, print_metrics

logger = LoggerFactory.create_logger(__name__)

ALL_SCENARIOS = [
#    ("baseline",    False, False),
#    ("regime_only", True,  False),
    ("hmm_only",    False, True),
    ("both",        True,  True),
]


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 Portfolio Training - Scenario Comparison")
    parser.add_argument("--regime", action="store_true",
                        help="Run only scenarios with regime awareness enabled")
    parser.add_argument("--hmm", action="store_true",
                        help="Run only scenarios with HMM regime enabled")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["baseline", "regime_only", "hmm_only", "both"],
                        help="Run a single named scenario")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per iteration (default: 100)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of independent training runs (default: 5)")
    return parser.parse_args()


def select_scenarios(args):
    """Determine which scenarios to run based on CLI flags."""
    if args.scenario:
        for name, regime, hmm in ALL_SCENARIOS:
            if name == args.scenario:
                return [(name, regime, hmm)]
        raise ValueError(f"Unknown scenario: {args.scenario}")

    if args.regime or args.hmm:
        return [
            (name, regime, hmm)
            for name, regime, hmm in ALL_SCENARIOS
            if (not args.regime or regime) and (not args.hmm or hmm)
        ]

    return list(ALL_SCENARIOS)


def run_scenario(name, enable_regime, enable_hmm, episodes, iterations):
    """Run one full training scenario and return summary stats."""
    logger.info("=" * 60)
    logger.info("SCENARIO: %s  (regime=%s, hmm=%s)", name, enable_regime, enable_hmm)
    logger.info("=" * 60)

    experiment_metrics = ExperimentMetrics()
    experiment_dir, plots_dir, models_dir, weights_dir, rewards_dir = create_experiment_directories(
        prefix=name,
    )

    app_config = AppConfig(
        enable_regime_awareness=enable_regime,
        enable_hmm_regime=enable_hmm,
        number_of_episodes=episodes,
        iterations=iterations,
    )
    app_config.to_json(experiment_dir)

    dp = DataProcessor(data_dir=app_config.data_dir, parquet_dir=None)
    data_3d_features, data_3d_prices, tickers, features, dates, hmm_regime_probs = dp.load_data(
        app_config, add_regime=app_config.enable_regime_awareness
    )
    plot_price_history(data_3d_prices, tickers, dates, save_path=plots_dir + "/price_history.png")

    logger.info("Dates: %d | Tickers: %s | Features: %s", len(dates), tickers, features)

    final_rewards = []
    final_values = []
    last_portfolio_series = None  # V series from the last episode of the last iteration

    for iteration in range(app_config.iterations):
        logger.info("Starting iteration %d", iteration)
        create_directory(f"{weights_dir}/run_{iteration}")

        env = PortfolioEnv(
            features=data_3d_features, prices=data_3d_prices, tickers=tickers,
            app_config=app_config,
            hmm_regime_probs=hmm_regime_probs,
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
                td3_agent.save_model(models_dir, filename=f'td3_model_run_{iteration}_{episode}.pth')

            logger.info("Starting episode %d / %d", episode + 1, app_config.number_of_episodes)
            episode_metrics = run_metrics.start_episode()
            state = env.reset()
            td3_agent.set_episode_and_noise(episode)
            episode_metrics.exploration_noise = float(td3_agent.policy_noise)
            done = False
            step_count = 0
            t_select = 0.0; t_step = 0.0; t_update = 0.0

            while True:
                if done:
                    is_last_episode = episode == app_config.number_of_episodes - 1
                    if episode % 10 == 0 or is_last_episode:
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
                    # Capture portfolio value series before aggregate() clears steps
                    if episode == app_config.number_of_episodes - 1:
                        last_portfolio_series = np.array(
                            [s.portfolio_value for s in episode_metrics.steps]
                        )
                    episode_metrics.aggregate()
                    logger.info(
                        "TIMING ep %d | select_action: %.4fs | env.step: %.4fs | td3.update: %.4fs | steps: %d",
                        episode, t_select, t_step, t_update, step_count,
                    )
                    break

                t0 = time.time()
                action, _ = td3_agent.select_action(state)
                t_select += time.time() - t0

                t0 = time.time()
                new_state, reward_val, done, _, _ = env.step(action)
                t_step += time.time() - t0

                if new_state is not None:
                    env.replay_buffer.add(state, action, reward_val, done, new_state)
                    state = new_state
                    step_count += 1

                    step_metrics = StepMetrics()
                    if step_count % app_config.update_every == 0:
                        t0 = time.time()
                        step_metrics = td3_agent.update(
                            env.replay_buffer,
                            batch_size=app_config.batch_size,
                        )
                        t_update += time.time() - t0

                    episode_metrics.update(
                        step_metrics.set_basic(
                            float(reward_val), float(env.portfolio_value.curr), env.weights.curr
                        )
                    )

        td3_agent.save_model(models_dir, filename=f'td3_model_run_{iteration}.pth')

        final_rewards.append(run_metrics.episodes[-1].total_reward)
        final_values.append(run_metrics.episodes[-1].final_portfolio_value)

        logger.info(
            "Train stats | Iteration: %d | Total reward: %.6f | Portfolio value: %.2f",
            iteration, final_rewards[-1], final_values[-1],
        )

    provide_train_graphs(plots_dir, experiment_metrics)

    # Compute unified metrics on the last episode's portfolio value series
    scenario_metrics = None
    if last_portfolio_series is not None and len(last_portfolio_series) > 1:
        scenario_metrics = compute_all_metrics(
            last_portfolio_series, bars_per_day=1,
        )
        print_metrics(scenario_metrics, label=f"TD3 {name.upper()} (last episode)")

    return {
        "scenario": name,
        "regime": enable_regime,
        "hmm": enable_hmm,
        "experiment_dir": experiment_dir,
        "final_rewards": final_rewards,
        "final_values": final_values,
        "metrics": scenario_metrics,
    }


def main():
    args = parse_args()
    scenarios = select_scenarios(args)

    print(f"\nRunning {len(scenarios)} scenario(s): "
          f"{[s[0] for s in scenarios]}")
    print(f"Episodes: {args.episodes} | Iterations: {args.iterations}\n")

    results = []
    for name, regime, hmm in scenarios:
        t0 = time.time()
        result = run_scenario(name, regime, hmm, args.episodes, args.iterations)
        result["elapsed_s"] = time.time() - t0
        results.append(result)

    # Print comparison summary
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Scenario':<15} {'Regime':<8} {'HMM':<6} "
          f"{'Avg Reward':>12} {'Avg Value':>12} "
          f"{'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Time (s)':>10}")
    print("-" * 100)
    for r in results:
        avg_reward = sum(r["final_rewards"]) / len(r["final_rewards"])
        avg_value = sum(r["final_values"]) / len(r["final_values"])
        m = r.get("metrics")
        sharpe = f"{m['sharpe_ratio']:.4f}" if m else "N/A"
        mdd = f"{m['max_drawdown']:.2%}" if m else "N/A"
        cagr_val = f"{m['cagr']:.2%}" if m else "N/A"
        print(f"{r['scenario']:<15} {str(r['regime']):<8} {str(r['hmm']):<6} "
              f"{avg_reward:>12.4f} {avg_value:>12.2f} "
              f"{sharpe:>8} {mdd:>8} {cagr_val:>8} {r['elapsed_s']:>10.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
