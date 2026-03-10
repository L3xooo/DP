from typing import List

from td3.metrics.metrics import ExperimentMetrics
from td3.utils.graph_utils import plot_multi_line_chart, plot_episode_weights


def provide_test_graphs(plots_dir: str, weight_dir: str, experiment_metrics: ExperimentMetrics,
                        model_names: List[str], tickers: List[str]):

    plot_multi_line_chart(
        data_series=[[step.reward for step in run.episodes[:-1]] for run in experiment_metrics.runs],
        labels=model_names,
        title="Cumulative per Run",
        image_name="episode_rewards.png",
        save_dir=plots_dir,
        y_label="Total Reward",
    )

    plot_multi_line_chart(
        data_series=[
            [step.portfolio_value for step in run.episodes[:-1]] for run in experiment_metrics.runs
        ],
        labels=model_names,
        title="Portfolio Value per Run",
        image_name="portfolio_reward.png",
        save_dir=plots_dir,
        y_label="Total Value",
    )

    for run, name in zip(experiment_metrics.runs, model_names):
        # Take all episodes except the last one
        data_series = [step.weights for step in run.episodes[:-1]]

        plot_episode_weights(
            data_series,
            tickers,
            episode=0,
            save_dir=weight_dir,
            filename=f"weights_{name}.png",
            title=f"Weights per Episode - {name}"
        )

def provide_graphs(plots_dir: str, experiment_metrics):
    plot_multi_line_chart(
        data_series=[[ep.total_reward for ep in run.episodes] for run in experiment_metrics.runs],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Episode Total Reward per Run",
        image_name="episode_rewards.png",
        save_dir=plots_dir,
        y_label="Total Reward",
    )

    plot_multi_line_chart(
        data_series=[
            [ep.final_portfolio_value for ep in run.episodes] for run in experiment_metrics.runs
        ],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Portfolio Value per Run",
        image_name="portfolio_value.png",
        save_dir=plots_dir,
        y_label="Total Portfolio Value",
    )

    plot_multi_line_chart(
        data_series=[
            [loss for ep in run.episodes[1:] for loss in ep.actor_loss_all_steps]
            for run in experiment_metrics.runs
        ],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Actor Loss per Training Step",
        image_name="actor_loss_steps.png",
        save_dir=plots_dir,
        y_label="Actor Loss",
    )

    for run in experiment_metrics.runs:
        plot_multi_line_chart(
            data_series=[
                [
                    float(loss)
                    for ep in run.episodes
                    for loss in getattr(ep, "all_critic1_loss", [])
                    if loss is not None
                ]
            ],
            labels=[r.run_id for r in experiment_metrics.runs],
            title="Critic1 Loss per Update",
            image_name=f"all_critic1_loss_{run.run_id}.png",
            save_dir=plots_dir,
            x_label="Training update",  # nie Episode
            y_label="Critic1 Loss",
            y_scale="log",
            stride=10,
            skip_first=0,
        )

    for run in experiment_metrics.runs:
        plot_multi_line_chart(
            data_series=[
                [
                    float(loss)
                    for ep in run.episodes
                    for loss in getattr(ep, "all_critic2_loss", [])
                    if loss is not None
                ]
            ],
            labels=[r.run_id for r in experiment_metrics.runs],
            title="Critic2 Loss per Update",
            image_name=f"all_critic2_loss_{run.run_id}.png",
            save_dir=plots_dir,
            x_label="Training update",  # nie Episode
            y_label="Critic1 Loss",
            y_scale="log",  # odporúčam pre critic loss
            stride=10,  # downsample (zlepší čitateľnosť)
            skip_first=0,  # alebo napr. 1000 len na vizualizáciu
        )
