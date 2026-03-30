"""


"""

from typing import List

from td3.metrics.metrics import ExperimentMetrics
from td3.utils.graph_utils import plot_multi_line_chart, plot_episode_weights


def provide_test_graphs(
    plots_dir: str,
    weight_dir: str,
    experiment_metrics: ExperimentMetrics,
    model_names: List[str],
    tickers: List[str],
) -> None:
    """
    Show graphs comparing the performance of different models based on the metrics collected during testing.

    Args:
        plots_dir: Directory where the performance plots will be saved.
        weight_dir: Directory where the weight trajectory plots will be saved.
        experiment_metrics: Metrics object containing the runs and episodes with their respective rewards, portfolio values, and weights.
        model_names: List of model names corresponding to the runs in experiment_metrics.
        tickers: List of tickers corresponding to the weights, including "Cash" as the first entry.
    """

    plot_multi_line_chart(
        data_series=[
            [step.reward for step in run.episodes[:-1]] for run in experiment_metrics.runs
        ],
        labels=model_names,
        title="Cumulative reward",
        image_name="episode_rewards.png",
        save_dir=plots_dir,
        y_label="Reward",
        x_label="Episode",
    )

    plot_multi_line_chart(
        data_series=[
            [step.portfolio_value for step in run.episodes[:-1]] for run in experiment_metrics.runs
        ],
        labels=model_names,
        title="Cumulative portfolio value",
        image_name="portfolio_reward.png",
        save_dir=plots_dir,
        y_label="Portfolio Value",
        x_label="Episode",
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
            title=f"Weights per Episode - {name}",
        )


def provide_train_graphs(plots_dir: str, experiment_metrics: ExperimentMetrics) -> None:
    """
    Provide graphs comparing the performance of different runs based on the metrics collected during training.

    Args:
        plots_dir: Directory where the performance plots will be saved.
        experiment_metrics: ExperimentMetrics object containing the runs and episodes with
            their respective rewards, portfolio values, and losses.
    """

    plot_multi_line_chart(
        data_series=[[ep.total_reward for ep in run.episodes] for run in experiment_metrics.runs],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Cumulative reward over episodes",
        image_name="episode_rewards.png",
        save_dir=plots_dir,
        y_label="Reward",
        x_label="Episode",
    )

    plot_multi_line_chart(
        data_series=[
            [ep.final_portfolio_value for ep in run.episodes] for run in experiment_metrics.runs
        ],
        labels=[r.run_id for r in experiment_metrics.runs],
        title="Cumulative portfolio value over episodes",
        image_name="portfolio_value.png",
        save_dir=plots_dir,
        y_label="Portfolio value",
        x_label="Episode",
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
            title="Critic1 Loss per Training Step",
            image_name=f"all_critic1_loss_{run.run_id}.png",
            save_dir=plots_dir,
            y_label="Critic Loss",
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
            title="Critic2 Loss per Training Step",
            image_name=f"all_critic2_loss_{run.run_id}.png",
            save_dir=plots_dir,
            y_label="Critic Loss",
            y_scale="log",
            stride=10,
            skip_first=0,
        )
