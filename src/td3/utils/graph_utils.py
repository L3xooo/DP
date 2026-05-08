from typing import List

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_price_history(
    data: np.ndarray,
    ticker_labels: list[str],
    dates: list,
    title: str = "Price History",
    save_path: str | None = None,
) -> None:
    """
    Plot price history for multiple tickers over time.

    Args:
        data (np.ndarray): Price data of shape (timesteps, n_tickers, 1).
        ticker_labels (list[str]): List of ticker names, length must match n_tickers.
        dates (list): List of dates/timestamps, length must match timesteps.
        title (str): Chart title. Defaults to "Price History".
        save_path (str | None): If provided, saves the chart to this path.
    """
    assert data.ndim == 3 and data.shape[2] == 1, "Expected shape (timesteps, n_tickers, 1)"
    assert len(ticker_labels) == data.shape[1], "ticker_labels length must match n_tickers"
    assert len(dates) == data.shape[0], "dates length must match timesteps"

    prices = data.squeeze(-1)  # (685, 10)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, ticker in enumerate(ticker_labels):
        ax.plot(dates, prices[:, i], linewidth=1.5, label=ticker)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, ncol=2, framealpha=0.7)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_multi_line_chart(
    data_series,
    labels,
    title,
    image_name="multi_line_chart.png",
    save_dir=None,
    x_label="Step",
    y_label="Value",
    dpi=200,
    y_scale=None,
    percentile_clip=None,
    skip_first=0,
    stride=1,
):
    plt.figure(figsize=(10, 5))

    for data, label in zip(data_series, labels):
        # convert + basic cleaning
        y = np.asarray(data, dtype=float)

        if skip_first > 0:
            y = y[skip_first:]

        if stride > 1:
            y = y[::stride]

        x = np.arange(len(y))
        plt.plot(x, y, label=label)

    if percentile_clip is not None:
        lo_p, hi_p = percentile_clip
        all_vals = []
        for data in data_series:
            yy = np.asarray(data, dtype=float)
            if skip_first > 0:
                yy = yy[skip_first:]
            if stride > 1:
                yy = yy[::stride]
            yy = yy[np.isfinite(yy)]
            if yy.size:
                all_vals.append(yy)
        if all_vals:
            all_vals = np.concatenate(all_vals)
            lo = np.percentile(all_vals, lo_p)
            hi = np.percentile(all_vals, hi_p)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                plt.ylim(lo, hi)

    if y_scale == "log":
        plt.yscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)
        plt.savefig(save_path, dpi=dpi)

    plt.close()


def plot_episode_weights(
    all_weights: List[List[float]],
    tickers: List[str],
    episode: int,
    save_dir: str = "plots",
    filename: str | None = None,
    title: str | None = None,
    dpi: int = 200,
) -> str | None:
    if len(all_weights) == 0:
        return None

    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"save_dir does not exist: {save_dir}")

    if filename is None:
        filename = f"episode_{episode}_weights.jpg"

    if title is None:
        title = f"Portfolio Weights Over Time - Episode {episode}"

    save_path = os.path.join(save_dir, filename)

    W = np.asarray(all_weights, dtype=float)
    T, N = W.shape
    x = np.arange(T)

    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(T)

    for i in range(N):
        plt.bar(x, W[:, i], bottom=bottom, label=tickers[i])
        bottom += W[:, i]

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Weight")
    plt.ylim(0, 1)
    plt.legend(title="Assets", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, format="jpg")
    plt.close()

    return save_path
