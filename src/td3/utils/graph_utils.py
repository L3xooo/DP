import numpy as np
import os
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_multi_line_chart(
    data_series,
    labels,
    title,
    image_name="multi_line_chart.png",
    save_dir=None,
    x_label="Step",  # default zmenené (keď je to per-update)
    y_label="Value",
    dpi=200,
    y_scale=None,  # None | "log"
    percentile_clip=None,  # napr. (1, 99)
    skip_first=0,  # napr. 1000 ak chceš odseknúť warmup v grafe
    stride=1,  # downsample (napr. 10)
):
    plt.figure(figsize=(10, 5))

    for data, label in zip(data_series, labels):
        # convert + basic cleaning
        y = np.asarray(data, dtype=float)

        # skip warmup only for plotting
        if skip_first > 0:
            y = y[skip_first:]

        # downsample for readability
        if stride > 1:
            y = y[::stride]

        # x axis as index
        x = np.arange(len(y))

        plt.plot(x, y, label=label)

    # percentile zoom (based on ALL plotted values)
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

    # y-scale
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

    plt.show()
    plt.close()


def plot_line_chart(
    data,
    label,
    title,
    image_name,
    save_dir=None,
    x_label="Episode",
    y_label="Value",
    dpi=200,
):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)
        plt.savefig(save_path, dpi=dpi)

    plt.show()
    plt.close()


def plot_episode_weights(
    all_weights,
    tickers,
    episode,
    save_dir="plots",
    filename=None,
    title=None,
    dpi=200,
):
    # print(all_weights)
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
