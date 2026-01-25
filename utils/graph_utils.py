import numpy as np
import os
import matplotlib.pyplot as plt

def plot_line_chart(
    data,
    label,
    title,
    image_name,
    save_dir=None,
    xlabel="Episode",
    ylabel="Value",
    dpi=200,
):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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

    print(all_weights)
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
    plt.show()
    # plt.savefig(save_path, dpi=dpi, format="jpg")
    plt.close()

    return save_path