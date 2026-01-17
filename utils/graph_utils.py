import os
import numpy as np
import matplotlib.pyplot as plt

def plot_episode_weights(all_weights, tickers, episode, save_dir="plots"):
    """
    Vykreslí a uloží stacked bar graf portfóliových váh pre jednu epizódu.

    Parameters:
    - all_weights: list of np.array, obsahuje váhy pre každý krok epizódy
    - tickers: list of str, názvy aktív
    - episode: int, číslo epizódy
    - save_dir: str, priečinok na uloženie grafov
    """
    if len(all_weights) == 0:
        return

    W = np.array(all_weights)   # (T, N)
    T, N = W.shape
    x = np.arange(T)

    # Normalizácia
    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(T)

    for i in range(N):
        plt.bar(x, W[:, i], bottom=bottom, label=tickers[i])
        bottom += W[:, i]

    plt.title(f"Portfolio Weights Over Time - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Weight")
    plt.ylim(0, 1)
    plt.legend(title="Assets", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"episode_{episode}_weights.png")
    plt.savefig(filepath)
    # plt.show()
    plt.close()
    return filepath