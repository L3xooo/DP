# TD3 — Twin Delayed Deep Deterministic Policy Gradient

A TD3-based portfolio allocation agent trained on technical indicator data.

---

## Installation

**Requirements:** Python 3.12

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

### App Config — `src/td3/config/app_config.py`

The central configuration for a training run. Key parameters you may want to adjust:

| Parameter | Description |
|---|---|
| `iterations` | Number of full training iterations |
| `number_of_episodes` | Number of episodes per iteration |
| `batch_size` | Minibatch size for gradient updates |
| `replay_buffer_size` | Maximum capacity of the experience replay buffer |
| `ticker_config` | Ticker configuration preset |
| `data_dir` | Path to the directory with precomputed indicator data |
| `start_date` / `end_date` | Date range for the trading environment |
| `initial_cash` | Starting portfolio cash balance |
| `hidden_size` | Hidden layer size for actor and critic networks |
| `lr` | Learning rate |
| `noise_init` / `noise_final` | Initial and final exploration noise (linear decay) |
| `dropout_rate` | Dropout rate applied in actor and critic networks |
| `normalization` | Normalization type applied in networks (`layer`, `batch`, or `None`) |
| `device` | Compute device (`cuda`, `cpu`, or `None` for auto-detect) |
| `filter_out` | Features excluded from the state observation |
| `filter_in` | Technical indicators included in the state observation |
---

### Ticker Config — `src/td3/config/ticker_config.py`

Controls which group of stocks the agent trades. Available presets:

| Name             | Description                                              |
|------------------|----------------------------------------------------------|
| `10_TICKERS`     | AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, JNJ, XOM |
| `RANDOM_TICKERS` | Randomly selected 10 tickers.                            |

To switch preset, update `ticker_config` in `AppConfig`:

```python
ticker_config: TickerConfig = TickerConfig("10_TICKERS")
```

---

### Data Directory — `data_dir`

Set `data_dir` to the root directory containing per-ticker subdirectories.

Expected structure:

```
data_dir/
├── AAPL/
│   └── normalized.csv
├── MSFT/
│   └── normalized.csv
└── ...
```

Each `normalized.csv` should contain a `date` column and pre-normalized technical indicator columns. The `start_date` and `end_date` parameters in `AppConfig` can be adjusted to match the date range available in your data.
See the example of `normalized.csv` in the `examples` directory of this repository.
---

## Training

Once configuration is set, start a training run:

```bash
python main_train.py
```

After training completes, the following are saved to the experiment output directory:

- **Training plots** — Different plots that visualize the training process.
- **Saved models** — actor and critic network checkpoints
- **Weights** — portfolio allocation weight history

---

## Testing

```bash
python main_test.py
```

Before running the tests, two configuration steps are required:

1. **Set the experiment directory** — update `EXPERIMENT_DIR` in `main_test.py` so the test execution picks up models from the correct directory.
2. **Configure the date range** — set `start_date` and `end_date` at the beginning of the `main` function in `main_test.py`. Ensure the test start date is not earlier than the training end date.

After execution, the following outputs are generated in the `simulations` directory under the corresponding run folder:

- **Plots** — visual results of the test run
- **Weights** — a CSV file containing the portfolio weights, located in the `weights` subdirectory
- **Diversification** — a chart showing diversification over the episode