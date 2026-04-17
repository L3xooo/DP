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
| `number_of_episodes` | Episodes per iteration |
| `batch_size` | Minibatch size for gradient updates |
| `learning_start_episode` | Episode at which learning begins |
| `start_date` / `end_date` | Date range for the trading environment |
| `initial_cash` | Starting portfolio cash balance |
| `hidden_size` | Actor and critic network hidden layer size |
| `lr` | Learning rate |
| `noise_init` / `noise_final` | Exploration noise schedule |
| `filter_in` | Technical indicators included in the state |

---

### Ticker Config — `src/td3/config/ticker_config.py`

Controls which group of stocks the agent trades. Available presets:

| Name | Description |
|---|---|
| `10_TICKERS` | AAPL, MSFT, AMZN, GOOGL, META, TSLA, NVDA, JPM, JNJ, XOM |
| `ANOTHER_10_TICKERS` | BRK.B, UNH, V, MA, AVGO, LLY, JPM, XOM, COST, HD |
| `30_TICKERS` | Diversified 30-stock universe across sectors |

To switch preset, update `ticker_config` in `AppConfig`:

```python
ticker_config: TickerConfig = TickerConfig("30_TICKERS")
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

---

## Training

Once configuration is set, start a training run:

```bash
python main_train.py
```

After training completes, the following are saved to the experiment output directory:

- **Training plots** — reward curves and portfolio value over episodes
- **Saved models** — actor and critic network checkpoints
- **Weights** — portfolio allocation weight history

---

## Testing

> 🚧 Testing pipeline is under development.