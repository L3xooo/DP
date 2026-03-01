# DP

## Training

Single-environment training:

```bash
python main_train.py
```

Parallel environment training (vectorized envs):

```bash
python main_train_parallel.py
```

Notes:
- `main_train_parallel.py` runs `num_envs` environments in parallel; each outer episode produces `num_envs` episode metrics.
- Adjust `num_envs` inside `main_train_parallel.py` based on your CPU capacity.
