# Isaac Sim Run Evidence

This directory contains compact evidence from selected Isaac Lab PPO experiments. The raw Windows training directory was:

```text
C:\IsaacLab\logs\rsl_rl\aic_task
```

Only lightweight files are committed here:

- `agent.yaml`: PPO/RSL-RL agent configuration.
- `env.yaml`: Isaac Lab task/environment configuration.
- `IsaacLab.diff`: source diff captured by the Isaac Lab logger for that run.
- `summary.json`: exported scalar summary from selected TensorBoard tags.

The raw event files, policy checkpoints, optimizer state, and rollout buffers are intentionally excluded because they are generated artifacts and can be large.

The V9-V12 sequence represents the final capstone PPO reward-shaping effort:

- V9: pure PPO from scratch.
- V10: removed lateral-excursion reward exploit.
- V11: stronger far-field approach shaping.
- V12: close-range insertion shaping with seated pose tracking and soft depth pull.

These runs also used recurrent policy memory. The exported `agent.yaml` files record `ActorCriticRecurrent` with a one-layer GRU (`rnn_hidden_dim=128`). This was the Isaac-side LSTM/GRU-style belief experiment for partial observability during contact.

