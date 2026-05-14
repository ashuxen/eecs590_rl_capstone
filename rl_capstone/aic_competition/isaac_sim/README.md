# Isaac Sim / Isaac Lab Integration Notes

The active Isaac Sim / Isaac Lab training workspace was maintained outside this repository on Windows:

```text
C:\IsaacLab\aic\aic_isaaclab
```

This repository records the capstone-level design and evidence, while the full Isaac workspace remains an external simulator project.

## What Was Trained

The main Isaac work was PPO for SFP insertion in the AIC v1.5 task. The policy experiments focused on:

- pure PPO from scratch after behavior cloning transfer failed,
- reward terms for approach, alignment, insertion depth, and seated pose,
- force/torque observations for contact-rich insertion,
- recurrent GRU policy memory for partial observability during contact,
- randomized port/NIC positions,
- deploy-safe actor observations when planning future contact-start RL.

## Why It Is Not Fully Vendored Here

The Isaac workspace contains simulator-specific generated files, logs, checkpoints, and dependencies. Copying the full workspace would make this repository hard to review. Instead, the final repo includes:

- documentation in `docs/docs/isaac-sim-ppo-training.md`,
- TensorBoard plot exports in `reports/figures/tensorboard/`,
- compact run metadata and diffs in `reports/isaac_sim_runs/`.

