# Isaac Sim / Isaac Lab PPO Training Record

Most Isaac Sim work was performed on Windows because that is where Isaac Lab was installed. This repository stores a compact record of the experiments rather than raw checkpoints and event files.

## External Location

Raw Isaac Lab training artifacts were kept outside the repository:

```text
C:\IsaacLab\logs\rsl_rl\aic_task
```

From WSL, that path is visible as:

```text
/mnt/c/IsaacLab/logs/rsl_rl/aic_task
```

The repository includes exported summaries and plots under:

```text
reports/isaac_sim_runs/
reports/figures/tensorboard/
```

## Why Isaac Sim Was Used

Gazebo was the final evaluation environment, but it was too slow for large-scale PPO iteration. Isaac Sim / Isaac Lab was used for reward design and contact policy experiments. Gazebo remained the validation environment because it matched the competition runtime.

## Recurrent Belief / Memory Experiment

The Isaac PPO experiments also tested recurrent policy memory as a learned belief-state approximation. The exported V9-V12 configs use:

```text
class_name: ActorCriticRecurrent
rnn_type: gru
rnn_hidden_dim: 128
rnn_num_layers: 1
```

I treated this as the neural version of the LSTM/GRU belief idea: recent force, pose, and action history should help the policy infer whether it is still approaching, sliding on the cage face, entering the port, or seated. The final code also includes an explicit Gaussian HMM phase estimator, which is easier to inspect and tune than the recurrent hidden state.

## PPO Run Progression

### V8: BC + PPO Warm Start

The first serious attempt used behavior cloning and DAgger data from Gazebo as a warm start for PPO. This did not work well because the initial pose distribution in Isaac did not match the Gazebo demonstrations closely enough. The learned policy failed before it could use the demonstration prior effectively.

Decision: stop treating BC as the main capstone path and move to pure PPO reward shaping.

### V9: PPO From Scratch

V9 removed the BC warm start and started PPO from scratch. The agent exploited lateral-excursion termination and did not learn useful seating behavior.

Evidence path:

```text
reports/isaac_sim_runs/2026-05-13_14-18-50_v15_ppo_sfp_v9_scratch/
```

### V10: Disable Lateral-Excursion Exploit

V10 removed the easy termination hack by disabling the lateral-excursion termination path. This avoided one failure mode but revealed a hover/stall behavior: the policy still did not learn strong enough approach and insertion progress.

Evidence path:

```text
reports/isaac_sim_runs/2026-05-13_14-36-21_v15_ppo_sfp_v10_nolateral/
```

### V11: Stronger Approach Shaping

V11 introduced a stronger pre-entrance depth-progress reward. This improved approach behavior and mean reward, but it still did not reliably reach strict seating.

Evidence path:

```text
reports/isaac_sim_runs/2026-05-13_15-28-51_v15_ppo_sfp_v11_strong/
```

### V12: Close-Range Seating Shaping

V12 resumed from the V11 direction and added close-range reward terms:

- seated pose tracking,
- soft gated depth pull,
- tighter lateral and alignment rewards,
- reduced exploration noise.

V12 showed the strongest improvement in mean reward and got nonzero Tier 2 signal, but did not produce robust Tier 3 seating before the capstone deadline.

Evidence path:

```text
reports/isaac_sim_runs/2026-05-13_17-48-41_v15_ppo_sfp_v12_closeft/
```

## TensorBoard Figures

The following exported figures are included:

- `reports/figures/tensorboard/isaac_v9_v12_mean_reward.png`
- `reports/figures/tensorboard/isaac_v9_v12_insertion_depth.png`
- `reports/figures/tensorboard/isaac_v9_v12_tier2.png`
- `reports/figures/tensorboard/isaac_v9_v12_tier3.png`
- `reports/figures/tensorboard/isaac_v9_v12_orientation_error.png`

The scalar summary is stored in:

```text
reports/isaac_sim_runs/summary.json
```

Important scalar takeaways:

- V11 improved mean reward into a positive range.
- V12 reached best mean reward around 676 and last mean reward around 632.
- V12 reached nonzero Tier 2 reward signal.
- No V9-V12 run achieved reliable Tier 3 seating in Isaac before the deadline.

## Why Raw Checkpoints Are Not Committed

Raw checkpoints, TensorBoard event files, rollout buffers, and simulation logs are generated artifacts. They are large, machine-specific, and not necessary for a reader to understand the algorithmic choices. This repo commits:

- exported training plots,
- environment and agent YAML files,
- source diffs for the Isaac Lab changes,
- a scalar summary JSON.

This keeps the repo usable while preserving evidence of the training work.

