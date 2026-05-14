# TensorBoard Evidence

These plots were exported from Isaac Lab TensorBoard event files stored outside this repo at `/mnt/c/IsaacLab/logs/rsl_rl/aic_task`. They summarize the V9-V12 SFP PPO reward-shaping runs. Raw event files and checkpoints are intentionally not committed because they are large generated artifacts.

- `isaac_v9_v12_mean_reward.png`: overall PPO reward progression.
- `isaac_v9_v12_insertion_depth.png`: insertion-depth metric progression.
- `isaac_v9_v12_tier2.png`: intermediate insertion reward signal.
- `isaac_v9_v12_tier3.png`: strict seating reward signal.
- `isaac_v9_v12_orientation_error.png`: orientation error metric.

The plots are included as static evidence for the V3 repository review. The scalar values used to produce them are summarized in `reports/isaac_sim_runs/summary.json`.
