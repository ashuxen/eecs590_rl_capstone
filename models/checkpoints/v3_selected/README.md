# V3 Selected Model Checkpoints

This folder contains a small set of checkpoint artifacts selected for the V3 submission. I did not commit every training snapshot because the full Isaac and Gazebo training folders contain many generated files.

## Included Checkpoints

- `isaac_ppo_v12_closeft/model_2150_best_reward.pt`: Isaac Lab recurrent PPO checkpoint from the V12 close-insertion run, selected because the V12 summary reported its best mean reward near step 2147.
- `isaac_ppo_v12_closeft/model_2575_final.pt`: final available checkpoint from the same V12 run.
- `bc_v15/model_best.pt`: best behavior-cloning warm-start checkpoint from the V15 data experiment.
- `dagger_gru/gru_residual_policy.pt`: recurrent GRU residual policy checkpoint from the DAgger/learned-memory experiment.
- `hmm_phase/phase_hmm_params.npz`: Gaussian HMM phase-belief parameters used for contact phase estimation.
- `residual_policy/residual.pt`: residual policy checkpoint from the Gazebo-side residual policy experiment.
- `port_2d_v5_entry_fine/port_2d.pt`: main 2D port perception checkpoint, with camera geometry and label-map files needed to interpret it.

## Source Locations

Original local sources:

- Isaac PPO V12: `/mnt/c/IsaacLab/logs/rsl_rl/aic_task/2026-05-13_17-48-41_v15_ppo_sfp_v12_closeft/`
- BC V15: `/home/ashutokumar/rl/bc_v15_ckpt/`
- DAgger GRU: `/home/ashutokumar/rl/dagger_gru_checkpoint/`
- HMM phase belief: `/home/ashutokumar/rl/yd_rrl_checkpoints/`
- Gazebo residual policy and perception checkpoint: `/home/ashutokumar/rl/perception_checkpoints/`

These checkpoints are included as evidence of the training work and are not guaranteed to run without the matching simulator/workspace dependencies documented elsewhere in the repository.

