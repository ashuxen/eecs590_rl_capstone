# Functional Readiness

This repository has two different execution levels:

- The lightweight course RL framework can be checked locally with Python.
- The full AIC robot stack requires the external ROS 2/Gazebo workspace and, for Isaac training, the external Windows Isaac Lab workspace.

## Checks Run for V3

The following checks were run from the repository root:

```bash
python3 -m py_compile rl_capstone/algorithms/*.py rl_capstone/mdp/*.py rl_capstone/agents/*.py rl_capstone/environments/*.py
python3 -m rl_capstone.environments.windy_chasm --episodes 2
python3 -m py_compile rl_capstone/aic_competition/perception/train_port_2d_detector.py rl_capstone/aic_competition/perception/train_port_2d_v4.py rl_capstone/aic_competition/phase_estimation/fit_phase_hmm.py rl_capstone/aic_competition/training/dagger_loop.py rl_capstone/aic_competition/training/relabel_rewards.py rl_capstone/aic_competition/training/train_ppo_gazebo.py rl_capstone/aic_competition/training/train_yd_rrl.py
```

Results:

- Core Python compilation completed successfully.
- The Windy Chasm demo ran successfully. With `B=0.5`, `gamma=0.99`, and 2 sample episodes, value iteration converged in 2591 iterations and both sampled episodes crashed. That is a stochastic rollout result, not an interpreter failure.
- AIC perception, phase-estimation, and training scripts compiled successfully.

The Windy Chasm module emitted a standard Python runtime warning about the module already being present in `sys.modules` before execution. The command still exited successfully.

## Known External Requirements

The following cannot be fully executed from this repo alone:

- Gazebo policy evaluation requires the AIC ROS 2 workspace and simulator entrypoint.
- Isaac PPO training requires Isaac Sim / Isaac Lab on the Windows environment used for training.
- Full camera/perception training requires local datasets that are not committed.
- Full model replay requires large checkpoints and TensorBoard event files that are intentionally excluded from Git.

