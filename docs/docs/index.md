# EECS 590 RL Capstone Documentation

This documentation is the final V3 record for my reinforcement learning capstone. The project combines a course RL framework with a robotics capstone for impedance-aware cable insertion.

## Read These First

- `final-decisions-v3.md` explains what I implemented, what I chose not to implement, and why.
- `isaac-sim-ppo-training.md` summarizes the Windows Isaac Sim / Isaac Lab PPO training work, including the recurrent GRU belief/memory experiment.
- `gazebo-evaluation.md` summarizes WSL/Gazebo scoring evidence.
- `technical-challenges.md` records failures, debugging lessons, and remaining risks.
- `functional-readiness.md` records the lightweight checks run for V3 and the external simulator requirements.

## Project Areas

- `rl_capstone/mdp/`: finite MDP and belief MDP utilities.
- `rl_capstone/algorithms/`: value iteration, Q-value iteration, policy iteration, and TD(lambda).
- `rl_capstone/agents/`: training/evaluation wrapper for the tabular RL work.
- `rl_capstone/environments/`: Windy Chasm and UR5e environment interfaces.
- `rl_capstone/aic_competition/`: perception, PPO/residual policy, phase belief tracking, and policy code for the cable insertion task.
- `reports/`: compact evidence artifacts from Isaac training and Gazebo evaluation.

## Final Status

The repository is intended to be a reproducible record of the capstone work. The lightweight Python components can be checked locally. Full robot evaluation requires the AIC simulator workspace and, for Isaac PPO training, the external Windows Isaac Lab installation.

