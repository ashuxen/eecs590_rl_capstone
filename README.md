# EECS 590 Reinforcement Learning Capstone - Final Version

**Author:** Ashutosh Kumar  
**Course:** EECS 590 - Reinforcement Learning, Spring 2026  
**Project:** Impedance-aware robotic cable insertion for the Intrinsic AI for Industry Challenge  
**Final version:** V3, due May 13, 2026

## Project Overview

This repository documents my capstone work on reinforcement learning for robotic cable insertion. The target task is to command a UR5e robot with a Robotiq Hand-E gripper, wrist force-torque sensing, and wrist-mounted cameras to insert SFP and SC connectors into randomized task-board ports. The final deployment environment is Gazebo/ROS 2 with `ground_truth=false`; Isaac Sim/Isaac Lab was used as a faster training and reward-design environment for PPO experiments.

The project has two connected tracks:

- A course RL framework using a finite MDP, dynamic programming, TD(lambda), and a Windy Chasm environment from earlier assignments.
- The capstone robotics system for the AIC cable insertion task, including perception, force-aware contact logic, PPO/residual policy experiments, HMM/recurrent belief tracking, and competition evaluation logs.

## Final V3 Summary

The final version records what was implemented, what was tested, and what was left unfinished. The project uses:

- **PPO** for continuous impedance-aware residual policy learning.
- **DAgger / behavior cloning** as a teacher and warm-start experiment, but not as the final deployment answer.
- **Bayesian belief tracking / HMMs** and **GRU recurrent policy memory** for contact phase estimation under partial observability.
- **Heuristic impedance control** for deployment, because it was more robust than the unfinished RL policy in Gazebo by the deadline.
- **Isaac Sim / Isaac Lab** for high-throughput PPO reward-shaping experiments.
- **Gazebo** for final deployment validation and scoring evidence.

The main final decision document is `docs/docs/final-decisions-v3.md`. Lightweight execution checks and known external requirements are documented in `docs/docs/functional-readiness.md`.

## Repository Map

```text
.
|-- README.md
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
|-- docs/
|   |-- mkdocs.yml
|   `-- docs/
|       |-- index.md
|       |-- final-decisions-v3.md
|       |-- isaac-sim-ppo-training.md
|       |-- gazebo-evaluation.md
|       |-- technical-challenges.md
|       `-- functional-readiness.md
|-- reports/
|   |-- figures/tensorboard/
|   |-- gazebo_eval_logs/
|   `-- isaac_sim_runs/
|-- rl_capstone/
|   |-- mdp/
|   |-- algorithms/
|   |-- agents/
|   |-- environments/
|   |-- visualization/
|   `-- aic_competition/
|       |-- policy/
|       |-- training/
|       |-- phase_estimation/
|       |-- perception/
|       |-- docs/
|       `-- isaac_sim/
|-- models/
|-- notebooks/
|-- references/
`-- scenes/
```

## Functional Readiness

The lightweight Python RL framework can be installed and imported with normal Python tooling. The AIC competition code is included as a record of the capstone implementation, but full policy execution requires the external AIC ROS 2/Gazebo workspace and, for Isaac training, the Windows Isaac Lab installation.

Recommended local checks:

```bash
python -m pip install -r requirements.txt
python -m py_compile rl_capstone/algorithms/*.py rl_capstone/mdp/*.py rl_capstone/agents/*.py rl_capstone/environments/*.py
python -m rl_capstone.environments.windy_chasm
```

Full Gazebo evaluation is run from the AIC workspace, not directly from this repository. Evidence from completed runs is stored in `reports/gazebo_eval_logs/`.

## Evidence Included

- `reports/figures/tensorboard/` contains exported TensorBoard plots from Isaac Lab PPO runs V9-V12.
- `reports/isaac_sim_runs/` contains compact run metadata: `agent.yaml`, `env.yaml`, and `IsaacLab.diff` for selected PPO runs.
- `reports/gazebo_eval_logs/` contains copied scoring YAML files from local Gazebo evaluations, including a best local score of about `98.82`.
- `rl_capstone/aic_competition/images/` contains project visuals for the AIC cable insertion work.

Large generated artifacts are intentionally excluded: raw TensorBoard event files, Isaac checkpoints, Gazebo bags, Docker layers, and full rollout datasets.

## Project Visuals

The images below show the Isaac Sim workcell, SC cable insertion, and the cable insertion setup used while developing the AIC policy.

![Isaac Sim AIC workcell](rl_capstone/aic_competition/images/isaac%20sim%20AIC%204%20bot.png)

![SC cable insertion](rl_capstone/aic_competition/images/cable%20insertion_SC_1.png)

![Cable insertion setup](rl_capstone/aic_competition/images/cable%20insertion.png)

## Results Snapshot

- **Windy Chasm MDP:** value iteration / policy iteration agents solve the tabular environment and save policy summaries under `models/policy_kernel/`.
- **Gazebo heuristic policy:** local perception and impedance policy reached a best recorded score of `98.82 / 300` in `reports/gazebo_eval_logs/aic_results_scoring.yaml`.
- **Isaac PPO:** reward shaping improved substantially from V9-V12. V12 reached high mean reward and nonzero Tier 2 signal, but no robust Tier 3 seating before the capstone deadline.
- **Deployment status:** final scoring still depends on perception quality, port randomization, and blind contact search after the plug reaches the NIC card face.

## Important Design Decisions

PPO was used for continuous residual control, belief tracking was used for partial observability, and Bayesian tuning remains a useful option for impedance parameters. Swarm/coordination methods were not implemented because the task has one robot and one cable under direct control. The final decisions and rejected alternatives are documented in `docs/docs/final-decisions-v3.md`.

## Acknowledgments and Assistance

I thank Dr. Alex Lowenstein, the course instructor, for the guidance and support throughout the semester. I used course materials, public documentation for ROS 2, Gazebo, Isaac Sim/Isaac Lab, PyTorch, simulator logs, and my own experiment notes while preparing the final repository. I also used the free version of ChatGPT LLM for some help and documentation.

## Citations

- Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*.
- Ross, S. et al. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*.
- Johannink, T. et al. (2019). *Residual Reinforcement Learning for Robot Control*.
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models*.
- ROS 2 documentation: https://docs.ros.org/
- Gazebo documentation: https://gazebosim.org/
- PyTorch documentation: https://pytorch.org/
- Intrinsic AI for Industry Challenge materials: https://github.com/intrinsic-dev/aic

