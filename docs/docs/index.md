# EECS 590 RL Capstone Documentation

## Description

EECS 590 Reinforcement Learning Capstone - Sim-to-Real Robotic Manipulation for the [AI for Industry Challenge](https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380).

## V1 — Foundation RL

- Windy Chasm MDP environment with Value Iteration, Policy Iteration, TD(λ)
- Agent framework with train/evaluate/save
- Isaac Sim visualization scenes
- See `rl_capstone/mdp/`, `rl_capstone/algorithms/`, `rl_capstone/agents/`

## V2 — AIC Competition (Deep RL)

All competition code lives in [`rl_capstone/aic_competition/`](../../rl_capstone/aic_competition/).

| Module | What It Does |
|--------|-------------|
| [`policy/SmartInsert.py`](../../rl_capstone/aic_competition/policy/SmartInsert.py) | Main insertion policy — PPO + DAgger hybrid with impedance control (2800+ lines) |
| [`perception/train_port_2d_v4.py`](../../rl_capstone/aic_competition/perception/train_port_2d_v4.py) | CNN training — ResNet-18 + FPN + FiLM for port detection |
| [`training/ppo_residual.py`](../../rl_capstone/aic_competition/training/ppo_residual.py) | PPO actor-critic with DAgger fallback |
| [`training/dagger_loop.py`](../../rl_capstone/aic_competition/training/dagger_loop.py) | DAgger expert data collection |
| [`training/residual_mlp.py`](../../rl_capstone/aic_competition/training/residual_mlp.py) | Shared MLP architecture for residual policy |
| [`phase_estimation/phase_belief.py`](../../rl_capstone/aic_competition/phase_estimation/phase_belief.py) | Gaussian HMM for Bayesian contact-phase tracking |
| [`phase_estimation/fit_phase_hmm.py`](../../rl_capstone/aic_competition/phase_estimation/fit_phase_hmm.py) | Fit HMM parameters from collected episodes |

## Model Checkpoints

Trained weights are in `models/checkpoints/`:

- `perception/port_2d_v4.pt` — CNN port detector (ResNet-18 + FPN)
- `ppo/ppo_actor.pt`, `ppo_critic.pt`, `residual_mlp.pt` — PPO policy
- `dagger/residual_mlp.pt` — DAgger expert baseline
- `hmm/phase_hmm_params.npz` — Fitted HMM parameters

## Training Images

Sample camera views from data collection are in [`rl_capstone/aic_competition/images/`](../../rl_capstone/aic_competition/images/) — showing SFP and SC insertion phases (start, approach, near, descent, hold).

## Other Docs

- [Getting Started](getting-started.md)
- [Technical Challenges](technical-challenges.md) — bugs, surprises, and lessons learned
- [Environment & Hardware](../../rl_capstone/aic_competition/docs/environment.md) — competition setup, observation/action space
- [Competition Status](../../rl_capstone/aic_competition/docs/status.md) — progress, scores, known issues
