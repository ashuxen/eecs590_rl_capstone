# EECS 590 Reinforcement Learning - Capstone Project V1

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**Author:** Ashutosh Kumar  
**Course:** EECS 590 - Reinforcement Learning, Spring 2026  
**Version:** V1  
**Due Date:** February 11, 2026

---

## V1 Requirements Compliance

| # | Requirement | Status | Location |
|---|-------------|--------|----------|
| 1 | GitHub Repository Formatting (Cookiecutter) | ✅ | Project structure |
| 2 | Documentation/Organization | ✅ | This README, `requirements.txt` |
| 3 | MDP Representation (rewards/states/beliefs) | ✅ | `rl_capstone/mdp/` |
| 4 | Dynamic Programming Implementation | ✅ | `rl_capstone/algorithms/` |
| 5 | Agent Framework (train/evaluate) | ✅ | `rl_capstone/agents/` |

---

## Project Overview

This capstone project explores **Sim-to-Real Reinforcement Learning for Robotic Manipulation**, aligned with the course's "Underexplored Frontiers" (Slide 9: Adaptable Sim-to-Real). The project builds toward the **Intrinsic AI for Industry Challenge**.

### Foundation Environment: Windy Chasm (Mini 2)

A discrete MDP where a drone navigates through a windy chasm:
- **State Space S**: Grid (i,j) ∈ [0,19]×[0,6] + terminal states {crash, goal}
- **Action Space A**: {Forward, Left, Right}
- **Transition P(s'|s,a)**: Deterministic action + stochastic wind p(j) = B^(1/(1+(j-3)²))
- **Reward R**: step=-1, goal=+20, crash=-5
- **Discount γ**: 0.99

### Capstone Environment: UR5e Cable Insertion (AI for Industry Challenge)

A continuous robotic manipulation task for the **Intrinsic AI for Industry Challenge**:
- **Robot**: Universal Robots UR5e (6-DOF arm)
- **Gripper**: Robotiq Hand-E (parallel jaw)
- **Sensor**: ATI Axia80 Force-Torque sensor
- **Vision**: Three wrist-mounted Basler cameras
- **Task**: Cable insertion into server rack ports
- **Challenge**: [AI for Industry Challenge](https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380)

> **Note**: This is a preliminary environment setup. The official participant toolkit will be available March 2, 2026.

---

## Project Organization

```
├── LICENSE
├── Makefile           <- Commands: make train, make evaluate, make demo
├── README.md          <- This file
├── pyproject.toml     <- Project metadata and tool configuration
├── requirements.txt   <- Python dependencies
│
├── models/            <- [REQ 5] Trained models and policy kernels
│   └── policy_kernel/ <- Best policy + value function pairs
│       ├── windy_chasm_B0.3.pkl    <- Trained agent (B=0.3)
│       └── windy_chasm_B0.3_summary.json
│
├── notebooks/         <- Jupyter notebooks for experiments
│   ├── 01_mini1_mrp_analysis.ipynb
│   ├── 02_mini2_mdp_analysis.ipynb
│   └── 03_v1_experiments.ipynb
│
├── scenes/            <- Isaac Sim USD scenes
│   ├── ur5e_cable_insertion_scene.usd  <- UR5e robot setup
│   └── windy_chasm_scene.usd           <- Windy Chasm visualization
│
├── references/        <- Course materials, papers
│
├── reports/           <- Generated analysis and figures
│   └── figures/
│
├── train_and_save.py       <- Script to train and save agent
├── load_and_use_agent.py   <- Script to load and use trained agent
├── load_ur5e_scene.py      <- Script to load UR5e Isaac Sim scene
│
└── rl_capstone/       <- Source code
    ├── __init__.py
    ├── config.py
    │
    ├── mdp/           <- [REQ 3] MDP representation with beliefs
    │   ├── __init__.py
    │   ├── finite_mrp.py      <- Markov Reward Process
    │   ├── finite_mdp.py      <- Markov Decision Process
    │   └── belief_mdp.py      <- Model-based beliefs (learnable P, R)
    │
    ├── algorithms/    <- [REQ 4] Dynamic Programming implementations
    │   ├── __init__.py
    │   ├── value_iteration.py      <- V-value iteration
    │   ├── q_value_iteration.py    <- Q-value iteration
    │   ├── policy_iteration.py     <- Policy iteration + improvement
    │   └── td_lambda.py            <- TD(λ) alternative
    │
    ├── agents/        <- [REQ 5] Agent framework
    │   ├── __init__.py
    │   ├── base_agent.py      <- Abstract agent interface
    │   ├── dp_agent.py        <- DP-based agent
    │   └── trainer.py         <- Train/evaluate/save framework
    │
    ├── environments/  <- Custom environments
    │   ├── __init__.py
    │   ├── windy_chasm.py           <- Mini 2 environment
    │   └── ur5e_cable_insertion.py  <- AI for Industry Challenge env
    │
    └── visualization/ <- Isaac Sim visualization
        └── windy_chasm_interactive.py
```

---

## Installation & Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA Isaac Sim 4.5+ (for visualization only)

### Installation

```bash
# Clone repository
git clone https://github.com/ashuxen/eecs590_rl_capstone.git
cd eecs590_rl_capstone

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run Demo

```bash
# Quick demo - solve Windy Chasm and simulate episodes
python -m rl_capstone.environments.windy_chasm --B 0.5 --gamma 0.99 --episodes 10

# Or use Makefile
make demo
```

### Train and Evaluate

```bash
# Train using Value Iteration
python -m rl_capstone.agents.trainer --method value_iteration --save models/policy_kernel/v1.pkl

# Evaluate trained policy
python -m rl_capstone.agents.trainer --evaluate --load models/policy_kernel/v1.pkl --episodes 100
```

### Isaac Sim Visualization

```powershell
# Windy Chasm Interactive Demo
C:\isaacsim\IsaacLab\_isaac_sim\python.bat rl_capstone\visualization\windy_chasm_interactive.py

# Load saved Windy Chasm scene
C:\isaacsim\IsaacLab\_isaac_sim\python.bat load_ur5e_scene.py
```

---

## Isaac Sim Scenes

Pre-built USD scenes are available in the `scenes/` folder:

| Scene | Description | File |
|-------|-------------|------|
| Windy Chasm | 3D visualization of Mini 2 MDP with wind indicators | `windy_chasm_scene.usd` |
| UR5e Cable Insertion | Preliminary setup for AI for Industry Challenge | `ur5e_cable_insertion_scene.usd` |

### Loading Scenes

```powershell
# Load UR5e scene
C:\isaacsim\IsaacLab\_isaac_sim\python.bat load_ur5e_scene.py
```

Or manually in Isaac Sim: **File > Open > scenes/ur5e_cable_insertion_scene.usd**

---

## AI for Industry Challenge (Capstone)

### Challenge Overview

The [Intrinsic AI for Industry Challenge](https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380) focuses on robotic cable manipulation tasks in industrial settings.

### Hardware Stack

| Component | Model | Specifications |
|-----------|-------|----------------|
| **Robot Arm** | Universal Robots UR5e | 6-DOF, 5kg payload, ±0.03mm repeatability |
| **Gripper** | Robotiq Hand-E | Parallel jaw, 50mm stroke, 130N grip force |
| **F/T Sensor** | ATI Axia80 | 6-axis, 80mm OD, EtherCAT interface |
| **Cameras** | Basler (x3) | Wrist-mounted, stereo vision |

### Preliminary Environment

The `ur5e_cable_insertion.py` provides a Gymnasium-style interface:

```python
from rl_capstone.environments import UR5eCableInsertionEnv

# Create environment
env = UR5eCableInsertionEnv(control_mode="joint_velocity")

# Reset and step
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Observation Space (Preliminary)
- Joint positions (6D)
- Joint velocities (6D)
- End-effector pose (7D - position + quaternion)
- Force-torque readings (6D)
- Gripper state (1D)

### Action Space (Preliminary)
- Joint velocities (6D) or
- End-effector delta pose (6D)

> **Timeline**: Official toolkit available March 2, 2026. Current implementation is for preliminary testing.

---

## Requirement 3: MDP Representation

### Belief MDP (Model-Based RL Foundation)

The `BeliefMDP` class allows agents to maintain and update beliefs about transition dynamics and rewards - essential for model-based RL:

```python
from rl_capstone.mdp import BeliefMDP

# Create MDP with learnable beliefs
belief_mdp = BeliefMDP(n_states=142, n_actions=3, gamma=0.99)

# Agent observes transition and updates beliefs
belief_mdp.update_beliefs(state=42, action=0, reward=-1.0, next_state=63)

# Query current beliefs
P_believed = belief_mdp.get_believed_transition(state=42, action=0)
R_believed = belief_mdp.get_believed_reward(state=42, action=0)
```

---

## Requirement 4: Dynamic Programming

### Implemented Algorithms

| Algorithm | Values V | Q-Values Q | Code |
|-----------|----------|------------|------|
| Value Iteration | ✅ | ✅ | `value_iteration.py`, `q_value_iteration.py` |
| Policy Iteration | ✅ | ✅ | `policy_iteration.py` |
| Policy Improvement | ✅ | ✅ | Greedy extraction |
| TD(λ) | ✅ | - | `td_lambda.py` |

### Key Equations

**Value Iteration (V):**
```
V_{k+1}(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V_k(s')]
```

**Q-Value Iteration:**
```
Q_{k+1}(s,a) = R(s,a) + γ Σ P(s'|s,a) max_{a'} Q_k(s',a')
```

**Policy Improvement:**
```
π'(s) = argmax_a Q(s,a)
```

**TD(λ) Update:**
```
V(s) ← V(s) + α [G_t^λ - V(s)]
where G_t^λ = (1-λ) Σ λ^(n-1) G_t^(n)
```

---

## Requirement 5: Agent Framework

### Usage Example

```python
from rl_capstone.agents import DPAgent, Trainer
from rl_capstone.environments import WindyChasmEnv

# Create environment
env = WindyChasmEnv(B=0.5, gamma=0.99)

# Create agent
agent = DPAgent(env, method="value_iteration")

# Train
trainer = Trainer(agent, env)
results = trainer.train()
print(f"V*(start) = {results['v_start']:.4f}")

# Evaluate
eval_results = trainer.evaluate(num_episodes=100)
print(f"Success rate: {eval_results['success_rate']:.1%}")

# Save best policy + value function
trainer.save("models/policy_kernel/best_v1.pkl")

# Load and continue
trainer.load("models/policy_kernel/best_v1.pkl")
```

---

## Results Summary

### V*(0,3) vs Wind Parameter B

| B | V*(0,3) | Success Rate | Convergence |
|---|---------|--------------|-------------|
| 0.3 | 12.38 | 95% | 847 iters |
| 0.5 | 7.82 | 78% | 923 iters |
| 0.7 | 3.21 | 52% | 1034 iters |

### Policy Behavior
- **Lower B** (weak wind): Aggressive forward-moving policy
- **Higher B** (strong wind): Conservative centering policy

---

## What I Built in V1

### Core RL Framework
1. ✅ **Cookiecutter project structure** following data science best practices
2. ✅ **MDP Framework** with belief updating for model-based RL
3. ✅ **Value Iteration** on both V and Q values
4. ✅ **Policy Iteration** with policy evaluation and improvement
5. ✅ **TD(λ)** as alternative to DP methods
6. ✅ **Agent Framework** with train/evaluate/save capabilities
7. ✅ **Trained Agent** saved at `models/policy_kernel/windy_chasm_B0.3.pkl`

### Isaac Sim Integration
8. ✅ **Windy Chasm Visualization** - Interactive 3D scene with UI controls
9. ✅ **UR5e Cable Insertion Scene** - Preliminary setup for AI for Industry Challenge
10. ✅ **USD Scene Files** - Reusable scenes in `scenes/` folder
11. ✅ **Scene Loading Scripts** - `load_ur5e_scene.py` for testing

### AI for Industry Challenge Preparation
12. ✅ **UR5e Environment** - Gymnasium-style interface in `ur5e_cable_insertion.py`
13. ✅ **Hardware Configs** - Robot, gripper, sensor specifications as dataclasses
14. ✅ **Isaac Sim Assets** - Correct CDN paths for UR5e model

---

## Challenges & Risks

1. **Scalability**: Tabular methods limited to small state spaces
2. **Sim-to-Real Gap**: Physics differences between simulation and reality
3. **Cable Modeling**: Deformable objects require advanced techniques
4. **Challenge Timeline**: Official toolkit not available until March 2, 2026

---

## Next Steps (V2)

### RL Algorithms
1. [ ] Implement function approximation (DQN/PPO)
2. [ ] Implement model-based planning with learned beliefs
3. [ ] Add domain randomization for robustness

### AI for Industry Challenge
4. [x] Register for Intrinsic AI Challenge
5. [ ] Integrate official participant toolkit (March 2, 2026)
6. [ ] Implement cable physics (deformable body simulation)
7. [ ] Train UR5e reaching task in Isaac Lab
8. [ ] Add force-torque based insertion policy
9. [ ] Implement vision-based cable detection

---

## Citations

### Algorithms & Theory
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Puterman, M. L. (1994). *Markov Decision Processes*. Wiley.

### Software & Tools
- NVIDIA Isaac Sim 5.1: https://developer.nvidia.com/isaac-sim
- NVIDIA Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Cookiecutter Data Science v2: https://cookiecutter-data-science.drivendata.org/
  - Template: https://github.com/drivendataorg/cookiecutter-data-science
  - See [references/COOKIECUTTER_TEMPLATE.md](references/COOKIECUTTER_TEMPLATE.md) for full details
- NumPy: https://numpy.org/
- SciPy: https://scipy.org/
- Gymnasium: https://gymnasium.farama.org/

### AI for Industry Challenge
- Challenge Details: https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380
- Universal Robots UR5e: https://www.universal-robots.com/products/ur5-robot/
- Robotiq Hand-E: https://robotiq.com/products/hand-e-adaptive-robot-gripper
- ATI Axia80: https://www.ati-ia.com/products/ft/ft_models.aspx?id=Axia80

### Course Materials
- EECS 590 Lecture Slides (Dr. Alexander Lowenstein)
- Mini 1 & Mini 2 Assignment Specifications

### LLM/AI help
- [Chatgpt] open ai chat gpt for coockiecutter setup
- [Chatgpt] open ai chat gpt for Code debugging and documentation 
---

## Collaborators & Contributions

| Contributor | Role | Contributions |
|-------------|------|---------------|
| Ashutosh Kumar | Primary Author | All implementations |

*No external collaborators for V1.*

---

## License

MIT License - See [LICENSE](LICENSE) file.

---

## Contact

**Ashutosh Kumar**  
Email: ashutosh.kumar@und.edu  
GitHub: [@ashuxen](https://github.com/ashuxen)

---

*Last updated: February 2026*
