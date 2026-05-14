# Final V3 Decisions

This file is the final decision record for Version 3. It explains which algorithms I chose to use, which I chose not to use, and how those choices fit the cable insertion problem.

## Final Problem Definition

The capstone target is a partially observable, contact-rich manipulation problem:

- A UR5e robot must insert SFP and SC connectors into randomized ports.
- The final policy cannot use ground-truth transforms.
- Cameras become less useful near contact because the plug and NIC face occlude the port entrance.
- Force/torque feedback becomes essential during the last few centimeters.
- The controller must manage both pose and impedance to avoid excessive force.

The final system uses a hybrid strategy: perception and deterministic impedance logic for deployment, plus PPO and residual learning experiments for the contact-search portion.

## Algorithms I Implemented or Used

### Dynamic Programming and TD(lambda)

I kept the tabular RL foundation from earlier versions:

- finite MDP representation,
- value iteration,
- Q-value iteration,
- policy iteration,
- TD(lambda),
- a Windy Chasm environment for controlled testing.

These algorithms are not the final answer for cable insertion because the robot task is continuous and high-dimensional, but they were useful for showing the course foundations and for validating the agent/trainer structure.

### PPO

I chose PPO as the main deep RL algorithm for the capstone because:

- the action space is continuous,
- the policy needs conservative updates,
- impedance-aware residual actions can become unsafe if updated too aggressively,
- PPO is easier to stabilize than more complex trust-region methods.

In Isaac Sim / Isaac Lab, I trained a sequence of PPO reward-shaping runs. The most important runs were V9-V12:

- **V9:** pure PPO from scratch after behavior cloning transfer failed.
- **V10:** disabled lateral-excursion termination that the agent was exploiting.
- **V11:** added stronger far-field approach shaping through pre-entrance depth progress.
- **V12:** added close-range seating rewards and soft gated depth pulling.

PPO improved approach behavior and achieved nonzero Tier 2 signal, but it did not produce a robust deployable Tier 3 insertion policy before the deadline.

### Behavior Cloning and DAgger

I used behavior cloning and DAgger-style teacher data as a warm-start experiment. The idea was to initialize residual policies from demonstrations collected with ground truth enabled.

I deprioritized BC for final deployment because the initial pose distribution did not match well enough between Gazebo data and Isaac training. Warm-started PPO inherited a bias that caused early failures. More demonstrations did not fix the distribution mismatch by itself.

### Belief Tracking: HMM and Recurrent Memory

I kept a Gaussian HMM phase estimator because the task is partially observable. Force thresholds alone were too brittle. A belief tracker is useful for estimating whether the robot is in free-space approach, near contact, alignment, insertion, or seated.

I also tested recurrent PPO as a learned memory/belief experiment in Isaac Lab. The V9-V12 Isaac configs used `ActorCriticRecurrent` with a one-layer GRU (`rnn_hidden_dim=128`). I considered this in the same role as an LSTM/GRU belief state: the policy can summarize recent pose, action, and contact history instead of making each action from a single observation. For the final repository, I documented both the explicit HMM belief estimator and the recurrent GRU policy configs.

I did not fully formulate the entire task as an online POMDP solver because that would be too large for the deadline. I used belief updates and recurrent memory in the parts where partial observability was most visible.

### Heuristic Impedance Policy

The final deployment path used a deterministic perception-plus-impedance policy. The heuristic policy scored better locally than the unfinished RL policy, and it was the only path ready for competition-style `ground_truth=false` evaluation by the deadline.

The final heuristic work focused on:

- dynamic NIC geometry instead of fixed world axes,
- contact detection from force/torque,
- bounded search on the NIC face,
- SFP-specific seating depth and axial push logic,
- avoiding excessive cable deflection during blind tactile search.

## Algorithms I Chose Not to Implement

### Swarm or Multi-Agent Coordination

I did not implement swarm or coordination algorithms. The AIC task has one robot manipulating one cable at a time, so there was no direct multi-agent coordination problem.

### Full POMDP Planning

The task is partially observable, but full online POMDP planning was not practical for the continuous robot state and contact dynamics. I used the HMM phase estimator and recurrent PPO memory instead.

### SAC, TD3, and DDPG

I considered off-policy continuous-control algorithms, but did not prioritize them for the final version. They are sample-efficient, but replay-buffer instability is a concern when the policy transitions between free-space motion, first contact, sliding, jamming, and insertion. PPO's conservative update rule was a better first choice for this capstone.

### Transformer Policies

I did not implement an ACT or transformer policy for the final submission. The bottleneck was not long-horizon sequence modeling. The bottleneck was accurate perception, geometry transfer, and blind tactile search after face contact.

### Metaheuristics as a Main Policy

Metaheuristics could help tune impedance parameters, search radii, and reward weights. I did not make them the main policy because they do not learn tactile state feedback directly. They remain a good future tool for hyperparameter tuning.

## What I Would Do Next

The next logical step is a focused contact-start RL problem:

1. Use the deterministic policy to bring the SFP plug within 20-30 mm of the port or to first face contact.
2. Start short RL episodes from randomized contact states.
3. Use ground truth only for reward, reset, and teacher labels during training.
4. Keep actor observations deploy-safe: perception estimate, TCP state, force/torque, and action history.
5. Train in Isaac Sim for speed, then validate transfer in Gazebo.

This is narrower and more realistic than trying to train the entire insertion pipeline end-to-end from scratch.

