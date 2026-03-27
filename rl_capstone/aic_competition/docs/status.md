# Competition Progress & Status

## Current Status (February 2026)

### What's Working
- PPO + DAgger hybrid residual policy trained and deployed
- CNN perception (Port2DNetV4) detecting both SFP and SC ports from camera images
- 3-camera triangulation giving ~2mm accuracy at approach distance
- Impedance-aware control adapting stiffness/damping during insertion
- Gaussian HMM phase estimator replacing brittle rule-based classifier
- Retry mechanism for failed insertion attempts
- Full evaluation pipeline running in Gazebo without ground truth TF

### What's In Progress
- SC connector insertion depth and alignment tuning
- HMM evaluation to confirm score improvement over hard-threshold classifier
- Further PPO training with more successful SC episodes

### Known Issues
- SC insertion success rate still low — needs more training data with successful insertions
- Perception drift at close range when port partially leaves camera FOV
- CNN center camera confidence drops at very close distances (σ > 0.15)

## Training Data Collected

| Dataset | Episodes | Connectors | Used For |
|---------|----------|------------|----------|
| PPO episodes | 154 | SFP + SC | PPO policy training |
| Perception data | 194 | SFP + SC | CNN port detector training |
| DAgger round 1 | ~50 | SFP + SC | Base policy |
| DAgger round 2 | ~50 | SFP + SC | Improved policy |
| DAgger round 3 | ~50 | SFP + SC | Final expert baseline |

## Model Versions

| Model | File | Details |
|-------|------|---------|
| CNN perception | `models/checkpoints/perception/port_2d_v4.pt` | ResNet-18 + FPN + FiLM, 300 epochs, 194 episodes |
| PPO actor | `models/checkpoints/ppo/ppo_actor.pt` | 128-128-128 MLP, warm-started from DAgger |
| PPO critic | `models/checkpoints/ppo/ppo_critic.pt` | 128-128 MLP value estimator |
| DAgger expert | `models/checkpoints/dagger/residual_mlp.pt` | 3 rounds of interactive correction |
| Phase HMM | `models/checkpoints/hmm/phase_hmm_params.npz` | 5-phase Gaussian emissions, fitted from 154 episodes |

## Evaluation Scores

| Run | Ground Truth | Trial 1 | Notes |
|-----|-------------|---------|-------|
| Pre-HMM | No | 19.0 | Premature SEATED at 10mm depth |
| Post-HMM | No | In progress | Evaluating with depth gate + HMM |
| With TF | Yes | ~75+ | Works well when TF is available |
