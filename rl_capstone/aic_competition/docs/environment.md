# AIC Competition Environment

## Challenge Overview

The [Intrinsic AI for Industry Challenge](https://discourse.openrobotics.org/t/ai-for-industry-challenge-challenge-details/52380) requires a UR5e robot to insert cables (SFP and SC connectors) into server rack ports autonomously.

## Hardware Stack

| Component | Model | Specs |
|-----------|-------|-------|
| Robot Arm | Universal Robots UR5e | 6-DOF, 5kg payload, ±0.03mm repeatability |
| Gripper | Robotiq Hand-E | Parallel jaw, 50mm stroke, 130N grip force |
| F/T Sensor | ATI Axia80 | 6-axis, 80mm OD, EtherCAT interface |
| Cameras | Basler (x3) | Wrist-mounted, left/center/right views |

## Simulator

- **Gazebo** with ROS 2 Humble
- Physics simulation of UR5e + Robotiq gripper + cables
- Force/torque sensor feedback
- Three simulated cameras matching real hardware placement

## Evaluation Rules

- 3 trials per evaluation run
- Each trial: robot must pick up a cable and insert it into the assigned port
- Connectors: SFP (small form-factor pluggable) and SC (subscriber connector)
- **Ground truth TF is NOT available** during competition — robot must rely on perception (cameras) and force feedback only
- Scoring is based on insertion depth and alignment quality (max ~75-100 per trial)
- Time limit per trial

## Observation Space

| Signal | Dimensions | Source |
|--------|-----------|--------|
| Camera images | 1024×1152×3 per camera (3 cameras) | Basler cameras |
| Force/torque | 6D (Fx, Fy, Fz, Tx, Ty, Tz) | ATI Axia80 |
| Joint positions | 6D | UR5e encoders |
| Joint velocities | 6D | UR5e encoders |
| End-effector pose | 7D (position + quaternion) | Forward kinematics |
| Gripper state | 1D | Robotiq Hand-E |

## Action Space

The policy outputs a 24D action vector:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| Δpose | 6D | Position (x,y,z) + rotation (rx,ry,rz) corrections |
| ΔK (stiffness) | 6D | Impedance stiffness modulation |
| ΔD (damping) | 6D | Impedance damping modulation |
| ΔF (feedforward) | 6D | Feedforward force/torque |

## Connector Types

### SFP (Small Form-Factor Pluggable)
- Rectangular cross-section
- Insertion depth target: ~25mm
- Relatively forgiving alignment tolerance

### SC (Subscriber Connector)
- Circular cross-section, smaller
- Insertion depth target: ~45mm (deeper than SFP)
- Tighter alignment tolerance — harder to insert
- Was the main challenge during development (see technical-challenges.md)
