# Gazebo Evaluation Evidence

Gazebo was the final validation environment because it matches the AIC competition runtime more closely than Isaac Sim. All final deployment policies were tested with `ground_truth=false`, using perception and force/torque feedback rather than privileged target transforms.

## Evidence Files

Copied scoring outputs are stored in:

```text
reports/gazebo_eval_logs/
```

Included files:

- `aic_results_scoring.yaml`: best local scoring evidence copied from the active AIC results directory.
- `cc15c_recovery_gui_20260510_103252_scoring.yaml`: earlier recovery run with partial insertion evidence.
- `cc15c_recovery_20260513_182000_scoring.yaml`: later recovery run that loaded but failed to complete insertion.

## Best Local Score

The best included local score is:

```text
total: 98.818617164106826
```

This run showed partial insertion on SFP trials:

- trial 1 Tier 3: about 41.98, partial insertion with distance about 0.03 m.
- trial 3 Tier 3: about 46.03, partial insertion with distance about 0.01 m.

The score was not a complete insertion solution. It shows that the perception and impedance heuristic could reach partial SFP insertion, while the final blind seating phase remained unfinished.

## Main Gazebo Lessons

- Perception and geometry generalization mattered more than another generic RL algorithm.
- Fixed world-axis search did not generalize to randomized NIC positions.
- Post-contact search had to be bounded to the NIC face to avoid cable deflection.
- Excessive force penalties were a real concern, so insertion logic needed force-aware stiffness and feedforward tuning.
- Ground-truth data was useful for training and calibration, but not allowed in the final deployment policy.

## Relationship to Isaac Sim

Isaac Sim was used for fast PPO reward shaping and future contact-start training. Gazebo was used to validate whether the ideas transferred into the actual competition stack. This separation was a deliberate engineering decision:

- Isaac: faster iteration and reward debugging.
- Gazebo: slower but authoritative deployment validation.

