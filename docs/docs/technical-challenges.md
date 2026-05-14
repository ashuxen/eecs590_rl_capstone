# Technical Challenges and Lessons

This file records the main issues I hit while developing the capstone: what worked, what failed, and why I changed direction.

## 1. Ground Truth Data Did Not Transfer Cleanly

I collected Gazebo data with ground truth enabled and used it for behavior cloning and DAgger-style warm starts. The problem was that initial positions and randomization distributions did not match Isaac training closely enough. PPO warm-started from these demonstrations inherited the wrong bias and failed early.

Lesson: demonstrations are only helpful when their state distribution overlaps the deployment or training distribution.

## 2. PPO Reward Hacking

Early PPO runs learned to exploit termination behavior instead of inserting the cable. In V9, lateral-excursion termination became an easy way to end episodes without learning useful approach or seating behavior.

Fix: remove or soften termination paths that reward the wrong behavior, then add positive shaping for real progress.

## 3. Hover and Pre-Entrance Stall

After removing the lateral-excursion exploit, the agent could still hover or stall before meaningful insertion. It needed a reward term that specifically encouraged progress toward the port face before the actual seating phase.

Fix: V11 added pre-entrance depth progress and stronger approach shaping.

## 4. Strict Seating Was Still Hard

V12 improved close-range reward shaping, but strict Tier 3 seating did not become reliable. The policy learned useful approach and some depth behavior, but the blind contact-search phase remained difficult.

Lesson: full end-to-end PPO was too broad for the deadline. A narrower contact-start problem is a better next step.

## 5. Perception Generalization Was a Deployment Bottleneck

The Gazebo heuristic originally used fixed world axes during lateral search. This failed when the NIC card and board were randomized. The fix was to build the search frame from detected NIC geometry instead of global axes.

Lesson: policy logic can still fail if its coordinate frame is wrong.

## 6. Contact Search Can Deflect the Cable

When search radius was too large, the cable could bend off the NIC face and push into air. That produced poor insertion behavior and risked force penalties.

Fix: bound the search to the NIC card face, keep raster/spiral motions small, and use force-aware axial push.

## 7. Force Penalties Matter

The scoring rules penalize excessive force. Local logs showed that high peak force could be tolerated briefly, but sustained force above threshold caused penalties. This made impedance tuning and contact dwell logic central to the project.

Lesson: the reward and final heuristic both need to reflect safety and scoring rules, not just final pose.

## 8. Repository Readiness Is Different From Competition Readiness

The full AIC workspace, Docker image, Isaac Lab installation, and raw training logs live outside this repository. For V3, this repository keeps a compact record:

- algorithm code,
- policy code snapshots,
- training summaries,
- scoring evidence,
- final decisions,
- known limitations.

Large generated artifacts are not committed unless they are needed to review the work.

