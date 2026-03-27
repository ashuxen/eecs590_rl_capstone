"""Retroactively relabel rewards in collected PPO episodes.

Uses the same reward formula as the updated _ppo_compute_reward, but
reconstructs inputs from the stored 26-dim state vector:

  dims  0-2  : F_local (force in port-local frame)
  dims  3-5  : τ_local (torque in port-local frame)
  dims  6-8  : pose_error_local xyz (port_xyz - tcp_xyz, rotated by yaw)
  dim   12   : phase_norm  (phase / 4.0)  — always 0 in old data
  dim   16   : contact_flag
  dims 17-18 : connector one-hot [sfp, sc]

z_offset is stored per-step.  Episode-level: success flag.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

PHASE_INSERTION = 3
PHASE_SEATED = 4
FUNNEL_CONTACT_THRESHOLD = 3.0


def relabel_episode(states, z_offsets, success_flag):
    """Recompute rewards from stored state vectors.

    Returns new reward array with the same shape as z_offsets.
    """
    n_steps = len(states)
    rewards = np.zeros(n_steps, dtype=np.float32)

    is_sc = states[0, 18] > 0.5

    prev_dist = 0.0
    for i in range(n_steps):
        s = states[i]
        z_off = float(z_offsets[i])
        reward = -0.01  # time penalty

        # TCP-to-port distance from pose_error_local (norm preserved by rotation)
        dist = float(np.linalg.norm(s[6:9]))

        # Distance progress reward
        if prev_dist > 0:
            progress = prev_dist - dist
            reward += 5.0 * progress
        prev_dist = dist

        # Proximity bonus
        if dist < 0.02:
            reward += 0.1 * (0.02 - dist) / 0.02

        # Insertion bonus (connector-aware thresholds)
        phase = int(round(s[12] * 4.0))
        if is_sc:
            if z_off < -0.01 and (dist < 0.025 or phase >= PHASE_INSERTION):
                reward += 10.0
        else:
            if z_off < -0.01 and dist < 0.01:
                reward += 10.0

        # Force-based reward (available without TF)
        F_axial = abs(float(s[2]))
        F_lateral = float(np.linalg.norm(s[0:2]))

        if z_off < 0.02 and F_axial > 3.0:
            reward += 0.05 * min(F_axial / 10.0, 1.0)

        if F_lateral > FUNNEL_CONTACT_THRESHOLD:
            reward -= 0.02 * min(F_lateral / 10.0, 1.0)

        # Phase bonuses (will be 0 for old data where phase=0)
        if phase == PHASE_INSERTION:
            reward += 0.5
        elif phase == PHASE_SEATED:
            reward += 2.0

        rewards[i] = reward

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Relabel PPO episode rewards")
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "rl" / "ppo_training_data"),
        help="Root of PPO training data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without modifying files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes = sorted(data_dir.glob("iter_*/episode_*.npz"))
    print(f"Found {len(episodes)} episodes in {data_dir}")

    sfp_count, sc_count = 0, 0
    sfp_old_mean, sfp_new_mean = [], []
    sc_old_mean, sc_new_mean = [], []

    for ep_path in episodes:
        ep = np.load(ep_path, allow_pickle=True)
        states = ep["states"]
        z_offsets = ep["z_offsets"]
        old_rewards = ep["rewards"]
        success = ep["success"]

        is_sc = states[0, 18] > 0.5
        if is_sc:
            new_rewards = relabel_episode(states, z_offsets, success)
            sc_count += 1
            sc_old_mean.append(old_rewards.mean())
            sc_new_mean.append(new_rewards.mean())

            if not args.dry_run:
                np.savez_compressed(
                    ep_path,
                    states=states,
                    actions=ep["actions"],
                    log_probs=ep["log_probs"],
                    values=ep["values"],
                    rewards=new_rewards,
                    z_offsets=z_offsets,
                    success=success,
                )
        else:
            sfp_count += 1
            sfp_old_mean.append(old_rewards.mean())

    print(f"\nSFP episodes: {sfp_count} (unchanged)")
    if sfp_old_mean:
        print(f"  Reward mean: {np.mean(sfp_old_mean):.4f}")

    print(f"\nSC episodes: {sc_count} (relabeled)")
    if sc_old_mean:
        print(f"  Old reward mean: {np.mean(sc_old_mean):.4f}")
        print(f"  New reward mean: {np.mean(sc_new_mean):.4f}")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
    else:
        print(f"\nRelabeled {len(episodes)} episodes in-place.")


if __name__ == "__main__":
    main()
