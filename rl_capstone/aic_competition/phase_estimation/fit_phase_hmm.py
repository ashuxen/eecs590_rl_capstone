"""Fit Gaussian HMM parameters for contact-phase estimation.

Extracts force/position features from collected PPO episodes and fits
per-phase Gaussian emission parameters + transition matrix via MLE.

The 5 phases map to the SmartInsert state machine:
  0: FREE_SPACE, 1: NEAR_CONTACT, 2: ALIGNMENT, 3: INSERTION, 4: SEATED

Observation vector (5D):
  [delta_lateral_force, delta_axial_force, z_offset,
   axial_force_derivative, axial_force_variance_10step]

For episodes with missing phase labels (old data where phase_norm=0),
pseudo-labels are generated from z_offset and force readings using
the same heuristic as the original _classify_contact_state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PHASE_FREE_SPACE = 0
PHASE_NEAR_CONTACT = 1
PHASE_ALIGNMENT = 2
PHASE_INSERTION = 3
PHASE_SEATED = 4
N_PHASES = 5

FUNNEL_CONTACT_THRESHOLD = 3.0
FUNNEL_INSERTION_THRESHOLD = 5.0
FUNNEL_SEATED_Z_THRESHOLD = 2.0


def heuristic_phase(delta_lat: float, delta_ax: float, z_offset: float) -> int:
    """Pseudo-label a timestep when no phase label is available."""
    if z_offset < -0.020:
        if delta_ax < FUNNEL_SEATED_Z_THRESHOLD and delta_lat < 2.0:
            return PHASE_SEATED
        return PHASE_INSERTION
    if z_offset < -0.005:
        return PHASE_INSERTION
    if delta_lat > FUNNEL_CONTACT_THRESHOLD:
        if delta_ax > FUNNEL_INSERTION_THRESHOLD and delta_lat < delta_ax * 0.5:
            return PHASE_INSERTION
        return PHASE_ALIGNMENT
    if z_offset < 0.02:
        return PHASE_NEAR_CONTACT
    return PHASE_FREE_SPACE


def extract_features(states: np.ndarray, z_offsets: np.ndarray):
    """Extract 5D observation vectors and phase labels from an episode.

    State layout (26D):
      0-2: F_local, 3-5: τ_local, 6-8: pose_error_local xyz,
      9-11: pose_error_local rpy, 12: phase_norm, 13: force_deriv_axial,
      14: lateral_angle, 15: insertion_progress, 16: contact_flag,
      17-18: connector_type [sfp, sc], 19: time_remaining, 20-25: a_base_local
    """
    T = len(states)
    obs_vectors = np.zeros((T, 5), dtype=np.float32)
    phase_labels = np.zeros(T, dtype=np.int32)

    F_local = states[:, 0:3]
    lateral_forces = np.linalg.norm(F_local[:, :2], axis=1)
    axial_forces = np.abs(F_local[:, 2])

    baseline_lat = float(np.median(lateral_forces[:min(20, T)]))
    baseline_ax = float(np.median(axial_forces[:min(20, T)]))

    for t in range(T):
        delta_lat = abs(lateral_forces[t] - baseline_lat)
        delta_ax = abs(axial_forces[t] - baseline_ax)
        z_off = float(z_offsets[t])

        force_deriv = 0.0
        if t > 0:
            force_deriv = axial_forces[t] - axial_forces[t - 1]

        window_start = max(0, t - 10)
        force_var = float(np.var(axial_forces[window_start:t + 1]))

        obs_vectors[t] = [delta_lat, delta_ax, z_off, force_deriv, force_var]

        stored_phase = int(round(states[t, 12] * 4.0))
        if stored_phase > 0:
            phase_labels[t] = min(stored_phase, PHASE_SEATED)
        else:
            phase_labels[t] = heuristic_phase(delta_lat, delta_ax, z_off)

    return obs_vectors, phase_labels


def fit_gaussians(observations: np.ndarray, labels: np.ndarray):
    """Fit per-phase Gaussian emission parameters via MLE."""
    n_features = observations.shape[1]
    means = np.zeros((N_PHASES, n_features), dtype=np.float64)
    covariances = np.zeros((N_PHASES, n_features, n_features), dtype=np.float64)
    counts = np.zeros(N_PHASES, dtype=np.int64)

    for phase in range(N_PHASES):
        mask = labels == phase
        counts[phase] = mask.sum()
        if counts[phase] < 2:
            means[phase] = 0.0
            covariances[phase] = np.eye(n_features) * 1.0
            continue
        phase_obs = observations[mask]
        means[phase] = phase_obs.mean(axis=0)
        covariances[phase] = np.cov(phase_obs, rowvar=False) + np.eye(n_features) * 1e-6

    return means, covariances, counts


def estimate_transitions(labels: np.ndarray, episode_lengths: list[int]):
    """Estimate transition matrix from phase label sequences."""
    T_matrix = np.zeros((N_PHASES, N_PHASES), dtype=np.float64)

    offset = 0
    for length in episode_lengths:
        ep_labels = labels[offset:offset + length]
        for t in range(len(ep_labels) - 1):
            s_from = ep_labels[t]
            s_to = ep_labels[t + 1]
            T_matrix[s_from, s_to] += 1.0
        offset += length

    for i in range(N_PHASES):
        row_sum = T_matrix[i].sum()
        if row_sum > 0:
            T_matrix[i] /= row_sum
        else:
            T_matrix[i, i] = 1.0

    return T_matrix


def main():
    parser = argparse.ArgumentParser(description="Fit Gaussian HMM for phase estimation")
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / "rl" / "ppo_training_data" / "iter_combined"),
    )
    parser.add_argument(
        "--output",
        default=str(Path.home() / "rl" / "yd_rrl_checkpoints" / "phase_hmm_params.npz"),
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episodes = sorted(data_dir.glob("episode_*.npz"))
    print(f"Loading {len(episodes)} episodes from {data_dir}")

    all_obs = []
    all_labels = []
    episode_lengths = []

    for ep_path in episodes:
        ep = np.load(ep_path, allow_pickle=True)
        states = ep["states"]
        z_offsets = ep["z_offsets"]

        obs, labels = extract_features(states, z_offsets)
        all_obs.append(obs)
        all_labels.append(labels)
        episode_lengths.append(len(obs))

    all_obs = np.concatenate(all_obs)
    all_labels = np.concatenate(all_labels)

    print(f"Total timesteps: {len(all_obs)}")
    for phase in range(N_PHASES):
        names = ["FREE_SPACE", "NEAR_CONTACT", "ALIGNMENT", "INSERTION", "SEATED"]
        count = (all_labels == phase).sum()
        print(f"  {names[phase]:15s}: {count:6d} ({100*count/len(all_labels):.1f}%)")

    print("\nFitting Gaussian emission models...")
    means, covariances, counts = fit_gaussians(all_obs, all_labels)

    print("Estimating transition matrix...")
    T_matrix = estimate_transitions(all_labels, episode_lengths)

    print("\nTransition matrix:")
    names = ["FREE", "NEAR", "ALIGN", "INSERT", "SEAT"]
    print(f"{'':>8s}", end="")
    for n in names:
        print(f"{n:>8s}", end="")
    print()
    for i, n in enumerate(names):
        print(f"{n:>8s}", end="")
        for j in range(N_PHASES):
            print(f"{T_matrix[i, j]:8.3f}", end="")
        print()

    print(f"\nPer-phase Gaussian means (5D):")
    feat_names = ["Δlat", "Δax", "z_off", "f_deriv", "f_var"]
    for i in range(N_PHASES):
        names_full = ["FREE_SPACE", "NEAR_CONTACT", "ALIGNMENT", "INSERTION", "SEATED"]
        print(f"  {names_full[i]:15s}: ", end="")
        for j, fn in enumerate(feat_names):
            print(f"{fn}={means[i, j]:7.4f} ", end="")
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        means=means.astype(np.float32),
        covariances=covariances.astype(np.float32),
        transition_matrix=T_matrix.astype(np.float32),
        counts=counts,
    )
    print(f"\nSaved HMM parameters to {output_path}")


if __name__ == "__main__":
    main()
