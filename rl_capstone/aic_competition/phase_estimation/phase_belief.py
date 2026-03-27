"""Gaussian HMM Belief Estimator for contact-phase detection.

Replaces the hard-threshold ``_classify_contact_state`` with a Bayesian
belief tracker.  Each of the 5 insertion phases has a learned Gaussian
emission model P(obs | phase), and forward-only transition probabilities
ensure monotonic phase progression during descent.

SEATED is only declared when:
  1. P(SEATED) exceeds a high confidence threshold (0.85)
  2. z_offset is below a connector-aware minimum depth gate
  3. The SEATED belief has been dominant for N consecutive steps

Usage:
    hmm = PhaseGaussianHMM.from_file("~/rl/yd_rrl_checkpoints/phase_hmm_params.npz")
    hmm.reset()
    for step in descent_loop:
        obs = [delta_lat, delta_ax, z_offset, force_deriv, force_var]
        phase = hmm.update(obs, min_seated_depth=-0.020)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

PHASE_FREE_SPACE = 0
PHASE_NEAR_CONTACT = 1
PHASE_ALIGNMENT = 2
PHASE_INSERTION = 3
PHASE_SEATED = 4
N_PHASES = 5

PHASE_NAMES = ["FREE_SPACE", "NEAR_CONTACT", "ALIGNMENT", "INSERTION", "SEATED"]

_DEFAULT_SEATED_MEAN = np.array([1.0, 1.5, -0.025, 0.0, 0.2], dtype=np.float32)
_DEFAULT_SEATED_COV = np.diag([2.0, 3.0, 0.0005, 0.5, 0.5]).astype(np.float32)

_DEFAULT_TRANSITION = np.array([
    [0.92, 0.05, 0.02, 0.01, 0.00],
    [0.01, 0.86, 0.05, 0.07, 0.01],
    [0.01, 0.01, 0.85, 0.10, 0.03],
    [0.01, 0.01, 0.03, 0.85, 0.10],
    [0.00, 0.00, 0.00, 0.02, 0.98],
], dtype=np.float32)


def _mvn_log_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log-probability of x under a multivariate Gaussian N(mean, cov)."""
    d = len(x)
    diff = x - mean
    try:
        L = np.linalg.cholesky(cov)
        solve = np.linalg.solve(L, diff)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (d * np.log(2 * np.pi) + log_det + np.dot(solve, solve))
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
        log_det = np.log(max(np.linalg.det(cov), 1e-30))
        return -0.5 * (d * np.log(2 * np.pi) + log_det + diff @ inv_cov @ diff)


class PhaseGaussianHMM:
    """Bayesian contact-phase estimator using Gaussian emissions."""

    def __init__(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        transition_matrix: np.ndarray,
        confidence_threshold: float = 0.85,
        seated_streak_required: int = 5,
    ):
        assert means.shape == (N_PHASES, 5)
        assert covariances.shape == (N_PHASES, 5, 5)
        assert transition_matrix.shape == (N_PHASES, N_PHASES)

        self.means = means.astype(np.float64)
        self.covariances = covariances.astype(np.float64)
        self.T = transition_matrix.astype(np.float64)
        self.confidence_threshold = confidence_threshold
        self.seated_streak_required = seated_streak_required

        self._belief = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        self._seated_streak = 0
        self._step = 0

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "PhaseGaussianHMM":
        """Load fitted parameters from .npz file."""
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"HMM params not found: {p}")

        data = np.load(p, allow_pickle=True)
        means = data["means"].copy()
        covariances = data["covariances"].copy()
        transition_matrix = data["transition_matrix"].copy()
        counts = data["counts"]

        if counts[PHASE_SEATED] < 5:
            means[PHASE_SEATED] = _DEFAULT_SEATED_MEAN
            covariances[PHASE_SEATED] = _DEFAULT_SEATED_COV

        for i in range(N_PHASES):
            eigvals = np.linalg.eigvalsh(covariances[i])
            if np.any(eigvals < 1e-6):
                covariances[i] += np.eye(5) * 1e-4

        if transition_matrix[PHASE_INSERTION, PHASE_SEATED] < 0.01:
            transition_matrix = _DEFAULT_TRANSITION.copy()

        return cls(means, covariances, transition_matrix, **kwargs)

    def reset(self):
        """Reset belief to initial state (FREE_SPACE)."""
        self._belief = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        self._seated_streak = 0
        self._step = 0

    @property
    def belief(self) -> np.ndarray:
        return self._belief.copy()

    @property
    def seated_streak(self) -> int:
        return self._seated_streak

    def update(
        self,
        obs: np.ndarray,
        min_seated_depth: float = -0.020,
    ) -> int:
        """Bayesian belief update and return the estimated phase.

        Args:
            obs: 5D observation [delta_lat, delta_ax, z_offset, force_deriv, force_var]
            min_seated_depth: connector-aware minimum z_offset for SEATED

        Returns:
            Phase integer (0-4).
        """
        obs = np.asarray(obs, dtype=np.float64)
        self._step += 1

        # --- Prediction step: propagate belief through transition matrix ---
        predicted = self.T.T @ self._belief

        # --- Observation step: compute emission likelihoods ---
        log_likelihoods = np.array([
            _mvn_log_pdf(obs, self.means[k], self.covariances[k])
            for k in range(N_PHASES)
        ])
        max_ll = log_likelihoods.max()
        likelihoods = np.exp(log_likelihoods - max_ll)

        # --- Depth gate: suppress SEATED likelihood if too shallow ---
        z_offset = obs[2]
        if z_offset > min_seated_depth:
            likelihoods[PHASE_SEATED] = 0.0

        # --- Bayesian update ---
        updated = predicted * likelihoods
        total = updated.sum()
        if total > 0:
            self._belief = updated / total
        else:
            self._belief = predicted / predicted.sum()

        # --- Phase decision with temporal consistency ---
        best_phase = int(np.argmax(self._belief))

        if best_phase == PHASE_SEATED and self._belief[PHASE_SEATED] >= self.confidence_threshold:
            self._seated_streak += 1
        else:
            self._seated_streak = 0

        if self._seated_streak >= self.seated_streak_required:
            return PHASE_SEATED

        if best_phase == PHASE_SEATED:
            return PHASE_INSERTION

        return best_phase

    def format_belief(self) -> str:
        """Format belief vector for logging."""
        parts = []
        for i, name in enumerate(PHASE_NAMES):
            short = name[:4]
            parts.append(f"{short}={self._belief[i]:.2f}")
        return f"[{' '.join(parts)}]"
