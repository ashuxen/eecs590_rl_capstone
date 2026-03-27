"""
Impedance Residual MLP for Impedance-Aware ResiP (upgraded from YD-RRL).

Architecture extends ResiP paper (2407.16677) with:
  - Phase-adaptive impedance residual (ΔK, ΔD, ΔF per insertion phase)
  - Contact-state features in observation (force derivative, phase encoding)
  - Force-funnel-aware state (lateral force direction)

State (26D):
  F_local(3), τ_local(3), pose_error_local(6),
  contact_features(3):  [phase_norm, force_deriv_axial, lateral_angle],
  insertion_progress(1), contact_flag(1), connector_type(2),
  time_remaining(1), a_base_local(6)

  Note: dims 12-14 were previously `cable_tension` (always zeros).
  Now repurposed for contact-state features.  Backward-compatible:
  old data has zeros → model treats as "no contact info available".

Action (24D):
  Δpose(6): position ±1mm, orientation ±0.5°
  ΔK(6): stiffness correction (diagonal), bounded ±30 N/m
  ΔD(6): damping correction (diagonal), bounded ±20 Ns/m
  ΔF(6): feedforward wrench at tip, bounded ±5 N / ±1 Nm
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# --- V2 dimensions (impedance residual) ---
STATE_DIM = 26
ACTION_DIM = 24

# Action bounds per component
POS_BOUND = 0.001        # ±1mm
ROT_BOUND = 0.0087       # ±0.5°
K_BOUND = 30.0           # ±30 N/m stiffness correction
D_BOUND = 20.0           # ±20 Ns/m damping correction
F_BOUND_LIN = 5.0        # ±5 N feedforward force
F_BOUND_ROT = 1.0        # ±1 Nm feedforward torque

# Residual scaling (per ResiP paper: multiply raw output by alpha before adding)
RESIDUAL_ALPHA = 0.1

# V1 dims (backward compat for loading old checkpoints)
STATE_DIM_V1 = 20
ACTION_DIM_V1 = 6

# Base impedance parameters (from SmartInsert._set_pose_target_soft)
K_BASE = np.array([90.0, 90.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float32)
D_BASE = np.array([50.0, 50.0, 30.0, 20.0, 20.0, 20.0], dtype=np.float32)

ACTION_BOUNDS = np.array([
    POS_BOUND, POS_BOUND, POS_BOUND,
    ROT_BOUND, ROT_BOUND, ROT_BOUND,
    K_BOUND, K_BOUND, K_BOUND, K_BOUND, K_BOUND, K_BOUND,
    D_BOUND, D_BOUND, D_BOUND, D_BOUND, D_BOUND, D_BOUND,
    F_BOUND_LIN, F_BOUND_LIN, F_BOUND_LIN,
    F_BOUND_ROT, F_BOUND_ROT, F_BOUND_ROT,
], dtype=np.float32)


def clip_action(action: np.ndarray) -> np.ndarray:
    """Clip all action components to their respective bounds."""
    a = np.asarray(action, dtype=np.float32).ravel()[:ACTION_DIM]
    return np.clip(a, -ACTION_BOUNDS[:len(a)], ACTION_BOUNDS[:len(a)])


def unpack_action(action: np.ndarray) -> dict:
    """Unpack 24D action into named components."""
    a = np.asarray(action, dtype=np.float32).ravel()
    return {
        "delta_pose": a[0:6],      # Δx,Δy,Δz,Δrx,Δry,Δrz in port-local
        "delta_K": a[6:12],         # stiffness correction (6 diagonal)
        "delta_D": a[12:18],        # damping correction (6 diagonal)
        "delta_F": a[18:24],        # feedforward wrench at tip
    }


def compute_impedance(action: np.ndarray) -> dict:
    """Compute final impedance parameters from base + residual."""
    parts = unpack_action(action)
    K_final = np.clip(K_BASE + parts["delta_K"], 5.0, 200.0)
    D_final = np.clip(D_BASE + parts["delta_D"], 2.0, 100.0)
    F_final = parts["delta_F"]
    return {
        "stiffness": K_final,
        "damping": D_final,
        "feedforward_wrench": F_final,
        "delta_pose": parts["delta_pose"],
    }


class ResidualMLP:
    """Impedance Residual MLP [128,128,128] with orthogonal init.

    Supports both V1 (20→6) and V2 (26→24) checkpoints.
    Falls back to zero output if no checkpoint loaded.
    """

    def __init__(self, checkpoint_path: Optional[os.PathLike | str] = None):
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self._use_torch = False
        self._model = None
        self._version = 2
        self._state_dim = STATE_DIM
        self._action_dim = ACTION_DIM

        if checkpoint_path and Path(checkpoint_path).expanduser().exists():
            self._try_load_torch()

        if not self._use_torch:
            self._numpy_zero_layers()

    def _numpy_zero_layers(self):
        """Zero-init numpy fallback (residual = 0, no correction)."""
        self._W1 = np.zeros((self._state_dim, 128), dtype=np.float32)
        self._b1 = np.zeros(128, dtype=np.float32)
        self._W2 = np.zeros((128, 128), dtype=np.float32)
        self._b2 = np.zeros(128, dtype=np.float32)
        self._W3 = np.zeros((128, 128), dtype=np.float32)
        self._b3 = np.zeros(128, dtype=np.float32)
        self._W4 = np.zeros((128, self._action_dim), dtype=np.float32)
        self._b4 = np.zeros(self._action_dim, dtype=np.float32)

    def _try_load_torch(self):
        try:
            import torch
            p = Path(self.checkpoint_path).expanduser()
            if (p / "residual_mlp.pt").exists():
                state = torch.load(p / "residual_mlp.pt", map_location="cpu", weights_only=True)
                first_key = next(iter(state))
                in_dim = state[first_key].shape[1] if state[first_key].ndim == 2 else state[first_key].shape[0]
                last_key = list(state.keys())[-1]
                out_dim = state[last_key].shape[0]

                if in_dim == STATE_DIM_V1 or out_dim == ACTION_DIM_V1:
                    self._version = 1
                    self._state_dim = STATE_DIM_V1
                    self._action_dim = ACTION_DIM_V1
                else:
                    self._version = 2
                    self._state_dim = STATE_DIM
                    self._action_dim = ACTION_DIM

                self._model = self._build_torch_model(self._state_dim, self._action_dim)
                self._model.load_state_dict(state)
                self._model.eval()
                self._use_torch = True
                return
        except Exception:
            pass
        self._use_torch = False

    @staticmethod
    def _build_torch_model(s_dim: int = STATE_DIM, a_dim: int = ACTION_DIM):
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(s_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, a_dim),
                )

            def forward(self, x):
                return self.fc(x)

        return MLP()

    @staticmethod
    def build_torch_model_with_ortho_init(s_dim: int = STATE_DIM, a_dim: int = ACTION_DIM):
        """Build MLP with orthogonal init and small final-layer gain (per ResiP)."""
        import torch
        import torch.nn as nn

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(s_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, a_dim),
                )
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight)
                        nn.init.zeros_(layer.bias)
                # Small gain on final layer so initial output ≈ 0
                final = self.fc[-1]
                nn.init.orthogonal_(final.weight, gain=0.01)

            def forward(self, x):
                return self.fc(x)

        return MLP()

    def forward(self, state: np.ndarray) -> np.ndarray:
        """state → action, clipped to bounds. Handles V1 (20→6) and V2 (26→24)."""
        state = np.asarray(state, dtype=np.float32).ravel()

        if self._version == 1 and state.size >= STATE_DIM:
            state = state[:STATE_DIM_V1]
        elif state.size < self._state_dim:
            state = np.pad(state, (0, self._state_dim - state.size))
        else:
            state = state[:self._state_dim]

        if self._use_torch and self._model is not None:
            import torch
            with torch.no_grad():
                x = torch.from_numpy(state).float().unsqueeze(0)
                out = self._model(x).squeeze(0).numpy()
        else:
            x = np.maximum(0, state @ self._W1 + self._b1)
            x = np.maximum(0, x @ self._W2 + self._b2)
            x = np.maximum(0, x @ self._W3 + self._b3)
            out = x @ self._W4 + self._b4

        out = out.ravel()

        if self._version == 1:
            # V1: 6D pose-only → pad to 24D (impedance = 0)
            full = np.zeros(ACTION_DIM, dtype=np.float32)
            full[:min(6, len(out))] = out[:6]
            full[:3] = np.clip(full[:3], -POS_BOUND, POS_BOUND)
            full[3:6] = np.clip(full[3:6], -ROT_BOUND, ROT_BOUND)
            return full

        return clip_action(out)


def build_contact_features(
    phase: int = 0,
    force_deriv_axial: float = 0.0,
    lateral_force_world: Optional[np.ndarray] = None,
    port_z_axis: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build 3D contact-state features for dims 12-14 of the state vector.

    Returns [phase_norm, force_deriv_axial_norm, lateral_angle]:
      - phase_norm: insertion phase 0-4 normalized to [0, 1]
      - force_deriv_axial_norm: rate of change of axial force (tanh-normalised)
      - lateral_angle: angle of lateral force in perpendicular plane [0, 2π]/2π
    """
    phase_norm = float(phase) / 4.0

    deriv_norm = float(np.tanh(force_deriv_axial / 5.0))

    lat_angle = 0.0
    if lateral_force_world is not None and port_z_axis is not None:
        F_lat = lateral_force_world
        lat_mag = np.linalg.norm(F_lat)
        if lat_mag > 0.5:
            perp1 = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(perp1, port_z_axis)) > 0.9:
                perp1 = np.array([0.0, 1.0, 0.0])
            perp1 = perp1 - np.dot(perp1, port_z_axis) * port_z_axis
            n = np.linalg.norm(perp1)
            if n > 1e-6:
                perp1 /= n
                perp2 = np.cross(port_z_axis, perp1)
                lat_angle = float(np.arctan2(np.dot(F_lat, perp2), np.dot(F_lat, perp1)))
                lat_angle = (lat_angle + np.pi) / (2 * np.pi)  # normalise to [0, 1]

    return np.array([phase_norm, deriv_norm, lat_angle], dtype=np.float32)


def build_yd_rrl_state(
    force_world: np.ndarray,
    torque_world: np.ndarray,
    pose_error_world: np.ndarray,
    yaw_rad: float,
    insertion_progress: float,
    contact_flag: float,
    connector_sfp: bool,
    time_remaining_norm: float,
    base_action_local: Optional[np.ndarray] = None,
    cable_tension_est: Optional[np.ndarray] = None,
    contact_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build 26-dim state in port-local frame.

    Dims 12-14 carry contact-state features when available (phase, force
    derivative, lateral angle).  Falls back to cable_tension_est or zeros
    for backward compatibility with older data.
    """
    try:
        from training.frame_decomposer import world_to_port_local_force_torque, yaw_rotation_matrix
    except ImportError:
        from frame_decomposer import world_to_port_local_force_torque, yaw_rotation_matrix

    F_local, τ_local = world_to_port_local_force_torque(
        np.asarray(force_world).ravel()[:3],
        np.asarray(torque_world).ravel()[:3],
        yaw_rad,
    )
    R = yaw_rotation_matrix(yaw_rad)
    pose_err = np.asarray(pose_error_world).ravel()
    if len(pose_err) >= 6:
        pose_error_local = np.concatenate([R @ pose_err[:3], R @ pose_err[3:6]])
    else:
        pose_error_local = np.concatenate([
            R @ pose_err[:3] if len(pose_err) >= 3 else np.zeros(3),
            np.zeros(3),
        ])

    # Dims 12-14: contact features (preferred) > cable_tension > zeros
    if contact_features is not None:
        feat_3d = np.asarray(contact_features, dtype=np.float32).ravel()[:3]
    elif cable_tension_est is not None:
        feat_3d = np.asarray(cable_tension_est).ravel()[:3].astype(np.float32)
    else:
        feat_3d = np.zeros(3, dtype=np.float32)

    conn = np.array([1.0, 0.0] if connector_sfp else [0.0, 1.0], dtype=np.float32)

    if base_action_local is None:
        base_action_local = np.zeros(6, dtype=np.float32)
    a_base = np.asarray(base_action_local, dtype=np.float32).ravel()
    if len(a_base) < 6:
        a_base = np.pad(a_base, (0, 6 - len(a_base)))
    a_base = a_base[:6]

    return np.concatenate([
        F_local.astype(np.float32),          # 3  (dims 0-2)
        τ_local.astype(np.float32),          # 3  (dims 3-5)
        pose_error_local.astype(np.float32), # 6  (dims 6-11)
        feat_3d,                              # 3  (dims 12-14)
        [float(insertion_progress)],          # 1  (dim 15)
        [float(contact_flag)],                # 1  (dim 16)
        conn,                                 # 2  (dims 17-18)
        [float(time_remaining_norm)],         # 1  (dim 19)
        a_base.astype(np.float32),           # 6  (dims 20-25)
    ]).ravel()[:STATE_DIM].astype(np.float32)


if __name__ == "__main__":
    mlp = ResidualMLP()
    s = np.zeros(STATE_DIM, dtype=np.float32)
    a = mlp.forward(s)
    assert a.shape == (ACTION_DIM,), f"Expected ({ACTION_DIM},), got {a.shape}"
    assert np.allclose(a, 0), f"Expected zero, got {a}"
    parts = unpack_action(a)
    assert parts["delta_pose"].shape == (6,)
    assert parts["delta_K"].shape == (6,)
    assert parts["delta_D"].shape == (6,)
    assert parts["delta_F"].shape == (6,)
    imp = compute_impedance(a)
    assert np.allclose(imp["stiffness"], K_BASE)
    assert np.allclose(imp["damping"], D_BASE)
    print(f"ResidualMLP V2 OK: state={STATE_DIM}D → action={ACTION_DIM}D")
    print(f"  Δpose bounds: ±{POS_BOUND*1000}mm / ±{np.degrees(ROT_BOUND):.1f}°")
    print(f"  ΔK bounds: ±{K_BOUND} N/m, ΔD bounds: ±{D_BOUND} Ns/m")
    print(f"  ΔF bounds: ±{F_BOUND_LIN}N / ±{F_BOUND_ROT}Nm")
