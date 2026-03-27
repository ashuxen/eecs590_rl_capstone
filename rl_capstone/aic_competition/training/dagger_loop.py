"""
DAgger + Impedance-ResiP Pipeline (V2)

Two modes of DAgger data collection:

  ONLINE (preferred, requires sim):
    Run SmartInsert with AIC_DAGGER_COLLECT=1 + ground_truth:=true.
    The policy executes its perception path but labels each step with
    the TF-derived expert target.  Data lands in ~/rl/dagger_data/round_XX/.

  OFFLINE (fast, no sim needed):
    Replay expert episodes from ~/aic_training_data, inject noise to
    simulate distribution shift, compute expert corrections at the
    noisy states.  Produces (state_26D, residual_24D) pairs.

After collection, supervised training produces a ResidualMLP checkpoint
that SmartInsert can load at inference time.

Usage:
  cd ~/rl

  # Offline DAgger (one round, episodes 0-52)
  python -m training.dagger_loop collect --round 1 \
      --data-dir ~/rl/diverse_training_data --episodes $(seq 0 52)

  # Retrain after collection
  python -m training.dagger_loop retrain --round 1

  # Full pipeline (3 rounds, offline)
  python -m training.dagger_loop full --total-rounds 3 \
      --data-dir ~/rl/diverse_training_data --episodes $(seq 0 52)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

rl_root = Path(__file__).resolve().parents[1]
if str(rl_root) not in sys.path:
    sys.path.insert(0, str(rl_root))


DAGGER_DATA_DIR = Path.home() / "rl" / "dagger_data"
RESIDUAL_DATA_DIR = Path.home() / "rl" / "residual_data"
ORIGINAL_DATA_DIR = Path.home() / "aic_training_data"
CHECKPOINT_DIR = Path.home() / "rl" / "yd_rrl_checkpoints"


def _load_episode(ep_dir: Path) -> dict | None:
    """Load and validate an episode's data.npz."""
    data_path = ep_dir / "data.npz"
    if not data_path.exists():
        return None
    d = dict(np.load(data_path, allow_pickle=False))
    required = [
        "tcp_position", "force", "torque", "z_offset",
        "port_position_gt", "insertion_axis", "expert_target",
    ]
    if not all(k in d for k in required):
        return None
    return d


def _read_connector_type(ep_dir: Path) -> bool:
    """Return True if connector is SFP, False for SC."""
    meta_path = ep_dir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            return "sfp" in str(meta.get("port_type", "sfp")).lower()
        except Exception:
            pass
    return True


def collect_dagger_data_offline(
    round_num: int,
    original_data_dir: Path,
    episodes: list[int],
    residual_checkpoint: str | None = None,
):
    """Offline DAgger: replay expert episodes with noise → V2 labels (26D/24D).

    For each expert episode, adds noise to the expert trajectory (simulating
    the learned policy's distribution shift), then computes:
      - 26D state (with base_action_local in last 6 dims)
      - 24D residual label (pose_6D + ΔK_6D + ΔD_6D + ΔF_6D)

    Impedance labels use force-reactive heuristics (same as SmartInsert).
    """
    from training.frame_decomposer import (
        yaw_from_insertion_axis,
        yaw_rotation_matrix,
        world_to_port_local_force_torque,
    )
    from training.residual_mlp import (
        ResidualMLP, build_yd_rrl_state,
        POS_BOUND, K_BOUND, D_BOUND, F_BOUND_LIN, F_BOUND_ROT,
        RESIDUAL_ALPHA,
    )

    round_dir = DAGGER_DATA_DIR / f"round_{round_num:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    residual = None
    if residual_checkpoint:
        ckpt_path = Path(os.path.expanduser(residual_checkpoint))
        if ckpt_path.exists():
            residual = ResidualMLP(checkpoint_path=str(ckpt_path))
            print(f"Loaded residual policy: {ckpt_path}")

    noise_scale = 0.002 * (1.0 / max(1, round_num))

    ep_count = 0
    for ep_num in episodes:
        ep_dir = original_data_dir / f"episode_{ep_num:04d}"
        if not ep_dir.exists():
            continue
        d = _load_episode(ep_dir)
        if d is None:
            continue

        connector_sfp = _read_connector_type(ep_dir)
        plug_xyz = d.get("plug_position_gt", d["tcp_position"].copy())
        if "plug_position_gt" not in d:
            plug_xyz[:, 2] -= 0.02

        n = len(d["tcp_position"])
        states_list, residuals_list, base_actions_list = [], [], []

        for i in range(n):
            ins_axis = d["insertion_axis"][i] if d["insertion_axis"].ndim > 1 else d["insertion_axis"]
            yaw = yaw_from_insertion_axis(ins_axis)
            R = yaw_rotation_matrix(yaw)

            port_xyz = d["port_position_gt"][i] if d["port_position_gt"].ndim > 1 else d["port_position_gt"]
            plug_pos = plug_xyz[i] if plug_xyz.ndim > 1 else plug_xyz
            z_off = float(d["z_offset"].ravel()[i])
            F = d["force"][i] if d["force"].ndim > 1 else d["force"]
            tau = d["torque"][i] if d["torque"].ndim > 1 else d["torque"]

            # Inject noise to simulate distribution shift
            noise = np.random.randn(3) * noise_scale
            noisy_tcp = d["tcp_position"][i] + noise

            # Pose error in world frame
            pose_error_world = np.concatenate([port_xyz[:3] - (plug_pos[:3] + noise), np.zeros(3)])

            # Force/torque in local frame
            F_local, tau_local = world_to_port_local_force_torque(F[:3], tau[:3], yaw)

            # Progress / contact / connector
            insertion_progress = np.clip(1.0 - (z_off + 0.015) / 0.215, 0.0, 1.0)
            force_mag = np.linalg.norm(F[:3])
            contact = 1.0 if force_mag > 2.0 else 0.0
            time_rem = 1.0 - (i / max(1, n - 1))

            # Base action in port-local frame (noisy_tcp → expert_target direction)
            expert_target_pos = d["expert_target"][i, :3] if d["expert_target"].ndim > 1 else d["expert_target"][:3]
            base_action_world = expert_target_pos - noisy_tcp + noise
            base_action_local = np.zeros(6, dtype=np.float32)
            base_action_local[:3] = R @ base_action_world[:3]

            # Build 26D state (V2)
            state = build_yd_rrl_state(
                F[:3], tau[:3], pose_error_world, yaw,
                insertion_progress, contact, connector_sfp, time_rem,
                base_action_local=base_action_local,
                cable_tension_est=None,
            )

            # Compute learned policy's correction if available
            if residual is not None:
                learned_action = residual.forward(state) * RESIDUAL_ALPHA
                from training.frame_decomposer import port_local_to_world_delta
                res_world = port_local_to_world_delta(learned_action[:6], yaw)
                learned_target = noisy_tcp + res_world[:3]
            else:
                learned_target = noisy_tcp

            # -- Pose residual (6D): expert - learned in port-local frame --
            correction_world = expert_target_pos - learned_target
            pose_res_local = np.clip(R @ correction_world, -POS_BOUND, POS_BOUND).astype(np.float32)
            orient_res = np.zeros(3, dtype=np.float32)

            # -- Impedance heuristic labels --
            # ΔK: soften when force is high
            gain_k = 2.0
            delta_K = np.full(6, -gain_k * max(0.0, force_mag - 5.0), dtype=np.float32)
            delta_K = np.clip(delta_K, -K_BOUND, K_BOUND)

            # ΔD: increase damping near contact
            vel_mag = 0.0
            if "tcp_linear_velocity" in d:
                v = d["tcp_linear_velocity"]
                vi = v[i] if v.ndim > 1 else v
                vel_mag = float(np.linalg.norm(vi[:3]))
            gain_d = 5.0
            delta_D = np.full(6, gain_d * vel_mag * contact, dtype=np.float32)
            delta_D = np.clip(delta_D, -D_BOUND, D_BOUND)

            # ΔF: compensate undesired forces
            gain_f, gain_t = 0.3, 0.1
            delta_F = np.zeros(6, dtype=np.float32)
            delta_F[:3] = np.clip(-gain_f * F_local[:3], -F_BOUND_LIN, F_BOUND_LIN)
            delta_F[3:6] = np.clip(-gain_t * tau_local[:3], -F_BOUND_ROT, F_BOUND_ROT)

            residual_24d = np.concatenate([
                pose_res_local, orient_res,
                delta_K, delta_D, delta_F,
            ]).astype(np.float32)

            states_list.append(state)
            residuals_list.append(residual_24d)
            base_actions_list.append(base_action_local.copy())

        if states_list:
            out_path = round_dir / f"episode_{ep_count:04d}.npz"
            np.savez_compressed(
                out_path,
                states=np.array(states_list, dtype=np.float32),
                residuals=np.array(residuals_list, dtype=np.float32),
                base_actions=np.array(base_actions_list, dtype=np.float32),
            )
            ep_count += 1
            print(f"  DAgger round {round_num}, ep {ep_count}: {len(states_list)} steps from {ep_dir.name}")

    print(f"DAgger round {round_num}: {ep_count} episodes → {round_dir}")
    return round_dir


def aggregate_and_retrain(round_num: int, args):
    """Aggregate all DAgger rounds + original data, then retrain supervised."""
    agg_dir = DAGGER_DATA_DIR / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    for f in agg_dir.glob("episode_*.npz"):
        f.unlink()

    ep_count = 0

    # Include original residual data (if any)
    if RESIDUAL_DATA_DIR.exists():
        for f in sorted(RESIDUAL_DATA_DIR.glob("episode_*.npz")):
            shutil.copy(f, agg_dir / f"episode_{ep_count:04d}.npz")
            ep_count += 1
        if ep_count:
            print(f"Added {ep_count} episodes from original residual data")

    # Include all DAgger rounds up to current
    for r in range(1, round_num + 1):
        round_dir = DAGGER_DATA_DIR / f"round_{r:02d}"
        if not round_dir.exists():
            continue
        round_eps = sorted(round_dir.glob("episode_*.npz"))
        for f in round_eps:
            shutil.copy(f, agg_dir / f"episode_{ep_count:04d}.npz")
            ep_count += 1
        print(f"Added {len(round_eps)} episodes from DAgger round {r}")

    print(f"Total aggregated: {ep_count} episodes in {agg_dir}")

    if ep_count == 0:
        print("No data to train on.")
        return

    out_dir = CHECKPOINT_DIR / f"dagger_r{round_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use pixi environment for torch availability
    pixi_dir = Path.home() / "ws_aic" / "src" / "aic"
    if (pixi_dir / "pixi.toml").exists():
        cmd = [
            "pixi", "run", "python3", "-m", "training.train_residual_supervised",
            "--data-dir", str(agg_dir),
            "--out-dir", str(out_dir),
            "--epochs", str(getattr(args, "retrain_epochs", 100)),
            "--batch-size", "64",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(rl_root) + ":" + env.get("PYTHONPATH", "")
        print(f"\nTraining supervised residual (pixi): {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(pixi_dir), env=env, check=True)
    else:
        cmd = [
            sys.executable, "-m", "training.train_residual_supervised",
            "--data-dir", str(agg_dir),
            "--out-dir", str(out_dir),
            "--epochs", str(getattr(args, "retrain_epochs", 100)),
            "--batch-size", "64",
        ]
        print(f"\nTraining supervised residual: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(rl_root), check=True)

    print(f"\nDAgger round {round_num} complete.")
    print(f"Checkpoint: {out_dir / 'residual_mlp.pt'}")
    print(f"To use: export AIC_RESIDUAL_CHECKPOINT={out_dir}")


def full_dagger_pipeline(args):
    """Run the complete DAgger pipeline (3-5 rounds)."""
    episodes = args.episodes or list(range(53))

    for round_num in range(1, args.total_rounds + 1):
        print(f"\n{'='*60}")
        print(f"DAgger Round {round_num}/{args.total_rounds}")
        print(f"{'='*60}")

        ckpt = None
        if round_num > 1:
            prev_dir = CHECKPOINT_DIR / f"dagger_r{round_num - 1}"
            if (prev_dir / "residual_mlp.pt").exists():
                ckpt = str(prev_dir)

        collect_dagger_data_offline(
            round_num=round_num,
            original_data_dir=Path(os.path.expanduser(args.data_dir)),
            episodes=episodes,
            residual_checkpoint=ckpt,
        )

        aggregate_and_retrain(round_num, args)

    print(f"\n{'='*60}")
    print(f"DAgger pipeline complete ({args.total_rounds} rounds)")
    final = CHECKPOINT_DIR / f"dagger_r{args.total_rounds}"
    print(f"Final checkpoint: {final}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="DAgger + Impedance-ResiP Pipeline (V2)",
    )
    sub = parser.add_subparsers(dest="command", help="Sub-command")

    # --- collect ---
    collect_p = sub.add_parser("collect", help="Collect one offline DAgger round")
    collect_p.add_argument("--round", type=int, required=True)
    collect_p.add_argument("--data-dir", type=str, default="~/rl/diverse_training_data")
    collect_p.add_argument("--episodes", type=int, nargs="*", default=None)
    collect_p.add_argument("--residual-checkpoint", type=str, default=None)

    # --- retrain ---
    retrain_p = sub.add_parser("retrain", help="Aggregate and retrain after collection")
    retrain_p.add_argument("--round", type=int, required=True)
    retrain_p.add_argument("--retrain-epochs", type=int, default=100)

    # --- full ---
    full_p = sub.add_parser("full", help="Run complete offline DAgger pipeline")
    full_p.add_argument("--total-rounds", type=int, default=3)
    full_p.add_argument("--data-dir", type=str, default="~/rl/diverse_training_data")
    full_p.add_argument("--episodes", type=int, nargs="*", default=None)
    full_p.add_argument("--retrain-epochs", type=int, default=100)

    args = parser.parse_args()

    if args.command == "collect":
        episodes = args.episodes or list(range(53))
        collect_dagger_data_offline(
            round_num=args.round,
            original_data_dir=Path(os.path.expanduser(args.data_dir)),
            episodes=episodes,
            residual_checkpoint=args.residual_checkpoint,
        )
    elif args.command == "retrain":
        aggregate_and_retrain(args.round, args)
    elif args.command == "full":
        full_dagger_pipeline(args)
    else:
        parser.print_help()
        print("\nExample workflows:")
        print()
        print("  # Offline DAgger (3 rounds):")
        print("  python -m training.dagger_loop full --total-rounds 3 \\")
        print("      --data-dir ~/rl/diverse_training_data --episodes $(seq 0 52)")
        print()
        print("  # Online DAgger (in sim, step by step):")
        print("  # Terminal 1 (sim):")
        print("  distrobox enter --root aic_eval -- env RUST_LOG=zenoh=off \\")
        print("      AIC_DAGGER_COLLECT=1 AIC_DAGGER_ROUND=1 \\")
        print("      /entrypoint.sh ground_truth:=true start_aic_engine:=true")
        print("  # Terminal 2 (policy):")
        print("  cd ~/ws_aic/src/aic && pixi run aic_model")
        print("  # Then retrain:")
        print("  python -m training.dagger_loop retrain --round 1")


if __name__ == "__main__":
    main()
