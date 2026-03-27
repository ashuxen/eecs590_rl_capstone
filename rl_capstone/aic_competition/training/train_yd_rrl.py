"""
Train ResiDAgger Impedance Residual with PPO (upgraded from SAC)

Architecture per ResiP (2407.16677) + our impedance innovation:
  - PPO (on-policy, per ResiP) instead of SAC
  - Orthogonal init with small final-layer gain
  - State = [s_t, a_base_t] (26D)
  - Action = [Δpose, ΔK, ΔD, ΔF] (24D)

Usage (offline, no sim):
  cd ~/ws_aic/src/aic
  PYTHONPATH=~/rl:$PYTHONPATH pixi run python -m training.train_yd_rrl \
      --mode offline --total-timesteps 200000

Usage (online, requires Gazebo):
  PYTHONPATH=~/rl:$PYTHONPATH pixi run python -m training.train_yd_rrl \
      --mode online --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

rl_root = Path(__file__).resolve().parents[1]
if str(rl_root) not in sys.path:
    sys.path.insert(0, str(rl_root))


def make_offline_env(**kwargs):
    from training.yd_rrl_env import ImpedanceResidualOfflineEnv
    return ImpedanceResidualOfflineEnv(**kwargs)


def make_online_env(**kwargs):
    from training.yd_rrl_env import ImpedanceResidualOnlineEnv
    return ImpedanceResidualOnlineEnv(**kwargs)


def build_policy_kwargs():
    """MLP [128,128,128] with orthogonal init (per ResiP Section II-C)."""
    import torch.nn as nn

    return dict(
        net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
        activation_fn=nn.ReLU,
        ortho_init=True,
    )


def train_offline(args):
    """PPO training on offline expert episodes."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    print("=" * 60)
    print("ResiDAgger Impedance Residual — PPO Training")
    print("=" * 60)

    env = make_offline_env(
        data_dir=args.data_dir,
        episodes=args.episodes,
        contact_phase_only=True,
        max_steps_per_episode=args.max_episode_steps,
    )
    eval_env = make_offline_env(
        data_dir=args.data_dir,
        episodes=args.episodes,
        contact_phase_only=True,
        max_steps_per_episode=args.max_episode_steps,
    )

    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=build_policy_kwargs(),
        verbose=1,
        device=args.device,
    )

    try:
        import tensorboard  # noqa: F401
        ppo_kwargs["tensorboard_log"] = str(out_dir / "tb_logs")
    except ImportError:
        pass

    if args.pretrained and Path(args.pretrained).exists():
        print(f"Loading pretrained: {args.pretrained}")
        model = PPO.load(args.pretrained, env=env)
    else:
        model = PPO(**ppo_kwargs)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.total_timesteps // 10),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="resid_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best_model"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(1, args.total_timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
    )

    from training.residual_mlp import STATE_DIM, ACTION_DIM
    print(f"\nTraining PPO for {args.total_timesteps} timesteps...")
    print(f"  State: {STATE_DIM}D (port-local + base action)")
    print(f"  Action: {ACTION_DIM}D (Δpose + ΔK + ΔD + ΔF)")
    print(f"  LR: {args.lr}, Gamma: {args.gamma}")
    print(f"  n_steps: {args.n_steps}, batch: {args.batch_size}, epochs: {args.n_epochs}")
    print(f"  Output: {out_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        log_interval=10,
    )

    final_path = out_dir / "final_model"
    model.save(str(final_path))
    print(f"\nSaved: {final_path}.zip")

    export_to_residual_mlp(model, out_dir / "residual_export")

    env.close()
    eval_env.close()
    return model


def train_online(args):
    """PPO fine-tuning in live Gazebo."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    print("=" * 60)
    print("ResiDAgger — Online PPO (live Gazebo)")
    print("=" * 60)

    env = make_online_env(use_ground_truth=True, max_steps=args.max_episode_steps)
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pretrained and Path(args.pretrained).exists():
        model = PPO.load(args.pretrained, env=env)
    else:
        ppo_kwargs = dict(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            policy_kwargs=build_policy_kwargs(),
            verbose=1,
            device=args.device,
        )
        model = PPO(**ppo_kwargs)

    cb = CheckpointCallback(
        save_freq=max(1, args.total_timesteps // 10),
        save_path=str(out_dir / "checkpoints"),
        name_prefix="resid_online",
    )
    model.learn(total_timesteps=args.total_timesteps, callback=[cb], log_interval=1)
    model.save(str(out_dir / "final_model"))
    export_to_residual_mlp(model, out_dir / "residual_export")
    env.close()


def export_to_residual_mlp(model, export_dir: Path):
    """Extract PPO actor weights into ResidualMLP-compatible checkpoint.

    PPO stores the actor as:
      mlp_extractor.policy_net: Sequential(Linear, ReLU, Linear, ReLU, Linear, ReLU)
      action_net: Linear(128, action_dim)
    Our ResidualMLP stores everything inside fc: Sequential(Linear, ReLU, ..., Linear).
    We map layer-by-layer in order (weight then bias).
    """
    import torch
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    from training.residual_mlp import STATE_DIM, ACTION_DIM

    try:
        from training.residual_mlp import ResidualMLP
        target = ResidualMLP.build_torch_model_with_ortho_init(STATE_DIM, ACTION_DIM)
        target_sd = target.state_dict()

        policy_net = model.policy.mlp_extractor.policy_net
        action_net = model.policy.action_net

        src_params = []
        for module in policy_net:
            if hasattr(module, "weight"):
                src_params.append(("weight", module.weight.data.clone()))
                src_params.append(("bias", module.bias.data.clone()))
        src_params.append(("weight", action_net.weight.data.clone()))
        src_params.append(("bias", action_net.bias.data.clone()))

        target_keys = list(target_sd.keys())
        mapped = 0
        for key, (_, param) in zip(target_keys, src_params):
            if target_sd[key].shape == param.shape:
                target_sd[key] = param
                mapped += 1
            else:
                print(f"  Shape mismatch: {key} target={target_sd[key].shape} src={param.shape}")

        target.load_state_dict(target_sd)
        torch.save(target_sd, export_dir / "residual_mlp.pt")
        print(f"Exported {mapped} params → {export_dir / 'residual_mlp.pt'}")
        print(f"  Use: export AIC_RESIDUAL_CHECKPOINT={export_dir}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Warning: Export failed: {e}")
        print("  Manual extraction may be needed from the .zip model.")


def main():
    parser = argparse.ArgumentParser(
        description="Train ResiDAgger Impedance Residual with PPO",
    )
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--data-dir", type=str, default="~/aic_training_data")
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--out-dir", type=str, default="~/rl/yd_rrl_checkpoints/ppo_impedance")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout steps")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO update epochs per rollout")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.mode == "offline":
        train_offline(args)
    else:
        train_online(args)


if __name__ == "__main__":
    main()
