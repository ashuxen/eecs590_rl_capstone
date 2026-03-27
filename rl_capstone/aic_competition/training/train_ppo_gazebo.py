"""
Batch PPO Trainer for Gazebo-collected trajectories.

Loads on-policy trajectory data collected by SmartInsert (AIC_PPO_TRAIN=1)
from ~/rl/ppo_training_data/iter_N/ and performs PPO gradient updates.

Each trajectory file (episode_XXXX.npz) contains:
  states:    (T, 26) float32
  actions:   (T, 24) float32
  log_probs: (T,)    float32
  values:    (T,)    float32
  rewards:   (T,)    float32
  z_offsets: (T,)    float32
  success:   (1,)    float32

Usage:
  cd ~/ws_aic/src/aic
  PYTHONPATH=~/rl:$PYTHONPATH pixi run python -m training.train_ppo_gazebo \
      --data-dir ~/rl/ppo_training_data/iter_000 \
      --dagger-checkpoint ~/rl/yd_rrl_checkpoints/dagger_r3

Reference: ResiP paper (2407.16677v4), PPO (Schulman et al., 2017)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rl_root = Path(__file__).resolve().parents[1]
if str(rl_root) not in sys.path:
    sys.path.insert(0, str(rl_root))

from training.ppo_residual import PPOActor, PPOCritic
from training.residual_mlp import (
    STATE_DIM, ACTION_DIM, ACTION_BOUNDS,
    ResidualMLP, clip_action,
)


def load_trajectories(data_dir: Path) -> list[dict]:
    """Load all episode_*.npz files from a directory."""
    trajectories = []
    for ep_path in sorted(data_dir.glob("episode_*.npz")):
        d = dict(np.load(ep_path, allow_pickle=False))
        required = ["states", "actions", "log_probs", "values", "rewards"]
        if not all(k in d for k in required):
            print(f"  Skipping {ep_path.name}: missing keys {set(required) - set(d.keys())}")
            continue
        trajectories.append(d)
    return trajectories


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Returns (advantages, returns) arrays of shape (T,).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def warm_start_from_dagger(actor: PPOActor, dagger_path: Path, device: torch.device) -> int:
    """Load DAgger weights into PPO actor. Returns number of mapped layers."""
    if not (dagger_path / "residual_mlp.pt").exists():
        return 0

    dagger_state = torch.load(
        dagger_path / "residual_mlp.pt", map_location=device, weights_only=True,
    )
    actor_state = actor.state_dict()
    DAGGER_TO_PPO = {
        "fc.0.weight": "net.0.weight", "fc.0.bias": "net.0.bias",
        "fc.2.weight": "net.2.weight", "fc.2.bias": "net.2.bias",
        "fc.4.weight": "net.4.weight", "fc.4.bias": "net.4.bias",
        "fc.6.weight": "mean_head.weight", "fc.6.bias": "mean_head.bias",
    }
    mapped = 0
    for dk, ak in DAGGER_TO_PPO.items():
        if dk in dagger_state and ak in actor_state:
            if dagger_state[dk].shape == actor_state[ak].shape:
                actor_state[ak] = dagger_state[dk]
                mapped += 1
    actor.load_state_dict(actor_state)
    return mapped


def export_to_residual_mlp(actor: PPOActor, export_dir: Path):
    """Export PPO actor weights to ResidualMLP-compatible format."""
    export_dir.mkdir(parents=True, exist_ok=True)

    PPO_TO_DAGGER = {
        "net.0.weight": "fc.0.weight", "net.0.bias": "fc.0.bias",
        "net.2.weight": "fc.2.weight", "net.2.bias": "fc.2.bias",
        "net.4.weight": "fc.4.weight", "net.4.bias": "fc.4.bias",
        "mean_head.weight": "fc.6.weight", "mean_head.bias": "fc.6.bias",
    }

    target = ResidualMLP._build_torch_model(STATE_DIM, ACTION_DIM)
    target_sd = target.state_dict()
    actor_sd = actor.state_dict()

    for ppo_k, dag_k in PPO_TO_DAGGER.items():
        if ppo_k in actor_sd and dag_k in target_sd:
            target_sd[dag_k] = actor_sd[ppo_k]

    torch.save(target_sd, export_dir / "residual_mlp.pt")
    print(f"Exported PPO actor -> {export_dir / 'residual_mlp.pt'}")


def train_ppo_batch(
    data_dir: str,
    output_dir: str = "~/rl/yd_rrl_checkpoints/ppo_resip",
    dagger_checkpoint: str | None = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    kl_coef: float = 0.05,
    max_grad_norm: float = 0.5,
    n_update_epochs: int = 10,
    batch_size: int = 256,
):
    """Run PPO gradient updates on Gazebo-collected trajectories."""

    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Batch PPO Trainer (Gazebo trajectories)")
    print("=" * 60)
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")

    # Load trajectories
    trajectories = load_trajectories(data_dir)
    if not trajectories:
        print(f"ERROR: No trajectory files in {data_dir}")
        return

    total_steps = sum(len(t["rewards"]) for t in trajectories)
    n_success = sum(1 for t in trajectories if t.get("success", [0])[0] > 0.5)
    print(f"  Loaded {len(trajectories)} episodes, {total_steps} total steps")
    print(f"  Success rate: {n_success}/{len(trajectories)}")

    # Initialize networks
    actor = PPOActor().to(device)
    critic = PPOCritic().to(device)

    # Load existing PPO weights or warm-start from DAgger
    loaded_ppo = False
    if (output_dir / "ppo_actor.pt").exists():
        actor.load_state_dict(
            torch.load(output_dir / "ppo_actor.pt", map_location=device, weights_only=True)
        )
        loaded_ppo = True
        print(f"  Loaded existing PPO actor from {output_dir}")
    elif dagger_checkpoint:
        dagger_path = Path(dagger_checkpoint).expanduser()
        n = warm_start_from_dagger(actor, dagger_path, device)
        print(f"  Warm-started from DAgger ({n}/8 layers)")

    if (output_dir / "ppo_critic.pt").exists():
        critic.load_state_dict(
            torch.load(output_dir / "ppo_critic.pt", map_location=device, weights_only=True)
        )
        print(f"  Loaded existing PPO critic from {output_dir}")

    # DAgger prior for KL penalty
    dagger_prior = None
    if dagger_checkpoint and kl_coef > 0:
        dagger_path = Path(dagger_checkpoint).expanduser()
        if (dagger_path / "residual_mlp.pt").exists():
            dagger_prior = ResidualMLP(checkpoint_path=str(dagger_path))
            print(f"  DAgger prior loaded for KL penalty (coef={kl_coef})")

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
    )

    # Aggregate all trajectories into flat buffers
    all_states = []
    all_actions = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []

    for traj in trajectories:
        states = traj["states"]
        actions = traj["actions"]
        old_lps = traj["log_probs"]
        values = traj["values"]
        rewards = traj["rewards"]

        advantages, returns = compute_gae(rewards, values, gamma, gae_lambda)

        all_states.append(states)
        all_actions.append(actions)
        all_old_log_probs.append(old_lps)
        all_advantages.append(advantages)
        all_returns.append(returns)

    states_t = torch.from_numpy(np.concatenate(all_states)).float().to(device)
    actions_t = torch.from_numpy(np.concatenate(all_actions)).float().to(device)
    old_lps_t = torch.from_numpy(np.concatenate(all_old_log_probs)).float().to(device)
    advantages_t = torch.from_numpy(np.concatenate(all_advantages)).float().to(device)
    returns_t = torch.from_numpy(np.concatenate(all_returns)).float().to(device)

    # Normalize advantages
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = len(states_t)
    print(f"\n  Training on {N} timesteps ({len(trajectories)} episodes)")
    print(f"  Epochs: {n_update_epochs}, batch: {batch_size}, lr: {lr}")
    print(f"  clip={clip_eps}, ent={entropy_coef}, vf={value_coef}, kl={kl_coef}")

    # PPO gradient updates
    actor.train()
    critic.train()

    epoch_stats = []
    for epoch in range(n_update_epochs):
        indices = np.random.permutation(N)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = torch.from_numpy(indices[start:end]).long().to(device)

            b_states = states_t[idx]
            b_actions = actions_t[idx]
            b_old_lp = old_lps_t[idx]
            b_adv = advantages_t[idx]
            b_ret = returns_t[idx]

            new_log_probs, ent = actor.evaluate_actions(b_states, b_actions)
            values = critic(b_states)

            # Clipped surrogate objective
            ratio = torch.exp(new_log_probs - b_old_lp)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, b_ret)
            entropy_loss = -ent.mean()

            # KL penalty against DAgger prior
            kl_loss = torch.tensor(0.0, device=device)
            if dagger_prior is not None:
                with torch.no_grad():
                    dagger_actions = []
                    for s in b_states.cpu().numpy():
                        da = dagger_prior.forward(s)
                        dagger_actions.append(da)
                    dagger_t = torch.from_numpy(np.array(dagger_actions)).float().to(device)
                mean, std = actor(b_states)
                dist = torch.distributions.Normal(mean, std)
                kl_loss = -dist.log_prob(dagger_t).sum(dim=-1).mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                + entropy_coef * entropy_loss
                + kl_coef * kl_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_entropy += ent.mean().item()
            epoch_kl += kl_loss.item()
            n_batches += 1

        avg_pl = epoch_policy_loss / max(1, n_batches)
        avg_vl = epoch_value_loss / max(1, n_batches)
        avg_ent = epoch_entropy / max(1, n_batches)
        avg_kl = epoch_kl / max(1, n_batches)
        epoch_stats.append((avg_pl, avg_vl, avg_ent, avg_kl))

        print(
            f"  Epoch {epoch+1:2d}/{n_update_epochs}  "
            f"pi_loss={avg_pl:.4f}  v_loss={avg_vl:.4f}  "
            f"entropy={avg_ent:.2f}  kl={avg_kl:.4f}"
        )

    # Save updated weights
    torch.save(actor.state_dict(), output_dir / "ppo_actor.pt")
    torch.save(critic.state_dict(), output_dir / "ppo_critic.pt")
    print(f"\nSaved PPO actor+critic to {output_dir}")

    # Export to ResidualMLP format for deployment compatibility
    export_to_residual_mlp(actor, output_dir)

    # Log summary
    avg_reward = np.mean([t["rewards"].sum() for t in trajectories])
    log_path = output_dir / "training.log"
    with open(log_path, "a") as f:
        f.write(
            f"episodes={len(trajectories)} steps={total_steps} "
            f"success={n_success}/{len(trajectories)} "
            f"avg_reward={avg_reward:.3f} "
            f"final_pi_loss={epoch_stats[-1][0]:.4f} "
            f"final_v_loss={epoch_stats[-1][1]:.4f}\n"
        )
    print(f"  Avg episode reward: {avg_reward:.3f}")
    print(f"  Log: {log_path}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Batch PPO Trainer (Gazebo trajectories)")
    parser.add_argument("--data-dir", required=True, help="Directory with episode_*.npz files")
    parser.add_argument("--output-dir", default="~/rl/yd_rrl_checkpoints/ppo_resip")
    parser.add_argument("--dagger-checkpoint", default=None,
                        help="DAgger checkpoint for warm-start and KL prior")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    train_ppo_batch(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dagger_checkpoint=args.dagger_checkpoint,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        kl_coef=args.kl_coef,
        n_update_epochs=args.n_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
