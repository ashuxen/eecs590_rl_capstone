"""
PPO-trained Impedance-Aware Residual Policy (ResiP + DAgger Fallback).

Implements the hybrid architecture:
  PRIMARY:  PPO-trained residual that outputs Δpose + ΔK + ΔD + ΔF
  FALLBACK: DAgger-trained residual queried when PPO is uncertain

The PPO policy is an Actor-Critic with:
  - Actor:  Gaussian MLP (26D state → 24D action mean + log_std)
  - Critic: MLP (26D state → 1D value)

Training uses the existing diverse_training_data as a lightweight
contact-physics replay environment (no Gazebo needed for initial training).

At runtime, the PPO actor's entropy indicates confidence. When entropy
exceeds a threshold (model is confused), it falls back to the DAgger
policy — "reaching out to the expert in confusion."

Reference: ResiP paper (2407.16677v4), Section II-C.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.residual_mlp import (
    STATE_DIM, ACTION_DIM,
    ACTION_BOUNDS, RESIDUAL_ALPHA,
    K_BASE, D_BASE,
    POS_BOUND, ROT_BOUND, K_BOUND, D_BOUND,
    ResidualMLP, clip_action, compute_impedance,
)


class PPOActor(nn.Module):
    """Gaussian policy: state -> action mean + learned log_std.

    Architecture matches DAgger ResidualMLP exactly [128,128,128] so that
    warm-start from DAgger checkpoint is a direct weight copy.

    Layer mapping (DAgger fc -> PPO net + mean_head):
      DAgger fc.0  (26,128)  -> PPO net.0  (26,128)
      DAgger fc.2  (128,128) -> PPO net.2  (128,128)
      DAgger fc.4  (128,128) -> PPO net.4  (128,128)
      DAgger fc.6  (128,24)  -> PPO mean_head (128,24)
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, state: torch.Tensor):
        features = self.net(state)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        mean, std = self.forward(state)
        if deterministic:
            return mean, torch.zeros(1), torch.zeros(1)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class PPOCritic(nn.Module):
    """Value function: state -> scalar value."""

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class HybridResidualPolicy:
    """Runtime policy: PPO primary + DAgger fallback.

    The PPO actor outputs a correction at each timestep. If the actor's
    entropy exceeds `entropy_threshold` (the model is uncertain/confused),
    it falls back to the DAgger-trained residual — querying the "expert."

    This implements the user's insight: "use PPO primarily, but reach out
    to DAgger (the expert) when confused."
    """

    def __init__(
        self,
        ppo_checkpoint: Optional[str] = None,
        dagger_checkpoint: Optional[str] = None,
        entropy_threshold: float = 2.0,
        blend_alpha: float = 0.7,
    ):
        self.entropy_threshold = entropy_threshold
        self.blend_alpha = blend_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # PPO actor
        self.ppo_actor = PPOActor().to(self.device)
        self._ppo_loaded = False
        if ppo_checkpoint:
            ppo_path = Path(ppo_checkpoint).expanduser()
            if (ppo_path / "ppo_actor.pt").exists():
                state = torch.load(ppo_path / "ppo_actor.pt", map_location=self.device, weights_only=True)
                self.ppo_actor.load_state_dict(state)
                self.ppo_actor.eval()
                self._ppo_loaded = True

        # DAgger fallback
        self.dagger_residual = ResidualMLP(checkpoint_path=dagger_checkpoint)

        self._last_entropy = 0.0
        self._last_source = "ppo"

    def forward(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """state (26D) → action (24D), using PPO or DAgger fallback."""
        state = np.asarray(state, dtype=np.float32).ravel()[:STATE_DIM]

        if not self._ppo_loaded:
            action = self.dagger_residual.forward(state)
            self._last_source = "dagger_only"
            return action

        with torch.no_grad():
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            mean, std = self.ppo_actor(s_tensor)
            entropy = torch.distributions.Normal(mean, std).entropy().sum().item()
            self._last_entropy = entropy

            if deterministic:
                ppo_action = mean.squeeze(0).cpu().numpy()
            else:
                dist = torch.distributions.Normal(mean, std)
                ppo_action = dist.sample().squeeze(0).cpu().numpy()

        if entropy > self.entropy_threshold:
            # PPO is uncertain → blend with DAgger expert
            dagger_action = self.dagger_residual.forward(state)
            alpha = self.blend_alpha
            action = alpha * dagger_action + (1 - alpha) * ppo_action
            self._last_source = "blend"
        else:
            action = ppo_action
            self._last_source = "ppo"

        return clip_action(action)

    @property
    def last_entropy(self) -> float:
        return self._last_entropy

    @property
    def last_source(self) -> str:
        return self._last_source


# ---------------------------------------------------------------------------
# Contact-physics replay environment for PPO training
# ---------------------------------------------------------------------------

class InsertionReplayEnv:
    """Lightweight insertion environment that replays recorded episodes.

    Instead of running Gazebo (slow), we use the recorded force/torque and
    position data from diverse_training_data to create a semi-realistic
    training environment. The agent starts at a random offset from the
    expert trajectory and must correct to achieve insertion.

    Reward: sparse +1 for insertion (z_offset < -0.010 and low lateral force)
            -0.01 per step (time pressure)
            -0.1 for excessive force (> 10N lateral delta)
            +0.3 for reaching ALIGNMENT phase (intermediate reward)
    """

    def __init__(self, data_dir: str, episodes: list[int] | None = None):
        self.data_dir = Path(data_dir).expanduser()
        self.episodes_data = []
        self._load_episodes(episodes)
        self._current_ep = None
        self._step_idx = 0
        self._offset_noise = np.zeros(3)

    def _load_episodes(self, episodes: list[int] | None):
        import json
        ep_dirs = sorted(self.data_dir.glob("episode_*"))
        if episodes is not None:
            ep_dirs = [self.data_dir / f"episode_{e:04d}" for e in episodes
                       if (self.data_dir / f"episode_{e:04d}").exists()]

        for ep_dir in ep_dirs:
            data_path = ep_dir / "data.npz"
            if not data_path.exists():
                continue
            d = dict(np.load(data_path, allow_pickle=False))
            required = ["tcp_position", "force", "torque", "z_offset",
                        "port_position_gt", "insertion_axis", "expert_target"]
            if not all(k in d for k in required):
                continue

            meta_path = ep_dir / "metadata.json"
            is_sfp = True
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                is_sfp = "sfp" in str(meta.get("port_type", "sfp")).lower()

            self.episodes_data.append({"data": d, "is_sfp": is_sfp})

        print(f"InsertionReplayEnv: loaded {len(self.episodes_data)} episodes")

    def reset(self) -> np.ndarray:
        """Reset to a random episode at a random starting point with noise."""
        from training.residual_mlp import build_yd_rrl_state
        from training.frame_decomposer import yaw_from_insertion_axis

        idx = np.random.randint(len(self.episodes_data))
        self._current_ep = self.episodes_data[idx]
        d = self._current_ep["data"]
        n = len(d["tcp_position"])

        # Start at a random point in the descent phase (skip first 30% approach)
        self._step_idx = np.random.randint(int(n * 0.3), max(int(n * 0.3) + 1, n - 50))

        # Add lateral offset noise (3-8mm) to simulate perception error
        self._offset_noise = np.random.randn(3) * 0.004
        self._offset_noise[2] *= 0.3  # less noise along insertion axis

        return self._get_state()

    def step(self, action: np.ndarray):
        """Apply residual action, advance one step, return (state, reward, done, info)."""
        d = self._current_ep["data"]
        n = len(d["tcp_position"])

        action = clip_action(action)
        parts = dict(
            delta_pose=action[:6],
            delta_K=action[6:12],
            delta_D=action[12:18],
            delta_F=action[18:24],
        )

        # Simulate the effect of the residual correction on the offset
        self._offset_noise[:3] -= parts["delta_pose"][:3] * 0.5

        self._step_idx = min(self._step_idx + 1, n - 1)

        state = self._get_state()

        # Compute reward
        z_off = float(d["z_offset"].ravel()[self._step_idx])
        F = d["force"][self._step_idx] if d["force"].ndim > 1 else d["force"]
        lateral_f = float(np.linalg.norm(F[:2]))
        axial_f = float(abs(F[2]))
        offset_mag = float(np.linalg.norm(self._offset_noise[:2]))

        reward = -0.01  # time penalty

        # Insertion success: close to port with low lateral offset
        if z_off < -0.010 and offset_mag < 0.003:
            reward += 1.0
            done = True
        # Intermediate: got close to alignment
        elif offset_mag < 0.005 and z_off < 0.02:
            reward += 0.3
            done = False
        else:
            done = False

        # Force penalty
        if lateral_f > 10.0:
            reward -= 0.1

        # Episode timeout
        if self._step_idx >= n - 1:
            done = True

        info = {
            "z_offset": z_off,
            "offset_mag": offset_mag,
            "lateral_f": lateral_f,
        }
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        from training.residual_mlp import build_yd_rrl_state
        from training.frame_decomposer import yaw_from_insertion_axis

        d = self._current_ep["data"]
        i = self._step_idx
        n = len(d["tcp_position"])

        ins_axis = d["insertion_axis"][i] if d["insertion_axis"].ndim > 1 else d["insertion_axis"]
        yaw = yaw_from_insertion_axis(ins_axis)
        port_xyz = d["port_position_gt"][i] if d["port_position_gt"].ndim > 1 else d["port_position_gt"]
        noisy_tcp = d["tcp_position"][i] + self._offset_noise
        F = d["force"][i] if d["force"].ndim > 1 else d["force"]
        tau = d["torque"][i] if d["torque"].ndim > 1 else d["torque"]
        z_off = float(d["z_offset"].ravel()[i])

        pose_error = np.concatenate([port_xyz[:3] - noisy_tcp[:3], np.zeros(3)])
        insertion_progress = np.clip(1.0 - (z_off + 0.015) / 0.215, 0.0, 1.0)
        contact = 1.0 if np.linalg.norm(F[:3]) > 2.0 else 0.0
        time_rem = 1.0 - (i / max(1, n - 1))

        expert_pos = d["expert_target"][i, :3] if d["expert_target"].ndim > 1 else d["expert_target"][:3]
        base_action_world = expert_pos - noisy_tcp
        from training.frame_decomposer import yaw_rotation_matrix
        R = yaw_rotation_matrix(yaw)
        base_action_local = np.zeros(6, dtype=np.float32)
        base_action_local[:3] = R @ base_action_world[:3]

        return build_yd_rrl_state(
            F[:3], tau[:3], pose_error, yaw,
            insertion_progress, contact,
            self._current_ep["is_sfp"], time_rem,
            base_action_local=base_action_local,
        )


# ---------------------------------------------------------------------------
# PPO Training Loop
# ---------------------------------------------------------------------------

def train_ppo(
    data_dir: str = "~/rl/diverse_training_data",
    dagger_checkpoint: str | None = None,
    output_dir: str = "~/rl/yd_rrl_checkpoints/ppo_resip",
    episodes: list[int] | None = None,
    n_epochs: int = 200,
    steps_per_epoch: int = 2048,
    batch_size: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    kl_coef: float = 0.1,
    max_grad_norm: float = 0.5,
    n_update_epochs: int = 10,
):
    """Train the PPO residual with optional KL penalty against DAgger prior."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PPO ResiP Training — device={device}")

    # Environment
    if episodes is None:
        episodes = list(range(87))
    env = InsertionReplayEnv(data_dir, episodes)

    # Networks
    actor = PPOActor().to(device)
    critic = PPOCritic().to(device)

    # Warm-start actor from DAgger checkpoint (direct weight copy since
    # architectures now match: both [128,128,128] with same layer structure).
    if dagger_checkpoint:
        dagger_path = Path(dagger_checkpoint).expanduser()
        if (dagger_path / "residual_mlp.pt").exists():
            print(f"Warm-starting actor from DAgger: {dagger_path}")
            dagger_state = torch.load(
                dagger_path / "residual_mlp.pt", map_location=device, weights_only=True,
            )
            actor_state = actor.state_dict()
            # DAgger fc.{0,2,4}.{weight,bias} -> PPO net.{0,2,4}.{weight,bias}
            # DAgger fc.6.{weight,bias} -> PPO mean_head.{weight,bias}
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
                    else:
                        print(f"  Shape mismatch: {dk}{dagger_state[dk].shape} vs {ak}{actor_state[ak].shape}")
            actor.load_state_dict(actor_state)
            print(f"  Mapped {mapped}/8 layers from DAgger -> PPO actor")

    # DAgger prior for KL penalty (frozen)
    dagger_prior = None
    if dagger_checkpoint and kl_coef > 0:
        dagger_prior = ResidualMLP(checkpoint_path=dagger_checkpoint)
        print(f"DAgger prior loaded for KL penalty (coef={kl_coef})")

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
    )

    bounds_tensor = torch.from_numpy(ACTION_BOUNDS).float().to(device)

    best_reward = float("-inf")
    log_path = output_dir / "ppo_training.log"

    print(f"\nStarting PPO training: {n_epochs} epochs × {steps_per_epoch} steps")
    print(f"Output: {output_dir}\n")

    for epoch in range(n_epochs):
        # --- Rollout ---
        states_buf = []
        actions_buf = []
        log_probs_buf = []
        rewards_buf = []
        values_buf = []
        dones_buf = []

        state = env.reset()
        ep_reward = 0
        ep_count = 0
        ep_rewards = []

        actor.eval()
        critic.eval()

        for t in range(steps_per_epoch):
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                action, log_prob, entropy = actor.get_action(s_tensor, deterministic=False)
                value = critic(s_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            action_clipped = clip_action(action_np)

            next_state, reward, done, info = env.step(action_clipped)

            states_buf.append(state)
            actions_buf.append(action_clipped)
            log_probs_buf.append(log_prob.item())
            rewards_buf.append(reward)
            values_buf.append(value.item())
            dones_buf.append(done)

            ep_reward += reward
            state = next_state

            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0
                ep_count += 1
                state = env.reset()

        # Final value estimate for GAE
        with torch.no_grad():
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            last_value = critic(s_tensor).item()

        # --- Compute GAE ---
        advantages = np.zeros(steps_per_epoch, dtype=np.float32)
        returns = np.zeros(steps_per_epoch, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(steps_per_epoch)):
            if t == steps_per_epoch - 1:
                next_val = last_value
                next_done = 0
            else:
                next_val = values_buf[t + 1]
                next_done = dones_buf[t + 1]

            delta = rewards_buf[t] + gamma * next_val * (1 - dones_buf[t]) - values_buf[t]
            gae = delta + gamma * gae_lambda * (1 - dones_buf[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values_buf[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO Update ---
        states_t = torch.from_numpy(np.array(states_buf)).float().to(device)
        actions_t = torch.from_numpy(np.array(actions_buf)).float().to(device)
        old_log_probs_t = torch.tensor(log_probs_buf, dtype=torch.float32).to(device)
        advantages_t = torch.from_numpy(advantages).to(device)
        returns_t = torch.from_numpy(returns).to(device)

        actor.train()
        critic.train()

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0

        for _ in range(n_update_epochs):
            indices = np.random.permutation(steps_per_epoch)
            for start in range(0, steps_per_epoch, batch_size):
                end = min(start + batch_size, steps_per_epoch)
                idx = indices[start:end]

                b_states = states_t[idx]
                b_actions = actions_t[idx]
                b_old_lp = old_log_probs_t[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                new_log_probs, ent = actor.evaluate_actions(b_states, b_actions)
                values = critic(b_states)

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_ret)
                entropy_loss = -ent.mean()

                # KL penalty against DAgger prior (keeps PPO close to expert)
                kl_loss = torch.tensor(0.0, device=device)
                if dagger_prior is not None and kl_coef > 0:
                    with torch.no_grad():
                        dagger_actions = []
                        for s in b_states.cpu().numpy():
                            da = dagger_prior.forward(s)
                            dagger_actions.append(da)
                        dagger_t = torch.from_numpy(np.array(dagger_actions)).float().to(device)

                    mean, std = actor(b_states)
                    dist = torch.distributions.Normal(mean, std)
                    kl_loss = -dist.log_prob(dagger_t).sum(dim=-1).mean()

                loss = (policy_loss
                        + value_coef * value_loss
                        + entropy_coef * entropy_loss
                        + kl_coef * kl_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += ent.mean().item()
                total_kl += kl_loss.item()
                n_updates += 1

        # --- Logging ---
        avg_ep_reward = np.mean(ep_rewards) if ep_rewards else 0
        avg_policy_loss = total_policy_loss / max(1, n_updates)
        avg_value_loss = total_value_loss / max(1, n_updates)
        avg_entropy = total_entropy / max(1, n_updates)
        avg_kl = total_kl / max(1, n_updates)

        improved = avg_ep_reward > best_reward
        if improved:
            best_reward = avg_ep_reward
            torch.save(actor.state_dict(), output_dir / "ppo_actor.pt")
            torch.save(critic.state_dict(), output_dir / "ppo_critic.pt")

        log_line = (
            f"Ep {epoch+1:3d}/{n_epochs}  "
            f"reward={avg_ep_reward:.3f}  "
            f"episodes={ep_count}  "
            f"pi_loss={avg_policy_loss:.4f}  "
            f"v_loss={avg_value_loss:.4f}  "
            f"entropy={avg_entropy:.2f}  "
            f"kl={avg_kl:.4f}  "
            f"{'*BEST*' if improved else ''}"
        )

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            print(log_line)

        with open(log_path, "a") as f:
            f.write(log_line + "\n")

    print(f"\nPPO training complete. Best reward: {best_reward:.3f}")
    print(f"Checkpoint: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PPO ResiP Training")
    parser.add_argument("--data-dir", default="~/rl/diverse_training_data")
    parser.add_argument("--dagger-checkpoint", default=None,
                        help="DAgger checkpoint for warm-start and KL prior")
    parser.add_argument("--output-dir", default="~/rl/yd_rrl_checkpoints/ppo_resip")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-coef", type=float, default=0.1,
                        help="KL penalty against DAgger prior (0=no penalty)")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args()

    train_ppo(
        data_dir=args.data_dir,
        dagger_checkpoint=args.dagger_checkpoint,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
    )
