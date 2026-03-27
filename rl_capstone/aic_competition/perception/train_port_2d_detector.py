"""
Port2DNet — Per-camera 2D keypoint detector + 3D triangulation.

Instead of regressing 3D coordinates directly (which can't generalize to unseen
board positions), this splits the problem:
  1. CNN predicts (u, v) pixel coordinates of the port in each camera image
  2. Known camera geometry triangulates (u,v) × 3 cameras → 3D position

The 2D detection task generalizes perfectly because the port looks the same
regardless of board position — only its pixel location changes.

Architecture: Shared ResNet-18 backbone processes each camera independently,
predicts (u, v, confidence) per camera via spatial softmax heatmap.

Training labels: 3D GT port position projected to 2D pixel coords using
camera intrinsics + TCP pose + URDF camera extrinsics.

Usage:
  cd ~/ws_aic/src/aic
  PYTHONPATH=~/rl pixi run python -m training.train_port_2d_detector
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Camera geometry (from URDF + Gazebo sensor config)
# ---------------------------------------------------------------------------

# Native K at 1152×1024
K_NATIVE = np.array([
    [1236.63, 0, 576],
    [0, 1236.63, 512],
    [0, 0, 1],
], dtype=np.float64)

IMG_H, IMG_W = 256, 288
SCALE_X = IMG_W / 1152.0
SCALE_Y = IMG_H / 1024.0
K_SCALED = K_NATIVE.copy()
K_SCALED[0] *= SCALE_X
K_SCALED[1] *= SCALE_Y

# Distance range where port is visible in cameras (empirically verified)
MIN_DIST_M = 0.06
MAX_DIST_M = 0.30


def _urdf_T(xyz, rpy):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


# tool0 → camera optical frame transforms (from URDF)
_T_tcp_mount = _urdf_T([0, 0, -0.0265], [0, 0, 0])
_T_mount = {
    'center': _urdf_T([0, -0.1077, -0.00719], [0, -1.30899630, 1.57079623]),
    'left':   _urdf_T([-0.09326, -0.053843, -0.007188], [0, -1.30899630, 0.523599027]),
    'right':  _urdf_T([0.09326, -0.053843, -0.007188], [0, -1.30899630, 2.61799343]),
}
_T_optical = _urdf_T([0, 0, 0], [-np.pi / 2, 0, -np.pi / 2])

CAM_TRANSFORMS = {}
for name in ['left', 'center', 'right']:
    CAM_TRANSFORMS[name] = _T_tcp_mount @ _T_mount[name] @ _T_optical


def project_3d_to_2d(port_3d, tcp_pos, tcp_quat_xyzw, cam_name):
    """Project a 3D world point to 2D pixel coords in the given camera.

    Returns (u, v, z_cam) or None if point is behind camera or outside image.
    """
    from scipy.spatial.transform import Rotation as R

    T_world_tcp = np.eye(4)
    T_world_tcp[:3, :3] = R.from_quat(tcp_quat_xyzw).as_matrix()
    T_world_tcp[:3, 3] = tcp_pos

    T_cam_world = np.linalg.inv(T_world_tcp @ CAM_TRANSFORMS[cam_name])
    p_cam = T_cam_world @ np.array([*port_3d, 1.0])

    if p_cam[2] < 0.01:
        return None

    u = K_SCALED[0, 0] * p_cam[0] / p_cam[2] + K_SCALED[0, 2]
    v = K_SCALED[1, 1] * p_cam[1] / p_cam[2] + K_SCALED[1, 2]

    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None

    return float(u), float(v), float(p_cam[2])


def triangulate_3_views(detections, tcp_pos, tcp_quat_xyzw):
    """Triangulate 3D position from 2D detections in multiple cameras.

    detections: dict {cam_name: (u, v)} for at least 2 cameras
    Returns: 3D position in world frame
    """
    from scipy.spatial.transform import Rotation as R

    T_world_tcp = np.eye(4)
    T_world_tcp[:3, :3] = R.from_quat(tcp_quat_xyzw).as_matrix()
    T_world_tcp[:3, 3] = tcp_pos

    A_rows = []
    for cam_name, (u, v) in detections.items():
        T_world_cam = T_world_tcp @ CAM_TRANSFORMS[cam_name]
        T_cam_world = np.linalg.inv(T_world_cam)
        P = K_SCALED @ T_cam_world[:3]  # 3×4 projection matrix

        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])

    A = np.array(A_rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


def weighted_triangulate(detections, sigmas, tcp_pos, tcp_quat_xyzw):
    """Uncertainty-weighted DLT triangulation.

    Each camera's rows in A are scaled by 1/σ so that higher-confidence
    cameras contribute more to the least-squares solution.
    """
    from scipy.spatial.transform import Rotation as R

    T_world_tcp = np.eye(4)
    T_world_tcp[:3, :3] = R.from_quat(tcp_quat_xyzw).as_matrix()
    T_world_tcp[:3, 3] = tcp_pos

    A_rows = []
    for cam_name, (u, v) in detections.items():
        T_world_cam = T_world_tcp @ CAM_TRANSFORMS[cam_name]
        T_cam_world = np.linalg.inv(T_world_cam)
        P = K_SCALED @ T_cam_world[:3]

        w = 1.0 / max(sigmas[cam_name], 1e-6)
        A_rows.append(w * (u * P[2] - P[0]))
        A_rows.append(w * (v * P[2] - P[1]))

    A = np.array(A_rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(src: Path, episodes: list[int], sample_every: int = 10,
                  cache_dir: Path | None = None):
    """Build training dataset: per-camera images + 2D pixel labels.

    Memory-efficient: two-pass approach.
      Pass 1: compute valid frame indices and 2D labels (no images loaded).
      Pass 2: write images to memory-mapped file one episode at a time.

    Returns dict with mmap images + in-RAM labels.
    """

    if cache_dir is None:
        cache_dir = Path(os.path.expanduser("~/.cache/aic_train_2d"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    positions_seen = {}
    valid_entries = []  # (ep_num, frame_idx, pos_id, {cam: (u_norm, v_norm)})
    total_frames = 0

    # Pass 1: find valid frames and compute 2D labels (no images)
    for ep_num in episodes:
        ep_dir = src / f"episode_{ep_num:04d}"
        if not ep_dir.exists():
            continue

        data = np.load(ep_dir / "data.npz", allow_pickle=False)
        port = data['port_position_gt'][0]
        tcp_positions = data['tcp_position']
        tcp_orientations = data['tcp_orientation']
        n = len(tcp_positions)

        pos_key = tuple(port.round(3))
        if pos_key not in positions_seen:
            positions_seen[pos_key] = len(positions_seen)
        pos_id = positions_seen[pos_key]

        ep_valid = 0
        for i in range(0, n, sample_every):
            total_frames += 1
            tcp_pos = tcp_positions[i]
            tcp_quat = tcp_orientations[i]
            dist = np.linalg.norm(tcp_pos - port)

            if dist < MIN_DIST_M or dist > MAX_DIST_M:
                continue

            projections = {}
            for cam in ['left', 'center', 'right']:
                result = project_3d_to_2d(port, tcp_pos, tcp_quat, cam)
                if result is None:
                    break
                projections[cam] = (result[0] / IMG_W, result[1] / IMG_H)

            if len(projections) < 3:
                continue

            valid_entries.append((ep_num, i, pos_id, projections,
                                 port.astype(np.float32),
                                 tcp_pos.copy(), tcp_quat.copy()))
            ep_valid += 1

        if ep_valid > 0:
            print(f"  ep_{ep_num:02d}: {ep_valid}/{n} valid frames, "
                  f"port={port.round(4)}, pos_id={pos_id}")

        del data
        gc.collect()

    n_valid = len(valid_entries)
    print(f"\nTotal: {n_valid}/{total_frames} valid frames, "
          f"{len(positions_seen)} unique positions")

    # Build label arrays (tiny, in RAM)
    uv_labels = {cam: np.zeros((n_valid, 2), dtype=np.float32) for cam in ['left', 'center', 'right']}
    port_3d = np.zeros((n_valid, 3), dtype=np.float32)
    position_ids = np.zeros(n_valid, dtype=np.int32)

    tcp_positions = np.zeros((n_valid, 3), dtype=np.float64)
    tcp_orientations = np.zeros((n_valid, 4), dtype=np.float64)

    for idx, (_, _, pos_id, projs, port, t_pos, t_quat) in enumerate(valid_entries):
        for cam in ['left', 'center', 'right']:
            uv_labels[cam][idx] = projs[cam]
        port_3d[idx] = port
        position_ids[idx] = pos_id
        tcp_positions[idx] = t_pos
        tcp_orientations[idx] = t_quat

    # Pass 2: write images to mmap (one episode at a time, ~1MB per frame)
    mmap_paths = {}
    for cam in ['left', 'center', 'right']:
        mp = cache_dir / f"imgs_{cam}.npy"
        mmap = np.memmap(mp, dtype=np.uint8, mode='w+',
                         shape=(n_valid, IMG_H, IMG_W, 3))
        mmap_paths[cam] = (mp, mmap)

    # Group entries by episode for efficient I/O
    from collections import defaultdict
    ep_groups = defaultdict(list)
    for global_idx, (ep_num, frame_idx, _, _, _, _, _) in enumerate(valid_entries):
        ep_groups[ep_num].append((global_idx, frame_idx))

    for ep_num, entries in sorted(ep_groups.items()):
        ep_dir = src / f"episode_{ep_num:04d}"
        frame_indices = [fi for _, fi in entries]
        global_indices = [gi for gi, _ in entries]

        for cam in ['left', 'center', 'right']:
            all_imgs = np.load(ep_dir / f"{cam}_images.npz")['images']
            for gi, fi in zip(global_indices, frame_indices):
                mmap_paths[cam][1][gi] = all_imgs[fi]
            del all_imgs

        gc.collect()

    # Flush and reopen as read-only
    imgs_mmap = {}
    for cam in ['left', 'center', 'right']:
        mp, mmap = mmap_paths[cam]
        mmap.flush()
        del mmap
        imgs_mmap[cam] = np.memmap(mp, dtype=np.uint8, mode='r',
                                   shape=(n_valid, IMG_H, IMG_W, 3))

    print(f"Cache written: {n_valid} frames × 3 cameras")

    result = {}
    for cam in ['left', 'center', 'right']:
        result[f'{cam}_imgs'] = imgs_mmap[cam]
        result[f'{cam}_uv'] = uv_labels[cam]
    result['port_3d'] = port_3d
    result['position_id'] = position_ids
    result['tcp_position'] = tcp_positions
    result['tcp_orientation'] = tcp_orientations

    return result, positions_seen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FEAT_H = IMG_H // 4   # 64
FEAT_W = IMG_W // 4   # 72


def _quat_to_rotmat(q):
    """Quaternion (B, 4) xyzw → rotation matrix (B, 3, 3). Differentiable."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], dim=-1).view(-1, 3, 3)


def build_model():
    """Port2DNet V3: FPN to H/4 + soft-argmax + uncertainty σ."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet18, ResNet18_Weights

    class Port2DNetV3(nn.Module):

        def __init__(self):
            super().__init__()
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1 = base.layer1   # 64ch,  H/4
            self.layer2 = base.layer2   # 128ch, H/8
            self.layer3 = base.layer3   # 256ch, H/16
            self.layer4 = base.layer4   # 512ch, H/32

            # FPN lateral 1×1 convs
            ch = 128
            self.lat4 = nn.Conv2d(512, ch, 1)
            self.lat3 = nn.Conv2d(256, ch, 1)
            self.lat2 = nn.Conv2d(128, ch, 1)
            self.lat1 = nn.Conv2d(64,  ch, 1)

            self.smooth = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )

            self.heatmap_head = nn.Sequential(
                nn.Conv2d(ch, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )

            self.log_temp = nn.Parameter(torch.tensor([-2.0]))

            xs = torch.linspace(0, 1, FEAT_W).view(1, 1, 1, FEAT_W)
            ys = torch.linspace(0, 1, FEAT_H).view(1, 1, FEAT_H, 1)
            self.register_buffer('coord_x', xs)
            self.register_buffer('coord_y', ys)

        def forward(self, x):
            """x: (B,3,H,W) → uv (B,2), sigma (B,), heatmap (B,1,Hf,Wf)."""
            x0 = self.stem(x)
            c1 = self.layer1(x0)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)

            p4 = self.lat4(c4)
            p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:],
                                                mode='bilinear', align_corners=False)
            p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:],
                                                mode='bilinear', align_corners=False)
            p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:],
                                                mode='bilinear', align_corners=False)

            feat = self.smooth(p1)                      # (B, 128, H/4, W/4)
            heatmap = self.heatmap_head(feat)            # (B, 1, Hf, Wf)

            B = heatmap.shape[0]
            temp = torch.exp(self.log_temp).clamp(0.01, 10.0)
            attn = F.softmax(heatmap.view(B, -1) / temp, dim=-1).view_as(heatmap)

            exp_u = (attn * self.coord_x).sum(dim=(2, 3)).squeeze(1)
            exp_v = (attn * self.coord_y).sum(dim=(2, 3)).squeeze(1)

            var_u = (attn * (self.coord_x - exp_u.view(B, 1, 1, 1)) ** 2).sum(dim=(2, 3)).squeeze(1)
            var_v = (attn * (self.coord_y - exp_v.view(B, 1, 1, 1)) ** 2).sum(dim=(2, 3)).squeeze(1)
            sigma = torch.sqrt(var_u + var_v + 1e-8)

            uv = torch.stack([exp_u, exp_v], dim=-1)
            return uv, sigma, heatmap

    return Port2DNetV3()


# ---------------------------------------------------------------------------
# Training dataset wrapper
# ---------------------------------------------------------------------------

class MultiViewDataset:
    """Per-frame dataset: returns all 3 camera images + labels + TCP pose.

    With augment=True, images get color/noise augmentation (but NOT spatial,
    so the camera geometry for consistency loss stays valid).
    """

    CAMS = ('left', 'center', 'right')

    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import torch
        real = self.indices[idx]

        result = {}
        for cam in self.CAMS:
            img = torch.from_numpy(
                self.dataset[f'{cam}_imgs'][real].copy()
            ).permute(2, 0, 1).float() / 255.0
            uv = torch.from_numpy(self.dataset[f'{cam}_uv'][real].copy())

            if self.augment:
                img = self._color_augment(img)

            result[f'{cam}_img'] = img
            result[f'{cam}_uv'] = uv

        result['tcp_pos'] = torch.from_numpy(
            self.dataset['tcp_position'][real].copy()).float()
        result['tcp_quat'] = torch.from_numpy(
            self.dataset['tcp_orientation'][real].copy()).float()
        return result

    @staticmethod
    def _color_augment(img):
        """Color-only augmentation (preserves pixel ↔ geometry mapping)."""
        import torch
        import torch.nn.functional as F

        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        mean = img.mean()
        img = ((img - mean) * contrast + mean) * brightness
        img = img.clamp(0, 1)

        if torch.rand(1).item() < 0.15:
            k = 3 if torch.rand(1).item() < 0.5 else 5
            pad = k // 2
            sigma = torch.empty(1).uniform_(0.5, 1.5).item()
            ax = torch.arange(k, dtype=img.dtype) - pad
            kernel_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
            kernel_2d = kernel_2d.expand(3, 1, k, k)
            img = F.conv2d(img.unsqueeze(0), kernel_2d, padding=pad,
                           groups=3).squeeze(0)

        img = (img + torch.randn_like(img) * 0.025).clamp(0, 1)
        return img


def _mv_collate(batch):
    """Custom collate for MultiViewDataset dicts."""
    import torch
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def compute_consistency_loss(preds, tcp_pos, tcp_quat, device):
    """Multi-view reprojection consistency loss (differentiable).

    Must be called outside autocast context (needs float32 for linalg.inv/SVD).
    """
    B = tcp_pos.shape[0]
    tcp_pos = tcp_pos.float()
    tcp_quat = tcp_quat.float()

    R = _quat_to_rotmat(tcp_quat)
    T_w_tcp = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    T_w_tcp[:, :3, :3] = R
    T_w_tcp[:, :3, 3] = tcp_pos

    K = torch.from_numpy(K_SCALED).float().to(device)

    Ps = {}
    for cam in ('left', 'center', 'right'):
        T_cam = torch.from_numpy(CAM_TRANSFORMS[cam]).float().to(device)
        T_w_cam = T_w_tcp @ T_cam.unsqueeze(0)
        T_cam_w = torch.linalg.inv(T_w_cam)
        Ps[cam] = K.unsqueeze(0) @ T_cam_w[:, :3, :]   # (B, 3, 4)

    # Differentiable DLT triangulation
    A_rows = []
    for cam in ('left', 'center', 'right'):
        u_px = preds[cam]['uv'][:, 0].float() * IMG_W
        v_px = preds[cam]['uv'][:, 1].float() * IMG_H
        P = Ps[cam]
        A_rows.append(u_px.unsqueeze(-1) * P[:, 2:3, :] - P[:, 0:1, :])
        A_rows.append(v_px.unsqueeze(-1) * P[:, 2:3, :] - P[:, 1:2, :])

    A = torch.cat(A_rows, dim=1)  # (B, 6, 4)
    _, _, Vh = torch.linalg.svd(A)
    X_h = Vh[:, -1, :]
    X_3d = X_h[:, :3] / X_h[:, 3:4]

    # Reprojection error (plain L2 -- no sigma weighting to avoid collapse)
    X_full = torch.cat([X_3d, torch.ones(B, 1, device=device)], dim=-1)
    loss = torch.tensor(0.0, device=device)
    for cam in ('left', 'center', 'right'):
        proj = torch.bmm(Ps[cam], X_full.unsqueeze(-1)).squeeze(-1)
        u_rep = proj[:, 0] / proj[:, 2] / IMG_W
        v_rep = proj[:, 1] / proj[:, 2] / IMG_H
        pred_uv = preds[cam]['uv'].float()
        err = (pred_uv[:, 0] - u_rep) ** 2 + (pred_uv[:, 1] - v_rep) ** 2
        loss = loss + err.mean()

    return loss / 3.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.expanduser("~/rl/diverse_training_data"))
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(87)))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--lambda-consist", type=float, default=0.05,
                        help="Weight for multi-view consistency loss")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    src = Path(args.src).expanduser()
    out = Path(args.out or os.path.expanduser("~/rl/perception_checkpoints/port_2d_v3"))
    out.mkdir(parents=True, exist_ok=True)

    print("Building dataset with 2D pixel labels...")
    dataset, pos_map = build_dataset(src, args.episodes, args.sample_every)

    n = len(dataset['port_3d'])
    print(f"\nDataset: {n} samples, {len(pos_map)} positions")

    # Position-stratified val split (hold out ~20% of positions)
    rng = np.random.RandomState(42)
    unique_pos_ids = sorted(pos_map.values())
    n_val_pos = max(1, len(unique_pos_ids) // 5)
    val_pos_ids = set(rng.choice(unique_pos_ids, n_val_pos, replace=False))

    pos_ids = dataset['position_id']
    val_mask = np.array([pid in val_pos_ids for pid in pos_ids])
    train_mask = ~val_mask

    val_positions = [k for k, v in pos_map.items() if v in val_pos_ids]
    print(f"Val positions ({n_val_pos}): {val_positions}")
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}")

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    train_ds = MultiViewDataset(dataset, train_idx, augment=True)
    val_ds = MultiViewDataset(dataset, val_idx, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, collate_fn=_mv_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0,
                        collate_fn=_mv_collate)
    print(f"Train frames: {len(train_ds)}, Val frames: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    if args.resume:
        ckpt = Path(args.resume)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Resumed from {ckpt}")
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_p:,} params (FPN H/4 + soft-argmax + σ)")
    print(f"Feature map: {FEAT_H}×{FEAT_W}, device={device}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=len(train_dl), pct_start=0.1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_px = float("inf")
    patience = 0
    PATIENCE = 60
    CAMS = ('left', 'center', 'right')

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        t_coord, t_consist, n_batch = 0.0, 0.0, 0
        for batch in train_dl:
            B = batch['left_img'].shape[0]
            all_imgs = torch.cat([batch[f'{c}_img'] for c in CAMS]).to(device)
            all_gt = torch.cat([batch[f'{c}_uv'] for c in CAMS]).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                all_uv, all_sigma, _ = model(all_imgs)
                coord_loss = F.smooth_l1_loss(all_uv, all_gt)

            # Consistency loss in float32 (linalg.inv needs fp32)
            # Warmup: only enable after epoch 30 when coordinates are reasonable
            CONSIST_WARMUP = 30
            if epoch >= CONSIST_WARMUP:
                preds = {}
                for i, cam in enumerate(CAMS):
                    preds[cam] = {
                        'uv': all_uv[i * B:(i + 1) * B].float(),
                        'sigma': all_sigma[i * B:(i + 1) * B].float(),
                    }
                consist_loss = compute_consistency_loss(
                    preds,
                    batch['tcp_pos'].to(device),
                    batch['tcp_quat'].to(device),
                    device,
                )
                loss = coord_loss + args.lambda_consist * consist_loss
            else:
                consist_loss = torch.tensor(0.0, device=device)
                loss = coord_loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            t_coord += coord_loss.item()
            t_consist += consist_loss.item()
            n_batch += 1

        t_coord /= max(1, n_batch)
        t_consist /= max(1, n_batch)

        # --- Val ---
        model.eval()
        all_pred, all_gt_list = [], []
        with torch.no_grad():
            for batch in val_dl:
                B = batch['left_img'].shape[0]
                all_imgs = torch.cat([batch[f'{c}_img'] for c in CAMS]).to(device)
                all_gt_uv = torch.cat([batch[f'{c}_uv'] for c in CAMS])

                with torch.amp.autocast("cuda", enabled=use_amp):
                    uv_pred, _, _ = model(all_imgs)

                all_pred.append(uv_pred.float().cpu().numpy())
                all_gt_list.append(all_gt_uv.numpy())

        pred = np.concatenate(all_pred)
        gt = np.concatenate(all_gt_list)
        err_u = np.abs(pred[:, 0] - gt[:, 0]) * IMG_W
        err_v = np.abs(pred[:, 1] - gt[:, 1]) * IMG_H
        err_px = np.sqrt(err_u ** 2 + err_v ** 2)
        mean_px = err_px.mean()

        improved = mean_px < best_px
        if improved:
            best_px = mean_px
            torch.save(model.state_dict(), out / "port_2d.pt")
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            lr = opt.param_groups[0]['lr']
            print(
                f"Ep {epoch + 1:3d}/{args.epochs}  "
                f"coord={t_coord:.6f}  consist={t_consist:.4f}  "
                f"px={mean_px:.1f}+-{err_px.std():.1f}  "
                f"u={err_u.mean():.1f}  v={err_v.mean():.1f}  "
                f"med={np.median(err_px):.1f}  "
                f"lr={lr:.2e}  "
                f"{'*BEST*' if improved else ''}"
            )

        if patience >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # -----------------------------------------------------------------------
    # Final evaluation: uncertainty-weighted triangulation
    # -----------------------------------------------------------------------
    print("\n--- Final evaluation: uncertainty-weighted triangulation ---")
    model.load_state_dict(
        torch.load(out / "port_2d.pt", map_location=device, weights_only=True)
    )
    model.eval()

    val_idx_arr = np.where(val_mask)[0]
    tri_errors_mm = []
    tri_weighted_mm = []
    px_errors_final = []

    with torch.no_grad():
        for idx in val_idx_arr:
            tcp_pos = dataset['tcp_position'][idx]
            tcp_quat = dataset['tcp_orientation'][idx]
            port_gt = dataset['port_3d'][idx]

            detections = {}
            sigmas = {}
            for cam in CAMS:
                img = torch.from_numpy(
                    dataset[f'{cam}_imgs'][idx].copy()
                ).permute(2, 0, 1).float().unsqueeze(0) / 255.0

                with torch.amp.autocast("cuda", enabled=use_amp):
                    uv_pred, sigma, _ = model(img.to(device))
                uv_np = uv_pred.float().cpu().numpy()[0]
                sig_np = sigma.float().cpu().item()

                u_px = uv_np[0] * IMG_W
                v_px = uv_np[1] * IMG_H
                detections[cam] = (u_px, v_px)
                sigmas[cam] = sig_np

                gt_uv = dataset[f'{cam}_uv'][idx]
                px_errors_final.append(np.sqrt(
                    (u_px - gt_uv[0] * IMG_W) ** 2 +
                    (v_px - gt_uv[1] * IMG_H) ** 2))

            # Standard triangulation
            port_pred = triangulate_3_views(detections, tcp_pos, tcp_quat)
            tri_errors_mm.append(np.linalg.norm(port_pred - port_gt) * 1000)

            # Uncertainty-weighted triangulation
            port_w = weighted_triangulate(detections, sigmas, tcp_pos, tcp_quat)
            tri_weighted_mm.append(np.linalg.norm(port_w - port_gt) * 1000)

    tri_errors_mm = np.array(tri_errors_mm)
    tri_weighted_mm = np.array(tri_weighted_mm)
    px_errors_final = np.array(px_errors_final)

    print(f"  Val frames: {len(val_idx_arr)}")
    print(f"  2D pixel error:  {px_errors_final.mean():.2f} +- "
          f"{px_errors_final.std():.2f} px  "
          f"(median {np.median(px_errors_final):.2f})")
    print(f"  3D equal-weight:  {tri_errors_mm.mean():.2f} +- "
          f"{tri_errors_mm.std():.2f} mm  "
          f"(median {np.median(tri_errors_mm):.2f})")
    print(f"  3D σ-weighted:    {tri_weighted_mm.mean():.2f} +- "
          f"{tri_weighted_mm.std():.2f} mm  "
          f"(median {np.median(tri_weighted_mm):.2f})")
    for thresh in (1, 2, 3, 5):
        pct = (tri_weighted_mm < thresh).mean() * 100
        print(f"  3D <{thresh}mm: {(tri_weighted_mm < thresh).sum()}"
              f"/{len(tri_weighted_mm)} ({pct:.0f}%)")

    # Save artifacts
    np.savez(
        out / "camera_geometry.npz",
        K_scaled=K_SCALED,
        T_tcp_left=CAM_TRANSFORMS['left'],
        T_tcp_center=CAM_TRANSFORMS['center'],
        T_tcp_right=CAM_TRANSFORMS['right'],
        img_h=IMG_H, img_w=IMG_W,
    )
    all_axes = []
    for ep_num in args.episodes:
        ep_dir = src / f"episode_{ep_num:04d}"
        if not ep_dir.exists():
            continue
        d = np.load(ep_dir / "data.npz", allow_pickle=False)
        all_axes.append(d['insertion_axis'][0])
    mean_axis = np.mean(all_axes, axis=0)
    mean_axis /= np.linalg.norm(mean_axis)
    np.save(out / "mean_axis.npy", mean_axis)

    print(f"\nSaved: {out}")
    print(f"  port_2d.pt — V3 detector (best={best_px:.2f}px)")
    print(f"  camera_geometry.npz — K + cam transforms")
    print(f"  mean_axis.npy — insertion axis")
    return 0


if __name__ == "__main__":
    exit(main())
