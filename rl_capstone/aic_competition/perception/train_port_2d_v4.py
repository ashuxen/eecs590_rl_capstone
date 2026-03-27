"""
Port2DNet V4 — Per-camera 2D keypoint detector + 3D triangulation.

Key improvements over V3:
  1. Spatial augmentation restored (translation, rotation, scale) with label
     adjustment — the port appearance is invariant, the model must learn to
     find it at any pixel location.
  2. Task-type conditioning (SFP=0, SC=1) via FiLM modulation — at runtime
     task.port_type tells us which port to detect; the model learns different
     visual patterns for each type.
  3. Per-camera training (dropped broken consistency loss) — simpler, more
     effective, each camera view is an independent training sample.
  4. FPN + spatial softmax architecture retained from V3 for subpixel precision.

Usage:
  cd ~/ws_aic/src/aic
  PYTHONPATH=~/rl pixi run python -m training.train_port_2d_v4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Camera geometry (from URDF + Gazebo sensor config)
# ---------------------------------------------------------------------------

K_NATIVE = np.array([
    [1236.63, 0, 576],
    [0, 1236.63, 512],
    [0, 0, 1],
], dtype=np.float64)

# Resolution stored on disk (data_collector saves at 0.25x)
STORED_H, STORED_W = 256, 288

# Model input resolution — set via --img-scale (default 0.25 → 288×256)
IMG_H, IMG_W = 256, 288   # overridden by _set_resolution()
K_SCALED = K_NATIVE.copy()

MIN_DIST_M = 0.03
MAX_DIST_M = 0.50


def _set_resolution(scale: float):
    """Recalculate global resolution constants for the given scale."""
    global IMG_H, IMG_W, K_SCALED, FEAT_H, FEAT_W
    IMG_W = int(1152 * scale)
    IMG_H = int(1024 * scale)
    K_SCALED = K_NATIVE.copy()
    K_SCALED[0] *= IMG_W / 1152.0
    K_SCALED[1] *= IMG_H / 1024.0
    FEAT_H = IMG_H // 4
    FEAT_W = IMG_W // 4

CAMS = ('left', 'center', 'right')
PORT_TYPES = {'sfp': 0, 'sc': 1}
NUM_PORT_TYPES = 2


def _urdf_T(xyz, rpy):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


_T_tcp_mount = _urdf_T([0, 0, -0.0265], [0, 0, 0])
_T_mount = {
    'center': _urdf_T([0, -0.1077, -0.00719], [0, -1.30899630, 1.57079623]),
    'left':   _urdf_T([-0.09326, -0.053843, -0.007188], [0, -1.30899630, 0.523599027]),
    'right':  _urdf_T([0.09326, -0.053843, -0.007188], [0, -1.30899630, 2.61799343]),
}
_T_optical = _urdf_T([0, 0, 0], [-np.pi / 2, 0, -np.pi / 2])

CAM_TRANSFORMS = {}
for name in CAMS:
    CAM_TRANSFORMS[name] = _T_tcp_mount @ _T_mount[name] @ _T_optical


def project_3d_to_2d(port_3d, tcp_pos, tcp_quat_xyzw, cam_name):
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
    from scipy.spatial.transform import Rotation as R

    T_world_tcp = np.eye(4)
    T_world_tcp[:3, :3] = R.from_quat(tcp_quat_xyzw).as_matrix()
    T_world_tcp[:3, 3] = tcp_pos

    A_rows = []
    for cam_name, (u, v) in detections.items():
        T_world_cam = T_world_tcp @ CAM_TRANSFORMS[cam_name]
        T_cam_world = np.linalg.inv(T_world_cam)
        P = K_SCALED @ T_cam_world[:3]
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])

    A = np.array(A_rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


def weighted_triangulate(detections, sigmas, tcp_pos, tcp_quat_xyzw):
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
# Dataset — per-camera samples with task-type label
# ---------------------------------------------------------------------------

def _upscale_img(img_256x288: np.ndarray) -> np.ndarray:
    """Bilinear upscale from stored 256×288 to current IMG_H×IMG_W."""
    if IMG_H == STORED_H and IMG_W == STORED_W:
        return img_256x288
    t = torch.from_numpy(img_256x288).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
    return t.squeeze(0).permute(1, 2, 0).byte().numpy()


def build_dataset(src: Path, episodes: list[int], sample_every: int = 10,
                  cache_dir: Path | None = None):
    """Build training dataset with task-type labels per episode."""

    suffix = f"_{IMG_H}x{IMG_W}"
    if cache_dir is None:
        cache_dir = Path(os.path.expanduser(f"~/.cache/aic_train_2d_v4{suffix}"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    positions_seen = {}
    valid_entries = []
    total_frames = 0

    for ep_num in episodes:
        ep_dir = src / f"episode_{ep_num:04d}"
        if not ep_dir.exists():
            continue

        data = np.load(ep_dir / "data.npz", allow_pickle=False)
        port = data['port_position_gt'][0]
        tcp_positions = data['tcp_position']
        tcp_orientations = data['tcp_orientation']
        n = len(tcp_positions)

        meta_path = ep_dir / "metadata.json"
        port_type_id = 0
        if meta_path.exists():
            meta = json.load(open(meta_path))
            port_type_id = PORT_TYPES.get(meta.get('port_type', 'sfp'), 0)

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

            # Per-camera projection: include frame if ANY camera can see the port.
            # Cameras that can't see the port get a None label — the dataset
            # class will skip those (camera, frame) pairs during training.
            projections = {}
            for cam in CAMS:
                result = project_3d_to_2d(port, tcp_pos, tcp_quat, cam)
                if result is not None:
                    projections[cam] = (result[0] / IMG_W, result[1] / IMG_H)

            if len(projections) < 1:
                continue

            valid_entries.append((ep_num, i, pos_id, projections,
                                  port.astype(np.float32),
                                  tcp_pos.copy(), tcp_quat.copy(),
                                  port_type_id))
            ep_valid += 1

        pt_name = {0: 'SFP', 1: 'SC'}[port_type_id]
        if ep_valid > 0:
            print(f"  ep_{ep_num:02d}: {ep_valid}/{n} valid, "
                  f"port={port.round(4)}, pos_id={pos_id}, type={pt_name}")

        del data
        gc.collect()

    n_valid = len(valid_entries)
    print(f"\nTotal: {n_valid}/{total_frames} valid frames, "
          f"{len(positions_seen)} unique positions")

    uv_labels = {cam: np.full((n_valid, 2), -1.0, dtype=np.float32) for cam in CAMS}
    cam_valid = {cam: np.zeros(n_valid, dtype=bool) for cam in CAMS}
    port_3d = np.zeros((n_valid, 3), dtype=np.float32)
    position_ids = np.zeros(n_valid, dtype=np.int32)
    port_type_ids = np.zeros(n_valid, dtype=np.int32)
    tcp_positions_arr = np.zeros((n_valid, 3), dtype=np.float64)
    tcp_orientations_arr = np.zeros((n_valid, 4), dtype=np.float64)

    for idx, (_, _, pos_id, projs, port, t_pos, t_quat, pt_id) in enumerate(valid_entries):
        for cam in CAMS:
            if cam in projs:
                uv_labels[cam][idx] = projs[cam]
                cam_valid[cam][idx] = True
        port_3d[idx] = port
        position_ids[idx] = pos_id
        port_type_ids[idx] = pt_id
        tcp_positions_arr[idx] = t_pos
        tcp_orientations_arr[idx] = t_quat

    # Write images to mmap
    mmap_paths = {}
    for cam in CAMS:
        mp = cache_dir / f"imgs_{cam}.npy"
        mmap = np.memmap(mp, dtype=np.uint8, mode='w+',
                         shape=(n_valid, IMG_H, IMG_W, 3))
        mmap_paths[cam] = (mp, mmap)

    ep_groups = defaultdict(list)
    for global_idx, (ep_num, frame_idx, *_) in enumerate(valid_entries):
        ep_groups[ep_num].append((global_idx, frame_idx))

    for ep_num, entries in sorted(ep_groups.items()):
        ep_dir = src / f"episode_{ep_num:04d}"
        frame_indices = [fi for _, fi in entries]
        global_indices = [gi for gi, _ in entries]

        for cam in CAMS:
            all_imgs = np.load(ep_dir / f"{cam}_images.npz")['images']
            for gi, fi in zip(global_indices, frame_indices):
                mmap_paths[cam][1][gi] = _upscale_img(all_imgs[fi])
            del all_imgs
        gc.collect()

    imgs_mmap = {}
    for cam in CAMS:
        mp, mmap = mmap_paths[cam]
        mmap.flush()
        del mmap
        imgs_mmap[cam] = np.memmap(mp, dtype=np.uint8, mode='r',
                                   shape=(n_valid, IMG_H, IMG_W, 3))

    print(f"Cache written: {n_valid} frames × 3 cameras")

    sfp_count = (port_type_ids == 0).sum()
    sc_count = (port_type_ids == 1).sum()
    print(f"Task types: SFP={sfp_count}, SC={sc_count}")
    for cam in CAMS:
        print(f"  {cam} camera visible: {cam_valid[cam].sum()}/{n_valid} "
              f"({cam_valid[cam].mean()*100:.0f}%)")

    result = {}
    for cam in CAMS:
        result[f'{cam}_imgs'] = imgs_mmap[cam]
        result[f'{cam}_uv'] = uv_labels[cam]
        result[f'{cam}_valid'] = cam_valid[cam]
    result['port_3d'] = port_3d
    result['position_id'] = position_ids
    result['port_type_id'] = port_type_ids
    result['tcp_position'] = tcp_positions_arr
    result['tcp_orientation'] = tcp_orientations_arr

    return result, positions_seen


# ---------------------------------------------------------------------------
# Per-camera dataset with spatial + color augmentation
# ---------------------------------------------------------------------------

class PerCamDataset:
    """Each sample: one camera image + UV label + task type.

    Only includes (camera, frame) pairs where the port is actually visible
    in that camera. This is critical for distance-aware training: at close
    range only 1-2 cameras may see the port; the model must learn that the
    port is invisible in certain views and output high uncertainty (sigma).

    With augment=True: spatial (affine) + color augmentation.
    The UV label is transformed by the same affine as the image.
    """

    def __init__(self, dataset, indices, augment=False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

        # Build a flat list of (frame_real_idx, cam_name) pairs
        # where the port is visible in that camera.
        self.samples = []
        for frame_idx in indices:
            for cam in CAMS:
                if dataset[f'{cam}_valid'][frame_idx]:
                    self.samples.append((frame_idx, cam))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        real, cam = self.samples[idx]

        img = torch.from_numpy(
            self.dataset[f'{cam}_imgs'][real].copy()
        ).permute(2, 0, 1).float() / 255.0

        uv = self.dataset[f'{cam}_uv'][real].copy()
        port_type = self.dataset['port_type_id'][real]

        if self.augment:
            img, uv = self._spatial_augment(img, uv)
            if uv is None:
                return self._fallback(real, cam)
            img = self._color_augment(img)

        return {
            'img': img,
            'uv': torch.from_numpy(uv).float(),
            'port_type': torch.tensor(port_type, dtype=torch.long),
        }

    def _fallback(self, real, cam):
        """When augmentation pushes target out of bounds, return un-augmented."""
        img = torch.from_numpy(
            self.dataset[f'{cam}_imgs'][real].copy()
        ).permute(2, 0, 1).float() / 255.0
        uv = self.dataset[f'{cam}_uv'][real].copy()
        port_type = self.dataset['port_type_id'][real]
        img = self._color_augment(img)
        return {
            'img': img,
            'uv': torch.from_numpy(uv).float(),
            'port_type': torch.tensor(port_type, dtype=torch.long),
        }

    @staticmethod
    def _spatial_augment(img, uv_norm):
        """Affine spatial augmentation: translate, rotate, scale.

        Returns (img, uv_norm) or (img, None) if target is out-of-bounds.
        """
        C, H, W = img.shape

        tx = (torch.rand(1).item() - 0.5) * 0.28  # ±14% of image width
        ty = (torch.rand(1).item() - 0.5) * 0.31  # ±15% of image height
        angle = (torch.rand(1).item() - 0.5) * 16.0  # ±8 degrees
        scale = 0.85 + torch.rand(1).item() * 0.30    # [0.85, 1.15]

        angle_rad = angle * np.pi / 180.0
        cos_a = np.cos(angle_rad) * scale
        sin_a = np.sin(angle_rad) * scale

        # Affine: rotate+scale around image center, then translate
        # theta maps output coords → input coords for grid_sample
        theta = torch.tensor([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty],
        ], dtype=torch.float32).unsqueeze(0)

        grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
        img_aug = F.grid_sample(img.unsqueeze(0), grid, mode='bilinear',
                                padding_mode='reflection',
                                align_corners=False).squeeze(0)

        # Transform UV label: convert normalized [0,1] → [-1,1] grid coords,
        # apply the FORWARD transform, convert back.
        u_grid = uv_norm[0] * 2.0 - 1.0
        v_grid = uv_norm[1] * 2.0 - 1.0

        # Forward transform (inverse of the grid_sample theta)
        det = cos_a * cos_a + sin_a * sin_a  # = scale^2
        inv_cos = cos_a / det
        inv_sin = sin_a / det
        u_centered = u_grid - tx
        v_centered = v_grid - ty
        u_new = inv_cos * u_centered + inv_sin * v_centered
        v_new = -inv_sin * u_centered + inv_cos * v_centered

        # Convert back to [0,1]
        u_out = (u_new + 1.0) / 2.0
        v_out = (v_new + 1.0) / 2.0

        margin = 0.02
        if u_out < margin or u_out > 1.0 - margin or v_out < margin or v_out > 1.0 - margin:
            return img_aug, None

        return img_aug, np.array([u_out, v_out], dtype=np.float32)

    @staticmethod
    def _color_augment(img):
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.6
        contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
        mean = img.mean()
        img = ((img - mean) * contrast + mean) * brightness
        img = img.clamp(0, 1)

        # Random Gaussian blur
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

        # Gaussian noise
        img = (img + torch.randn_like(img) * 0.025).clamp(0, 1)

        # Random erasing (simulate occlusion)
        if torch.rand(1).item() < 0.2:
            _, H, W = img.shape
            eh = int(H * torch.empty(1).uniform_(0.02, 0.12).item())
            ew = int(W * torch.empty(1).uniform_(0.02, 0.12).item())
            y0 = int(torch.randint(0, H - eh, (1,)).item())
            x0 = int(torch.randint(0, W - ew, (1,)).item())
            img[:, y0:y0 + eh, x0:x0 + ew] = torch.rand(3, eh, ew)

        return img


# ---------------------------------------------------------------------------
# Model V4: FPN + spatial softmax + FiLM task conditioning
# ---------------------------------------------------------------------------

FEAT_H = IMG_H // 4
FEAT_W = IMG_W // 4


class Port2DNetV4(nn.Module):
    """FPN decoder with FiLM-conditioned task-type modulation.

    The task type (SFP=0, SC=1) modulates intermediate features so
    the same backbone can specialize per port type.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1   # 64ch,  H/4
        self.layer2 = base.layer2   # 128ch, H/8
        self.layer3 = base.layer3   # 256ch, H/16
        self.layer4 = base.layer4   # 512ch, H/32

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

        # FiLM conditioning: task_type → (gamma, beta) for feature modulation
        self.task_embed = nn.Embedding(NUM_PORT_TYPES, 32)
        self.film_fc = nn.Sequential(
            nn.Linear(32, ch * 2),
            nn.ReLU(inplace=True),
            nn.Linear(ch * 2, ch * 2),
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

    def forward(self, x, port_type):
        """x: (B,3,H,W), port_type: (B,) long → uv (B,2), sigma (B,), heatmap."""
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

        feat = self.smooth(p1)   # (B, 128, H/4, W/4)

        # FiLM: modulate features based on task type
        emb = self.task_embed(port_type)       # (B, 32)
        film = self.film_fc(emb)               # (B, 256)
        gamma = film[:, :128].view(-1, 128, 1, 1) + 1.0
        beta = film[:, 128:].view(-1, 128, 1, 1)
        feat = feat * gamma + beta

        heatmap = self.heatmap_head(feat)      # (B, 1, Hf, Wf)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.expanduser("~/rl/diverse_training_data"))
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(87)))
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--img-scale", type=float, default=0.25,
                        help="Image scale factor (0.25=288x256, 0.5=576x512)")
    args = parser.parse_args()

    _set_resolution(args.img_scale)
    if args.batch_size is None:
        args.batch_size = 16 if args.img_scale >= 0.5 else 64

    from torch.utils.data import DataLoader

    src = Path(args.src).expanduser()
    res_tag = f"_{IMG_H}x{IMG_W}" if args.img_scale != 0.25 else ""
    out = Path(args.out or os.path.expanduser(f"~/rl/perception_checkpoints/port_2d_v4{res_tag}"))
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Port2DNet V4 — FPN + Spatial Aug + Task Conditioning")
    print(f"  Resolution: {IMG_W}×{IMG_H} (scale={args.img_scale})")
    print("=" * 60)
    print("\nBuilding dataset with 2D pixel labels + task types...")
    dataset, pos_map = build_dataset(src, args.episodes, args.sample_every)

    n = len(dataset['port_3d'])
    print(f"\nDataset: {n} samples, {len(pos_map)} positions")

    # Position-stratified val split
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

    train_ds = PerCamDataset(dataset, train_idx, augment=True)
    val_ds = PerCamDataset(dataset, val_idx, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    print(f"Train samples (per-cam): {len(train_ds)}, "
          f"Val samples (per-cam): {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Port2DNetV4().to(device)
    if args.resume:
        ckpt = Path(args.resume)
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  Missing keys (expected for new layers): {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        print(f"Resumed from {ckpt}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_p:,} params (V4: FPN + spatial softmax + FiLM)")
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
    PATIENCE = 40

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        t_loss, n_batch = 0.0, 0
        for batch in train_dl:
            imgs = batch['img'].to(device)
            gt_uv = batch['uv'].to(device)
            pt = batch['port_type'].to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_uv, _, _ = model(imgs, pt)
                loss = F.smooth_l1_loss(pred_uv, gt_uv)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            t_loss += loss.item()
            n_batch += 1

        t_loss /= max(1, n_batch)

        # --- Val ---
        model.eval()
        all_pred, all_gt_list = [], []
        with torch.no_grad():
            for batch in val_dl:
                imgs = batch['img'].to(device)
                gt_uv = batch['uv']
                pt = batch['port_type'].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_uv, _, _ = model(imgs, pt)

                all_pred.append(pred_uv.float().cpu().numpy())
                all_gt_list.append(gt_uv.numpy())

        pred = np.concatenate(all_pred)
        gt = np.concatenate(all_gt_list)
        err_u = np.abs(pred[:, 0] - gt[:, 0]) * IMG_W
        err_v = np.abs(pred[:, 1] - gt[:, 1]) * IMG_H
        err_px = np.sqrt(err_u ** 2 + err_v ** 2)
        mean_px = err_px.mean()
        med_px = np.median(err_px)

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
                f"loss={t_loss:.6f}  "
                f"px={mean_px:.1f}+-{err_px.std():.1f}  "
                f"u={err_u.mean():.1f}  v={err_v.mean():.1f}  "
                f"med={med_px:.1f}  "
                f"lr={lr:.2e}  "
                f"{'*BEST*' if improved else ''}"
            )

        if patience >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # -------------------------------------------------------------------
    # Final evaluation: per-camera + triangulation
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Final Evaluation")
    print("=" * 60)
    model.load_state_dict(
        torch.load(out / "port_2d.pt", map_location=device, weights_only=True)
    )
    model.eval()

    val_idx_arr = np.where(val_mask)[0]
    tri_errors_mm = []
    tri_weighted_mm = []
    px_errors_by_cam = {cam: [] for cam in CAMS}
    per_position_3d = defaultdict(list)

    with torch.no_grad():
        for idx in val_idx_arr:
            tcp_pos = dataset['tcp_position'][idx]
            tcp_quat = dataset['tcp_orientation'][idx]
            port_gt = dataset['port_3d'][idx]
            pt_id = dataset['port_type_id'][idx]
            pos_id = dataset['position_id'][idx]
            pt_tensor = torch.tensor([pt_id], dtype=torch.long, device=device)

            detections = {}
            sigmas = {}
            for cam in CAMS:
                img = torch.from_numpy(
                    dataset[f'{cam}_imgs'][idx].copy()
                ).permute(2, 0, 1).float().unsqueeze(0) / 255.0

                with torch.amp.autocast("cuda", enabled=use_amp):
                    uv_pred, sigma, _ = model(img.to(device), pt_tensor)
                uv_np = uv_pred.float().cpu().numpy()[0]
                sig_np = sigma.float().cpu().item()

                u_px = uv_np[0] * IMG_W
                v_px = uv_np[1] * IMG_H
                detections[cam] = (u_px, v_px)
                sigmas[cam] = sig_np

                gt_uv = dataset[f'{cam}_uv'][idx]
                px_err = np.sqrt(
                    (u_px - gt_uv[0] * IMG_W) ** 2 +
                    (v_px - gt_uv[1] * IMG_H) ** 2)
                px_errors_by_cam[cam].append(px_err)

            port_pred = triangulate_3_views(detections, tcp_pos, tcp_quat)
            err_mm = np.linalg.norm(port_pred - port_gt) * 1000
            tri_errors_mm.append(err_mm)

            port_w = weighted_triangulate(detections, sigmas, tcp_pos, tcp_quat)
            err_w_mm = np.linalg.norm(port_w - port_gt) * 1000
            tri_weighted_mm.append(err_w_mm)

            per_position_3d[pos_id].append(err_w_mm)

    tri_errors_mm = np.array(tri_errors_mm)
    tri_weighted_mm = np.array(tri_weighted_mm)

    print(f"\n  Val frames: {len(val_idx_arr)}")
    print(f"\n  2D pixel errors (per camera):")
    all_px = []
    for cam in CAMS:
        errs = np.array(px_errors_by_cam[cam])
        all_px.extend(errs)
        print(f"    {cam:6s}: {errs.mean():.2f} +- {errs.std():.2f} px  "
              f"(median {np.median(errs):.2f})")
    all_px = np.array(all_px)
    print(f"    TOTAL:  {all_px.mean():.2f} +- {all_px.std():.2f} px  "
          f"(median {np.median(all_px):.2f})")

    print(f"\n  3D triangulation errors:")
    print(f"    Equal-weight:  {tri_errors_mm.mean():.2f} +- "
          f"{tri_errors_mm.std():.2f} mm  "
          f"(median {np.median(tri_errors_mm):.2f})")
    print(f"    σ-weighted:    {tri_weighted_mm.mean():.2f} +- "
          f"{tri_weighted_mm.std():.2f} mm  "
          f"(median {np.median(tri_weighted_mm):.2f})")

    for thresh in (0.5, 1, 2, 3, 5):
        cnt = (tri_weighted_mm < thresh).sum()
        pct = cnt / len(tri_weighted_mm) * 100
        print(f"    <{thresh}mm: {cnt}/{len(tri_weighted_mm)} ({pct:.0f}%)")

    print(f"\n  Per-position 3D error (σ-weighted):")
    for pos_id in sorted(per_position_3d.keys()):
        errs = np.array(per_position_3d[pos_id])
        pos_key = [k for k, v in pos_map.items() if v == pos_id][0]
        print(f"    pos_{pos_id:2d} ({pos_key[0]:.3f},{pos_key[1]:.3f},{pos_key[2]:.3f}): "
              f"{errs.mean():.2f} +- {errs.std():.2f} mm  "
              f"(n={len(errs)})")

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
        if 'insertion_axis' in d:
            all_axes.append(d['insertion_axis'][0])
    if all_axes:
        mean_axis = np.mean(all_axes, axis=0)
        mean_axis /= np.linalg.norm(mean_axis)
        np.save(out / "mean_axis.npy", mean_axis)

    # Save port_type mapping for inference
    np.save(out / "port_type_map.npy", PORT_TYPES)

    print(f"\nSaved: {out}")
    print(f"  port_2d.pt — V4 detector (best={best_px:.2f}px)")
    print(f"  camera_geometry.npz — K + cam transforms")
    print(f"  port_type_map.npy — {{sfp:0, sc:1}}")
    return 0


if __name__ == "__main__":
    exit(main())
