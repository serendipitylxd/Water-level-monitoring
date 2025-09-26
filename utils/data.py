# utils/data.py
# -*- coding: utf-8 -*-
import os, re, math
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- Point cloud and add_info ----------
def load_points_any(path_no_ext: str) -> np.ndarray:
    """Try .bin then .npy. Return xyz float32 [N,3] (allow .bin with intensity column)."""
    p_bin = path_no_ext + ".bin"
    p_npy = path_no_ext + ".npy"
    if os.path.isfile(p_bin):
        arr = np.fromfile(p_bin, dtype=np.float32)
        if arr.size % 4 == 0:
            pts = arr.reshape(-1, 4)[:, :3]
        elif arr.size % 3 == 0:
            pts = arr.reshape(-1, 3)
        else:
            raise ValueError(f"Unexpected .bin size at {p_bin}")
        return pts.astype(np.float32)
    elif os.path.isfile(p_npy):
        arr = np.load(p_npy)
        if arr.ndim != 2 or arr.shape[1] < 3:
            arr = arr.reshape(-1, 3)
        return arr[:, :3].astype(np.float32)
    else:
        raise FileNotFoundError(f"No point file found for: {path_no_ext} (.bin/.npy)")

def parse_add_info(path: str) -> Dict[str, float]:
    """Parse format: <frame_id> <timestamp> <waterlevel> [period] -> {fid: waterlevel}"""
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 3:
                continue
            fid, ts, wl = parts[0], parts[1], parts[2]
            try:
                mp[fid] = float(wl)
            except Exception:
                continue
    if len(mp) == 0:
        raise RuntimeError(f"Parsed 0 rows from add_info: {path}")
    return mp

# ---------- 3D bounding boxes and point removal ----------
class OBB:
    __slots__ = ("cx","cy","cz","dx","dy","dz","yaw","label")
    def __init__(self, cx, cy, cz, dx, dy, dz, yaw, label):
        self.cx=float(cx); self.cy=float(cy); self.cz=float(cz)
        self.dx=float(dx); self.dy=float(dy); self.dz=float(dz)
        self.yaw=float(yaw); self.label=str(label)

def load_labels_one(path: str) -> List[OBB]:
    items = []
    if not os.path.isfile(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            cx, cy, cz, dx, dy, dz, yaw = map(float, parts[:7])
            label = parts[7]
            items.append(OBB(cx, cy, cz, dx, dy, dz, yaw, label))
    return items

def points_in_obb_mask(xyz: np.ndarray, boxes: List[OBB],
                       inflate_xy: float, inflate_z: float) -> np.ndarray:
    if len(boxes) == 0 or xyz.shape[0] == 0:
        return np.zeros((xyz.shape[0],), dtype=bool)
    P = xyz
    mask = np.zeros((P.shape[0],), dtype=bool)
    for b in boxes:
        PX = P[:,0] - b.cx
        PY = P[:,1] - b.cy
        c = math.cos(-b.yaw); s = math.sin(-b.yaw)
        X = c*PX - s*PY
        Y = s*PX + c*PY
        Z = P[:,2] - b.cz
        hx = b.dx*0.5 + float(inflate_xy)
        hy = b.dy*0.5 + float(inflate_xy)
        hz = b.dz*0.5 + float(inflate_z)
        inside = (np.abs(X) <= hx) & (np.abs(Y) <= hy) & (np.abs(Z) <= hz)
        mask |= inside
    return mask

# ---------- Wall-band sequence feature extraction ----------
def extract_wall_sequence(xyz: np.ndarray,
                          x_min: float, x_max: float,
                          y_min: float, y_max: float,
                          wall_band_x_half: float,
                          y_bin_len: float,
                          q_low: float,
                          qc_min_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      seq:  [B, 2]  per bin [z_left_q, z_right_q] (0 if invalid)
      mask: [B]     True if this bin is PAD (invalid)

    Notes:
      - If a bin has too few points (or no points on either wall) → mark as PAD;
      - Neighbor filling: only performed if at least one neighbor is valid;
        if all neighbors are invalid, keep as PAD;
      - Do not use np.nanmean to avoid 'Mean of empty slice' warnings.
    """
    import math

    # Compute binning along y-axis
    span_y = max(1e-6, float(y_max - y_min))
    bins = int(math.ceil(span_y / float(y_bin_len)))
    bins = max(1, bins)
    edges = np.linspace(y_min, y_max, bins + 1, dtype=np.float32)

    # Initialize
    seq = np.full((bins, 2), np.nan, dtype=np.float32)
    mask = np.ones((bins,), dtype=bool)  # True means currently invalid (PAD)

    # Empty point cloud → return directly
    if xyz.shape[0] == 0:
        return np.nan_to_num(seq, nan=0.0), mask

    # Restrict to ROI (lock chamber)
    m_roi = (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    P = xyz[m_roi]
    if P.shape[0] == 0:
        return np.nan_to_num(seq, nan=0.0), mask

    # Wall bands (left & right)
    L = P[np.abs(P[:, 0] - x_min) <= wall_band_x_half]
    R = P[np.abs(P[:, 0] - x_max) <= wall_band_x_half]

    # Per-bin quantile (valid only if enough points)
    for i in range(bins):
        yl, yr = float(edges[i]), float(edges[i + 1])

        if L.shape[0] > 0:
            zL = L[(L[:, 1] >= yl) & (L[:, 1] < yr), 2]
            if zL.size >= qc_min_pts:
                seq[i, 0] = np.quantile(zL, q_low)

        if R.shape[0] > 0:
            zR = R[(R[:, 1] >= yl) & (R[:, 1] < yr), 2]
            if zR.size >= qc_min_pts:
                seq[i, 1] = np.quantile(zR, q_low)

        # If at least one side is valid → not PAD
        if np.isfinite(seq[i, 0]) or np.isfinite(seq[i, 1]):
            mask[i] = False

    # --- Neighbor filling (only if neighbors exist and at least one column has valid value) ---
    for i in range(bins):
        if mask[i]:
            neigh = []
            if i - 1 >= 0 and not mask[i - 1]:
                neigh.append(seq[i - 1])
            if i + 1 < bins and not mask[i + 1]:
                neigh.append(seq[i + 1])

            if len(neigh) > 0:
                arr = np.stack(neigh, axis=0)  # [K,2] with K>=1
                # Column-wise mean of valid values (avoid nanmean)
                m = np.empty((2,), dtype=np.float32)
                any_valid_col = False
                for j in range(2):
                    col = arr[:, j]
                    finite = np.isfinite(col)
                    if np.any(finite):
                        m[j] = float(col[finite].mean())
                        any_valid_col = True
                    else:
                        m[j] = np.nan  # this column still invalid

                if not any_valid_col:
                    # Both columns invalid → keep PAD
                    continue

                # At least one column valid: fill NaN columns with 0 and mark as valid
                m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
                seq[i] = m
                mask[i] = False  # this bin becomes valid

    # Final output: replace NaN with 0 for model; mask still indicates PAD
    seq = np.nan_to_num(seq, nan=0.0).astype(np.float32)
    return seq, mask


# ---------- Dataset ----------
class WLFrames(Dataset):
    def __init__(self, frame_ids: List[str], gt_map: Dict[str, float],
                 points_dir: str, labels_dir: Optional[str],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 wall_band_x_half: float, y_bin: float,
                 q_low: float, qc_min_pts: int,
                 inflate_xy: float, inflate_z: float,
                 remove_labels: Optional[List[str]] = None,
                 use_point_removal: bool = True):
        self.fids = frame_ids
        self.gt_map = gt_map
        self.points_dir = points_dir
        self.labels_dir = labels_dir if use_point_removal else None
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.wall_band_x_half = wall_band_x_half
        self.y_bin = y_bin
        self.q_low = q_low
        self.qc_min_pts = qc_min_pts
        self.inflate_xy = inflate_xy
        self.inflate_z = inflate_z
        self.remove_labels = set(remove_labels) if remove_labels else None

    def __len__(self): return len(self.fids)

    def _remove_obb(self, xyz: np.ndarray, fid: str) -> np.ndarray:
        if self.labels_dir is None:
            return xyz
        path = os.path.join(self.labels_dir, f"{fid}.txt")
        obbs = load_labels_one(path)
        if len(obbs) == 0:
            return xyz
        boxes = []
        for b in obbs:
            if (self.remove_labels is not None) and (b.label not in self.remove_labels):
                continue
            if (self.x_min - b.dx*0.5 <= b.cx <= self.x_max + b.dx*0.5) and \
               (self.y_min - b.dy*0.5 <= b.cy <= self.y_max + b.dy*0.5):
                boxes.append(b)
        if len(boxes) == 0:
            return xyz
        m = points_in_obb_mask(xyz, boxes, self.inflate_xy, self.inflate_z)
        return xyz[~m]

    def __getitem__(self, idx):
        fid = self.fids[idx]
        xyz = load_points_any(os.path.join(self.points_dir, fid))
        if self.labels_dir is not None:
            xyz = self._remove_obb(xyz, fid)
        seq, mask = extract_wall_sequence(
            xyz,
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.wall_band_x_half, self.y_bin,
            self.q_low, self.qc_min_pts
        )
        gt = float(self.gt_map[fid])
        return torch.from_numpy(seq), torch.from_numpy(mask), torch.tensor(gt, dtype=torch.float32), fid

def collate_fn(batch):
    seq = torch.stack([b[0] for b in batch], dim=0)  # [B,T,2]
    mask= torch.stack([b[1] for b in batch], dim=0)  # [B,T]
    gt  = torch.stack([b[2] for b in batch], dim=0)  # [B]
    fids= [b[3] for b in batch]
    return seq.float(), mask.bool(), gt.float(), fids

