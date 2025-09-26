# utils/eval_utils.py
# -*- coding: utf-8 -*-
import os, re, json, math, warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import datetime as dt
import torch
from torch.utils.data import Dataset, DataLoader

from .io import ensure_dir, resolve_path, load_yaml  
from .data import extract_wall_sequence             
from .metrics import eval_metrics                   
from .trainer import build_model                    


# ---------- Strict add_info timestamp parsing ----------
def _parse_ts_str(ts: str) -> Optional[float]:
    """YYYY_MM_DD_HH_MM_SS[_micro] -> Unix seconds; return None if failed"""
    try:
        parts = re.split(r"[^\d]+", ts.strip())
        nums = [int(x) for x in parts if x != ""]
        if len(nums) >= 6:
            year, mon, day, hh, mm, ss = nums[:6]
            micro = nums[6] if len(nums) >= 7 else 0
            micro = int(micro) if micro < 1_000_000 else int(str(micro)[:6])
            dtobj = dt.datetime(year, mon, day, hh, mm, ss, micro)
            return dtobj.timestamp()
        return None
    except Exception:
        return None

def parse_add_info_strict(path: str) -> Dict[str, Tuple[float, float, str]]:
    """
    <frame_id> <timestamp> <waterlevel> [period]
    Returns {fid: (wl, tsec (strict non-None), ts_raw)};
    if any line fails to parse → raise
    """
    mp = {}
    bad = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", s)
            if len(parts) < 3:
                bad.append((i, s, "columns<3")); continue
            fid, ts_raw, wl = parts[0], parts[1], parts[2]
            try:
                wl_f = float(wl)
            except Exception:
                bad.append((i, s, "wl_not_float")); continue
            tsec = _parse_ts_str(ts_raw)
            if tsec is None:
                bad.append((i, s, "ts_parse_failed")); continue
            mp[fid] = (wl_f, float(tsec), ts_raw)
    if len(mp) == 0:
        raise RuntimeError(f"Parsed 0 valid rows from add_info: {path}")
    if bad:
        msg = "\n".join([f"  line#{ln}: reason={why} :: {raw}" for ln, raw, why in bad])
        raise RuntimeError(f"[STRICT TIMESTAMP] {len(bad)} rows failed to parse in {path}:\n{msg}")
    return mp
# Behavior aligned with your existing eval_wl_transformer/retnet/rwkv/mamba strict sorting implementation.


# ---------- Read 3D boxes from fused.det3d.json and perform point removal ----------
class OBB:
    __slots__ = ("cx","cy","cz","dx","dy","dz","yaw","label")
    def __init__(self, cx, cy, cz, dx, dy, dz, yaw, label):
        self.cx=float(cx); self.cy=float(cy); self.cz=float(cz)
        self.dx=float(dx); self.dy=float(dy); self.dz=float(dz)
        self.yaw=float(yaw); self.label=str(label)

def load_det3d_fused_json(path: str) -> Dict[str, List[OBB]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for fid, item in data.items():
        dets = item.get("detections", []) or []
        boxes = []
        for d in dets:
            b = d.get("box7d", None)
            if not b or len(b) < 7:
                continue
            label = d.get("label", "")
            boxes.append(OBB(b[0], b[1], b[2], b[3], b[4], b[5], b[6], label))
        out[fid] = boxes
    return out
# Equivalent implementation as in each eval_wl_*.py


def points_in_obb_mask(xyz: np.ndarray, boxes: List[OBB],
                       inflate_xy: float, inflate_z: float) -> np.ndarray:
    if len(boxes) == 0 or xyz.shape[0] == 0:
        return np.zeros((xyz.shape[0],), dtype=bool)
    P = xyz[:, :3]
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


# ---------- Eval dataset ----------
def _num_key(s: str):
    try: return (0, int(s))
    except Exception: return (1, s)

class EvalFrames(Dataset):
    def __init__(self, frames_sorted: List[Tuple[str, float, str]],
                 gt_map: Dict[str, Tuple[float, float, str]],
                 points_dir: str,
                 det_boxes: Dict[str, List[OBB]],
                 x_min: float, x_max: float, y_min: float, y_max: float,
                 wall_band_x_half: float, y_bin: float,
                 q_low: float, qc_min_pts: int,
                 inflate_xy: float, inflate_z: float,
                 rm_labels: Optional[List[str]] = None,
                 use_point_removal: bool = True):
        self.frames_sorted = frames_sorted
        self.gt_map = gt_map
        self.points_dir = points_dir
        self.det_boxes = det_boxes
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.wall_band_x_half = wall_band_x_half
        self.y_bin = y_bin
        self.q_low = q_low
        self.qc_min_pts = qc_min_pts
        self.inflate_xy = inflate_xy
        self.inflate_z = inflate_z
        self.rm_labels = set(rm_labels) if rm_labels else None  # None=remove all
        self.use_point_removal = use_point_removal

    def __len__(self): return len(self.frames_sorted)

    def __getitem__(self, idx):
        fid, tsec, ts_raw = self.frames_sorted[idx]
        # Read point cloud: same as your eval script, support .bin/.npy where first 2–3 cols are xyz  
        p_bin = os.path.join(self.points_dir, fid + ".bin")
        p_npy = os.path.join(self.points_dir, fid + ".npy")
        if os.path.isfile(p_bin):
            arr = np.fromfile(p_bin, dtype=np.float32)
            if arr.size % 4 == 0:
                pts = arr.reshape(-1, 4)[:, :3]
            else:
                pts = arr.reshape(-1, 3)
        elif os.path.isfile(p_npy):
            arr = np.load(p_npy)
            if arr.ndim != 2: arr = arr.reshape(-1, arr.shape[-1])
            pts = arr[:, :3]
        else:
            raise FileNotFoundError(f"No point file for {fid} (.bin/.npy)")

        m_roi = (pts[:,0]>=self.x_min)&(pts[:,0]<=self.x_max)&(pts[:,1]>=self.y_min)&(pts[:,1]<=self.y_max)
        P = pts[m_roi]

        if self.use_point_removal:
            boxes_all = self.det_boxes.get(fid, [])
            if self.rm_labels is None or len(self.rm_labels)==0:
                boxes = boxes_all
            else:
                boxes = [b for b in boxes_all if (b.label in self.rm_labels)]
            boxes = [b for b in boxes if (self.x_min - b.dx*0.5 <= b.cx <= self.x_max + b.dx*0.5) and
                                       (self.y_min - b.dy*0.5 <= b.cy <= self.y_max + b.dy*0.5)]
            if len(boxes) > 0 and P.shape[0] > 0:
                m = points_in_obb_mask(P[:, :3], boxes, self.inflate_xy, self.inflate_z)
                P = P[~m]

        seq, mask = extract_wall_sequence(
            P,
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.wall_band_x_half, self.y_bin,
            self.q_low, self.qc_min_pts
        )
        gt, _, _ = self.gt_map[fid]
        return torch.from_numpy(seq), torch.from_numpy(mask), torch.tensor(gt, dtype=torch.float32), fid, float(tsec), ts_raw

def collate_eval(batch):
    seq = torch.stack([b[0] for b in batch], dim=0)
    mask= torch.stack([b[1] for b in batch], dim=0)
    gt  = torch.stack([b[2] for b in batch], dim=0)
    fids= [b[3] for b in batch]
    tsec= np.array([b[4] for b in batch], dtype=np.float64)
    ts_raw = [b[5] for b in batch]
    return seq.float(), mask.bool(), gt.float(), fids, tsec, ts_raw


# ---------- Online Kalman filter ----------
def kf_online(times, obs, base_R, q_pos, q_vel, reset_gap,
              history_len=0, init_mode="use_obs", default_value=0.0,
              pos_var0=0.05, vel_var0=0.01, warmup_frames=0):
    n = len(obs); out = [np.nan]*n
    x = P = None; t_prev = None; age = 0
    for i, (ti, zi) in enumerate(zip(times, obs)):
        if i < warmup_frames:
            out[i] = float(zi) if (zi is not None and np.isfinite(zi)) else (
                     float(out[i-1]) if i>0 and np.isfinite(out[i-1]) else float(default_value))
            x = None; P = None; t_prev = ti; age = 0
            continue

        need_reset = (x is None) or (i>0 and (ti - t_prev) > reset_gap) or (history_len>0 and age>=history_len)
        if need_reset:
            if init_mode == "use_obs" and zi is not None and np.isfinite(zi):
                H0 = float(zi)
            elif init_mode == "use_last" and i>0 and np.isfinite(out[i-1]):
                H0 = float(out[i-1])
            else:
                H0 = float(default_value) if np.isfinite(default_value) else (float(zi) if np.isfinite(zi) else 0.0)
            x = np.array([[H0],[0.0]], dtype=np.float64)
            P = np.diag([pos_var0**2, vel_var0**2])
            t_prev = ti; age = 0
            out[i] = float(x[0,0]); continue

        dt_sec = max(1e-3, ti - t_prev); t_prev = ti
        F = np.array([[1.0, dt_sec],[0.0, 1.0]], dtype=np.float64)
        Q = np.array([[q_pos*dt_sec, 0.0],[0.0, q_vel*dt_sec]], dtype=np.float64)
        x = F @ x; P = F @ P @ F.T + Q

        if zi is not None and np.isfinite(zi):
            R = np.array([[max(1e-6, base_R**2)]], dtype=np.float64)
            H = np.array([[1.0, 0.0]], dtype=np.float64)
            y = np.array([[float(zi)]], dtype=np.float64) - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P

        out[i] = float(x[0,0]); age += 1
    return out
# Equivalent behavior as in each eval_wl_*.py Kalman filter implementation.

