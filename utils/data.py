# utils/data.py
# -*- coding: utf-8 -*-
import os, re, math, csv, json, hashlib
import datetime as dt
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURE_CACHE_FORMAT_VERSION = 1


def manifest_feature_spec(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    wall_band_x_half: float,
    y_bin: float,
    q_low: float,
    qc_min_pts: int,
    inflate_xy: float,
    inflate_z: float,
    remove_labels: Optional[List[str]] = None,
    use_point_removal: bool = True,
) -> Dict[str, Any]:
    """Return the exact preprocessing signature used by a feature cache."""
    normalized_labels = sorted(str(value) for value in remove_labels) if remove_labels else None
    return {
        "format_version": FEATURE_CACHE_FORMAT_VERSION,
        "geometry": {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "waterlevel": {
            "wall_band_x_half": float(wall_band_x_half),
            "y_bin": float(y_bin),
            "quantile_low": float(q_low),
            "qc_min_pts": int(qc_min_pts),
            "inflate_xy": float(inflate_xy),
            "inflate_z": float(inflate_z),
            "remove_labels": normalized_labels,
            "use_point_removal": bool(use_point_removal),
        },
    }


def feature_spec_sha256(spec: Dict[str, Any]) -> str:
    payload = json.dumps(
        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_record_source_key(record: Dict[str, Any]) -> str:
    """Identify the exact source/annotation mapping for one cached sample."""
    values = [
        str(record["points_path"]),
        str(record["annotation_type"]),
        str(record.get("annotation_path", "")),
        str(record.get("annotation_key", "")),
    ]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

# ---------- 点云与 add_info ----------
def load_points_any(path_no_ext: str) -> np.ndarray:
    """Try .bin then .npy. Return xyz float32 [N,3] (允许 .bin 带强度列)."""
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


def load_points_file(path: str) -> np.ndarray:
    """Load one explicit .bin/.npy path and return xyz float32 [N, 3]."""
    path = os.path.abspath(os.path.expanduser(str(path)))
    suffix = Path(path).suffix.lower()
    if suffix == ".bin":
        arr = np.fromfile(path, dtype=np.float32)
        if arr.size % 4 == 0:
            pts = arr.reshape(-1, 4)[:, :3]
        elif arr.size % 3 == 0:
            pts = arr.reshape(-1, 3)
        else:
            raise ValueError(f"Unexpected .bin size at {path}")
        return pts.astype(np.float32)
    if suffix == ".npy":
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] < 3:
            arr = arr.reshape(-1, 3)
        return arr[:, :3].astype(np.float32)
    raise ValueError(f"Unsupported point-cloud extension at {path}; expected .bin/.npy")

def parse_add_info(path: str) -> Dict[str, float]:
    """Parse: <frame_id> <timestamp> <waterlevel> [period] -> {fid: waterlevel}"""
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


# ---------- Operation-wise manifest ----------
_MANIFEST_REQUIRED_FIELDS = {
    "sample_id", "source", "frame_id", "timestamp", "date",
    "operation_id", "water_level", "points_path", "annotation_type",
    "annotation_path", "annotation_key", "split",
}


def _manifest_timestamp_seconds(timestamp: str) -> float:
    parts = re.split(r"[^\d]+", str(timestamp).strip())
    tokens = [value for value in parts if value]
    nums = [int(value) for value in tokens]
    if len(nums) < 6:
        raise ValueError(f"Invalid manifest timestamp: {timestamp!r}")
    year, month, day, hour, minute, second = nums[:6]
    microsecond = int((tokens[6] + "000000")[:6]) if len(tokens) >= 7 else 0
    return dt.datetime(
        year, month, day, hour, minute, second, microsecond
    ).timestamp()


def validate_operation_manifest(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fail loudly if samples or operations overlap across semantic splits."""
    if not records:
        raise RuntimeError("Operation manifest contains zero records")

    sample_ids = [str(record["sample_id"]) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Operation manifest contains duplicate sample_id values")

    operation_splits: Dict[int, set] = {}
    split_counts: Dict[str, int] = {}
    for record in records:
        split = str(record["split"])
        if split not in {"train", "val", "test"}:
            raise RuntimeError(
                f"Unsupported split {split!r} for sample {record['sample_id']}"
            )
        operation_id = int(record["operation_id"])
        operation_splits.setdefault(operation_id, set()).add(split)
        split_counts[split] = split_counts.get(split, 0) + 1

    leaking = {
        operation_id: sorted(splits)
        for operation_id, splits in operation_splits.items()
        if len(splits) != 1
    }
    if leaking:
        raise RuntimeError(f"Operation-level data leakage detected: {leaking}")

    split_operations = {
        split: sorted(
            operation_id
            for operation_id, values in operation_splits.items()
            if split in values
        )
        for split in ("train", "val", "test")
    }
    return {
        "num_frames": len(records),
        "num_operations": len(operation_splits),
        "split_counts": split_counts,
        "split_operations": split_operations,
        "operation_disjoint": True,
    }


def load_operation_manifest(
    path: str,
    split: Optional[str] = None,
    validate_paths: bool = True,
) -> List[Dict[str, Any]]:
    """Load and type-cast the source-aware operation manifest.

    Relative point/annotation paths are resolved against the manifest folder.
    The full manifest is audited before an optional split filter is applied.
    """
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_fields = sorted(_MANIFEST_REQUIRED_FIELDS - fields)
        if missing_fields:
            raise RuntimeError(
                f"Manifest {manifest_path} is missing fields: {missing_fields}"
            )
        raw_records = list(reader)

    records: List[Dict[str, Any]] = []
    for raw in raw_records:
        record: Dict[str, Any] = dict(raw)
        record["operation_id"] = int(raw["operation_id"])
        record["water_level"] = float(raw["water_level"])
        record["tsec"] = _manifest_timestamp_seconds(raw["timestamp"])
        for field in ("points_path", "annotation_path"):
            value = str(raw.get(field, "")).strip()
            if not value:
                record[field] = ""
                continue
            item_path = Path(os.path.expandvars(value)).expanduser()
            if not item_path.is_absolute():
                item_path = manifest_path.parent / item_path
            record[field] = str(item_path.resolve())
        if validate_paths:
            if not Path(record["points_path"]).is_file():
                raise FileNotFoundError(
                    f"Missing points for {record['sample_id']}: {record['points_path']}"
                )
            if record["annotation_type"] != "none" and not Path(
                record["annotation_path"]
            ).is_file():
                raise FileNotFoundError(
                    f"Missing annotation for {record['sample_id']}: "
                    f"{record['annotation_path']}"
                )
        records.append(record)

    validate_operation_manifest(records)
    if split is not None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown semantic split: {split!r}")
        records = [record for record in records if record["split"] == split]
        if not records:
            raise RuntimeError(
                f"Manifest {manifest_path} contains no records for split {split!r}"
            )
    records.sort(
        key=lambda record: (
            int(record["operation_id"]),
            float(record["tsec"]),
            str(record["sample_id"]),
        )
    )
    return records

# ---------- 3D 框与点云移除 ----------
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


@lru_cache(maxsize=8)
def load_det3d_boxes(path: str) -> Dict[str, List[OBB]]:
    """Load fused detector JSON as {annotation_key: [OBB, ...]}."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Detection JSON must contain an object: {path}")
    output: Dict[str, List[OBB]] = {}
    for key, item in data.items():
        detections = (item or {}).get("detections", []) or []
        boxes: List[OBB] = []
        for detection in detections:
            box = detection.get("box7d")
            if not box or len(box) < 7:
                continue
            boxes.append(
                OBB(
                    box[0], box[1], box[2], box[3], box[4], box[5], box[6],
                    detection.get("label", ""),
                )
            )
        output[str(key)] = boxes
    return output

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

# ---------- 墙带序列特征 ----------
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
    说明：
      - 当某个 bin 没有足够点（或左右墙都没有点）→ 记为 PAD；
      - 邻域填补：仅当“至少有一个邻居有效”时才做列均值；若邻居也全无效，则保持 PAD；
      - 不使用 np.nanmean，避免 'Mean of empty slice' 警告。
    """
    import math

    # 计算 y 向分箱
    span_y = max(1e-6, float(y_max - y_min))
    bins = int(math.ceil(span_y / float(y_bin_len)))
    bins = max(1, bins)
    edges = np.linspace(y_min, y_max, bins + 1, dtype=np.float32)

    # 初始化
    seq = np.full((bins, 2), np.nan, dtype=np.float32)
    mask = np.ones((bins,), dtype=bool)  # True 表示该 bin 目前无效（PAD）

    # 空点云直接返回
    if xyz.shape[0] == 0:
        return np.nan_to_num(seq, nan=0.0), mask

    # 限定闸室 ROI
    m_roi = (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    P = xyz[m_roi]
    if P.shape[0] == 0:
        return np.nan_to_num(seq, nan=0.0), mask

    # 墙带（左右）
    L = P[np.abs(P[:, 0] - x_min) <= wall_band_x_half]
    R = P[np.abs(P[:, 0] - x_max) <= wall_band_x_half]

    # 每个 bin 取分位数（足够点数才算有效）
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

        # 只要左右有一个是有效的，就不是 PAD
        if np.isfinite(seq[i, 0]) or np.isfinite(seq[i, 1]):
            mask[i] = False

    # --- 邻域填补（仅当邻居存在且有“至少一列”有有效值时才做） ---
    for i in range(bins):
        if mask[i]:
            neigh = []
            if i - 1 >= 0 and not mask[i - 1]:
                neigh.append(seq[i - 1])
            if i + 1 < bins and not mask[i + 1]:
                neigh.append(seq[i + 1])

            if len(neigh) > 0:
                arr = np.stack(neigh, axis=0)  # [K,2] 且 K>=1
                # 按列做“有效值均值”（避免 nanmean）
                m = np.empty((2,), dtype=np.float32)
                any_valid_col = False
                for j in range(2):
                    col = arr[:, j]
                    finite = np.isfinite(col)
                    if np.any(finite):
                        m[j] = float(col[finite].mean())
                        any_valid_col = True
                    else:
                        m[j] = np.nan  # 该列仍无效

                if not any_valid_col:
                    # 两列都无有效值 → 继续保持 PAD，不写入
                    continue

                # 至少一列有效：把 NaN 列按 0 回填，标记为有效
                m = np.where(np.isfinite(m), m, 0.0).astype(np.float32)
                seq[i] = m
                mask[i] = False  # 这个 bin 变为有效

    # 输出给模型：把 NaN 转 0；mask 仍指示哪些位置是 pad
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


class ManifestWLFrames(Dataset):
    """Water-level frames addressed by a source-aware operation manifest."""

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        wall_band_x_half: float,
        y_bin: float,
        q_low: float,
        qc_min_pts: int,
        inflate_xy: float,
        inflate_z: float,
        remove_labels: Optional[List[str]] = None,
        use_point_removal: bool = True,
        feature_cache_path: Optional[str] = None,
    ):
        self.records = list(records)
        if not self.records:
            raise RuntimeError("ManifestWLFrames received zero records")
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.wall_band_x_half = float(wall_band_x_half)
        self.y_bin = float(y_bin)
        self.q_low = float(q_low)
        self.qc_min_pts = int(qc_min_pts)
        self.inflate_xy = float(inflate_xy)
        self.inflate_z = float(inflate_z)
        self.remove_labels = set(remove_labels) if remove_labels else None
        self.use_point_removal = bool(use_point_removal)

        self.feature_spec = manifest_feature_spec(
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.wall_band_x_half,
            self.y_bin,
            self.q_low,
            self.qc_min_pts,
            self.inflate_xy,
            self.inflate_z,
            sorted(self.remove_labels) if self.remove_labels else None,
            self.use_point_removal,
        )
        self.feature_cache_path = (
            str(Path(feature_cache_path).expanduser().resolve())
            if feature_cache_path
            else None
        )
        self.cached_sequences: Optional[np.ndarray] = None
        self.cached_masks: Optional[np.ndarray] = None
        self.cached_sequence_tensors: Optional[torch.Tensor] = None
        self.cached_mask_tensors: Optional[torch.Tensor] = None
        self.cached_ground_truth: Optional[torch.Tensor] = None
        self.cached_metadata: Optional[List[Dict[str, Any]]] = None
        self.record_cache_positions: Optional[List[int]] = None
        self.cache_index: Dict[str, int] = {}
        if self.feature_cache_path:
            self._load_feature_cache(self.feature_cache_path)

        self.det3d_by_path: Dict[str, Dict[str, List[OBB]]] = {}
        if self.use_point_removal and self.cached_sequences is None:
            det_paths = sorted(
                {
                    str(record["annotation_path"])
                    for record in self.records
                    if record["annotation_type"] == "det3d_json"
                }
            )
            for det_path in det_paths:
                self.det3d_by_path[det_path] = load_det3d_boxes(det_path)

    def _load_feature_cache(self, path: str) -> None:
        cache_path = Path(path)
        if not cache_path.is_file():
            raise FileNotFoundError(f"Manifest feature cache does not exist: {cache_path}")
        with np.load(str(cache_path), allow_pickle=False) as cache:
            required_arrays = {
                "sample_ids", "source_keys", "sequences", "masks", "metadata_json"
            }
            missing = sorted(required_arrays - set(cache.files))
            if missing:
                raise RuntimeError(
                    f"Feature cache {cache_path} is missing arrays: {missing}"
                )
            sample_ids = np.asarray(cache["sample_ids"]).astype(str)
            source_keys = np.asarray(cache["source_keys"]).astype(str)
            sequences = np.asarray(cache["sequences"], dtype=np.float32)
            masks = np.asarray(cache["masks"], dtype=bool)
            metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))

        expected_hash = feature_spec_sha256(self.feature_spec)
        if metadata.get("feature_spec_sha256") != expected_hash:
            raise RuntimeError(
                "Feature-cache preprocessing mismatch: "
                f"cache={metadata.get('feature_spec_sha256')} expected={expected_hash} "
                f"path={cache_path}"
            )
        if metadata.get("feature_spec") != self.feature_spec:
            raise RuntimeError(
                f"Feature-cache specification differs from runtime config: {cache_path}"
            )
        if sample_ids.ndim != 1 or len(sample_ids) != len(set(sample_ids.tolist())):
            raise RuntimeError(f"Feature cache has duplicate/invalid sample IDs: {cache_path}")
        if source_keys.shape != sample_ids.shape:
            raise RuntimeError(f"Feature cache source-key shape mismatch: {cache_path}")
        if sequences.ndim != 3 or sequences.shape[0] != len(sample_ids) or sequences.shape[2] != 2:
            raise RuntimeError(f"Feature cache sequence shape is invalid: {sequences.shape}")
        if masks.shape != sequences.shape[:2]:
            raise RuntimeError(
                f"Feature cache mask shape {masks.shape} does not match {sequences.shape}"
            )

        expected_bins = max(
            1, int(math.ceil((self.y_max - self.y_min) / self.y_bin))
        )
        if sequences.shape[1] != expected_bins:
            raise RuntimeError(
                f"Feature cache has {sequences.shape[1]} bins; expected {expected_bins}"
            )

        cache_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids.tolist())}
        missing_ids = [
            str(record["sample_id"])
            for record in self.records
            if str(record["sample_id"]) not in cache_index
        ]
        if missing_ids:
            raise RuntimeError(
                f"Feature cache lacks {len(missing_ids)} required samples; "
                f"first={missing_ids[:3]} path={cache_path}"
            )
        for record in self.records:
            sample_id = str(record["sample_id"])
            cache_position = cache_index[sample_id]
            expected_source_key = manifest_record_source_key(record)
            if source_keys[cache_position] != expected_source_key:
                raise RuntimeError(
                    f"Feature cache source mapping changed for {sample_id}: {cache_path}"
                )

        self.cached_sequences = sequences
        self.cached_masks = masks
        self.cache_index = cache_index
        self.cached_sequence_tensors = torch.from_numpy(sequences)
        self.cached_mask_tensors = torch.from_numpy(masks)
        self.cached_ground_truth = torch.tensor(
            [float(record["water_level"]) for record in self.records],
            dtype=torch.float32,
        )
        self.cached_metadata = [
            self._metadata_for_record(record) for record in self.records
        ]
        self.record_cache_positions = [
            cache_index[str(record["sample_id"])] for record in self.records
        ]
        print(
            f"[feature-cache] loaded {cache_path} "
            f"samples={len(sample_ids)} shape={tuple(sequences.shape[1:])}"
        )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _metadata_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "sample_id": str(record["sample_id"]),
            "source": str(record["source"]),
            "frame_id": str(record["frame_id"]),
            "timestamp": str(record["timestamp"]),
            "date": str(record["date"]),
            "operation_id": int(record["operation_id"]),
            "tsec": float(record["tsec"]),
            "split": str(record["split"]),
        }

    def _boxes_for_record(self, record: Dict[str, Any]) -> List[OBB]:
        annotation_type = str(record["annotation_type"])
        if annotation_type == "none":
            return []
        if annotation_type == "label_txt":
            return load_labels_one(str(record["annotation_path"]))
        if annotation_type == "det3d_json":
            annotation_path = str(record["annotation_path"])
            annotation_key = str(record.get("annotation_key") or record["frame_id"])
            try:
                return self.det3d_by_path[annotation_path][annotation_key]
            except KeyError as exc:
                raise KeyError(
                    f"No detector entry for {record['sample_id']} "
                    f"(key={annotation_key!r}, json={annotation_path})"
                ) from exc
        raise ValueError(
            f"Unknown annotation_type={annotation_type!r} for {record['sample_id']}"
        )

    def _remove_obbs(self, xyz: np.ndarray, boxes: List[OBB]) -> np.ndarray:
        selected: List[OBB] = []
        for box in boxes:
            if self.remove_labels is not None and box.label not in self.remove_labels:
                continue
            if (
                self.x_min - box.dx * 0.5 <= box.cx <= self.x_max + box.dx * 0.5
                and self.y_min - box.dy * 0.5 <= box.cy <= self.y_max + box.dy * 0.5
            ):
                selected.append(box)
        if not selected:
            return xyz
        remove_mask = points_in_obb_mask(
            xyz, selected, self.inflate_xy, self.inflate_z
        )
        return xyz[~remove_mask]

    def __getitem__(self, idx):
        record = self.records[idx]
        if (
            self.cached_sequence_tensors is not None
            and self.cached_mask_tensors is not None
            and self.cached_ground_truth is not None
            and self.cached_metadata is not None
            and self.record_cache_positions is not None
        ):
            cache_position = self.record_cache_positions[idx]
            return (
                self.cached_sequence_tensors[cache_position],
                self.cached_mask_tensors[cache_position],
                self.cached_ground_truth[idx],
                self.cached_metadata[idx],
            )
        else:
            xyz = load_points_file(str(record["points_path"]))
            if self.use_point_removal:
                xyz = self._remove_obbs(xyz, self._boxes_for_record(record))
            seq, mask = extract_wall_sequence(
                xyz,
                self.x_min,
                self.x_max,
                self.y_min,
                self.y_max,
                self.wall_band_x_half,
                self.y_bin,
                self.q_low,
                self.qc_min_pts,
            )
        metadata = self._metadata_for_record(record)
        gt = torch.tensor(float(record["water_level"]), dtype=torch.float32)
        return torch.from_numpy(seq), torch.from_numpy(mask), gt, metadata

def collate_fn(batch):
    seq = torch.stack([b[0] for b in batch], dim=0)  # [B,T,2]
    mask= torch.stack([b[1] for b in batch], dim=0)  # [B,T]
    gt  = torch.stack([b[2] for b in batch], dim=0)  # [B]
    fids= [b[3] for b in batch]
    return seq.float(), mask.bool(), gt.float(), fids


def collate_manifest(batch):
    seq = torch.stack([item[0] for item in batch], dim=0)
    mask = torch.stack([item[1] for item in batch], dim=0)
    gt = torch.stack([item[2] for item in batch], dim=0)
    metadata = [item[3] for item in batch]
    return seq.float(), mask.bool(), gt.float(), metadata
