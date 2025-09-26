# -*- coding: utf-8 -*-
"""
infer_3d_pcdet.py — Inference script based on OpenPCDet

Functions:
- Read Water-level-monitoring's config.yaml
- Support sequential inference for multiple PCDet models and output one JSON per model
- After inference, perform 3D multi-model fusion using BEV rotated IoU + majority voting, and output fused.det3d.json
- The fused result includes lock_gate_present and ship_present

Implementation details:
- Rotated IoU computation does not depend on pcdet.ops.iou3d_nms. It inlines pure Python/NumPy polygon clipping
  (Sutherland–Hodgman) and area calculation.
  Input 7D box [x, y, z, dx, dy, dz, heading] (as in OpenPCDet). Convert to a 4-point polygon on BEV and compute IoU.
  
Usage:
python scripts/infer_3d_pcdet.py --cfg configs/infer_3d_pcdet.yaml
"""

import argparse
import json
from pathlib import Path
import glob
import os
from math import ceil
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from pcdet.config import cfg as PC_CFG, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import yaml
from tqdm import tqdm
from contextlib import contextmanager


@contextmanager
def temp_chdir(d: str):
    """Temporarily chdir into d and restore the original working directory on exit."""
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


class DemoDataset(DatasetTemplate):
    """Simple dataset: supports .bin and .npy point cloud files."""
    def __init__(self, dataset_cfg, class_names, root_path, ext=".bin", logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names,
                         training=False, root_path=root_path, logger=logger)
        self.root_path = Path(root_path)
        self.ext = ext
        if self.root_path.is_dir():
            self.sample_file_list = sorted(glob.glob(str(self.root_path / f"*{self.ext}")))
        else:
            self.sample_file_list = [str(self.root_path)]

        if not self.sample_file_list:
            raise RuntimeError(f"No point files found at {self.root_path} with ext {self.ext}")

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        """Load one frame of point cloud data."""
        fp = self.sample_file_list[index]
        ext = Path(fp).suffix.lower()

        if ext == ".bin":
            points = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)
        elif ext == ".npy":
            points = np.load(fp)
            # If only xyz is present, append one intensity channel filled with 0
            if points.ndim == 2 and points.shape[1] == 3:
                inten = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points.astype(np.float32), inten])
        else:
            raise NotImplementedError(f"Unsupported ext {ext}")

        input_dict = {
            "points": points,
            "frame_id": Path(fp).stem,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def build_label_name_map(class_names):
    """PCDet prediction labels start from 1; map id → class name."""
    return {i + 1: str(n) for i, n in enumerate(class_names)}


# -------------------- Geometry & IoU utilities (pure Python/NumPy, rotated IoU) --------------------
EPS = 1e-9

def _box7d_to_bev_corners(box7d: np.ndarray) -> np.ndarray:
    """
    Convert a 7D box [x, y, z, dx, dy, dz, heading] to a 4x2 polygon (clockwise) on BEV.
    Convention: dx is the length along the local x-axis (usually "length"), dy along the local y-axis (width);
    heading is rotation around the z-axis (right-handed).
    """
    x, y, _, dx, dy, _, yaw = box7d[:7]
    # If dimensions are invalid, return a degenerate polygon (zero area)
    dx = float(dx); dy = float(dy)
    if dx <= 0 or dy <= 0:
        c = np.array([[x, y]] * 4, dtype=np.float32)
        return c

    # Local coordinates (origin at box center), clockwise
    # Order: (+dx/2,+dy/2) -> (+dx/2,-dy/2) -> (-dx/2,-dy/2) -> (-dx/2,+dy/2)
    hx = dx / 2.0
    hy = dy / 2.0
    local = np.array([[ hx,  hy],
                      [ hx, -hy],
                      [-hx, -hy],
                      [-hx,  hy]], dtype=np.float32)

    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    world = (local @ R.T) + np.array([x, y], dtype=np.float32)
    return world  # (4,2)

def _poly_area(poly: np.ndarray) -> float:
    """
    Polygon area (vertices ordered either CW or CCW; implicit closure). Returns non-negative area.
    poly: (N,2)
    """
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Ensure polygon is CCW (to keep the 'left side' as the interior consistently)."""
    if poly.shape[0] < 3:
        return poly
    # Signed area > 0 means CCW
    x = poly[:, 0]; y = poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed < 0:
        return poly[::-1].copy()
    return poly

def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """
    Check whether point p lies on the left side of directed edge a->b (interior for CCW polygons).
    """
    return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -EPS

def _compute_intersection(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Intersection point of segments ab and pq, assuming an intersection exists (used during clipping).
    Solved using parametric line equations.
    """
    # a + t*(b-a) = p + u*(q-p)
    r = b - a
    s = q - p
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < EPS:
        # Parallel or overlapping; return a (best effort to keep algorithm going; typically not reached in clipping)
        return a.copy()
    t = ((p[0]-a[0])*s[1] - (p[1]-a[1])*s[0]) / rxs
    return a + t * r

def _sutherland_hodgman(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman polygon clipping: clip 'subject' inside 'clipper' (assumed convex).
    Vertex orders: both in CCW.
    Returns the clipped polygon vertices (always as np.ndarray; possibly empty with shape (0,2)).
    """
    subject = np.asarray(subject, dtype=np.float32).reshape(-1, 2)
    clipper = np.asarray(clipper, dtype=np.float32).reshape(-1, 2)

    if subject.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    output = subject.copy()
    cp = clipper
    for i in range(cp.shape[0]):
        a = cp[i]
        b = cp[(i + 1) % cp.shape[0]]
        input_list = output
        # 👇 If already empty, return a standard empty array to avoid returning a list
        if input_list.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        new_out = []
        S = input_list[-1]
        for E in input_list:
            if _inside(E, a, b):
                if not _inside(S, a, b):
                    inter = _compute_intersection(S, E, a, b)
                    new_out.append(inter)
                new_out.append(E)
            elif _inside(S, a, b):
                inter = _compute_intersection(S, E, a, b)
                new_out.append(inter)
            S = E
        output = np.asarray(new_out, dtype=np.float32)

    # Always return np.ndarray
    if output.ndim == 1:
        output = output.reshape(-1, 2)
    return output


def _iou_poly(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    IoU of two polygons (CCW). If either degenerates, return 0.
    """
    poly_a = np.asarray(poly_a, dtype=np.float32).reshape(-1, 2)
    poly_b = np.asarray(poly_b, dtype=np.float32).reshape(-1, 2)

    if poly_a.shape[0] < 3 or poly_b.shape[0] < 3:
        return 0.0
    A = _poly_area(poly_a)
    B = _poly_area(poly_b)
    if A <= EPS or B <= EPS:
        return 0.0
    inter_poly = _sutherland_hodgman(poly_a, poly_b)
    I = _poly_area(inter_poly) if inter_poly.shape[0] >= 3 else 0.0
    U = A + B - I
    if U <= EPS:
        return 0.0
    return float(I / U)

def bev_iou_seed_vs_rest(seed_box7d: np.ndarray, others_box7d: np.ndarray) -> np.ndarray:
    """
    Compute BEV **rotated** IoU between one 7D box and multiple 7D boxes.
    boxes: [x, y, z, dx, dy, dz, heading]
    Returns: (N,) array of IoUs.
    """
    if others_box7d.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # Preprocess to CCW polygons
    seed_poly = _ensure_ccw(_box7d_to_bev_corners(seed_box7d))
    ious = np.zeros((others_box7d.shape[0],), dtype=np.float32)

    for i in range(others_box7d.shape[0]):
        other_poly = _ensure_ccw(_box7d_to_bev_corners(others_box7d[i]))
        ious[i] = _iou_poly(seed_poly, other_poly)

    return ious


def is_ship_label(label: str, ship_set: set) -> bool:
    return label in ship_set


# -------------------- FUSION (majority voting + rotated BEV-IoU clustering) --------------------
def fuse_3d_jsons(
    json_paths: List[str],
    labels_cfg: Dict[str, Any],
    iou_thr: float = 0.7,
    vote_ratio: float = 0.5,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fuse multiple models' 3D detections using rotated BEV IoU + majority voting.
    Rules:
      - Apply voting only to the two groups: 'lock_gate' and 'ships';
      - Rotated BEV IoU >= iou_thr is considered the same object cluster;
      - If the number of supporting models in a cluster >= ceil(num_models * vote_ratio),
        keep the cluster and write the representative box with the highest score;
      - ship_present / lock_gate_present are derived from the fused results.
    """
    lock_gate_label = labels_cfg['lock_gate']
    ship_labels = set(labels_cfg['ships'])
    num_models = len(json_paths)
    vote_need = ceil(num_models * vote_ratio)

    # Read all model results
    all_res: List[Dict[str, Any]] = []
    for p in json_paths:
        with open(p, 'r', encoding='utf-8') as f:
            all_res.append(json.load(f))

    # Union of frame IDs
    all_fids = set()
    for res in all_res:
        all_fids.update(res.keys())

    fused: Dict[str, Any] = {}

    for fid in tqdm(sorted(all_fids), desc="[3D:FUSE]"):
        per_model = []
        points_path = None
        for _, res in enumerate(all_res):
            dets = []
            if fid in res:
                entry = res[fid]
                points_path = points_path or entry.get('points_path')
                dets = entry.get('detections', [])
            per_model.append(dets)

        fused_dets: List[Dict[str, Any]] = []

        def cluster_and_vote(target_filter_fn):
            # Gather props: (box7d, score, label, label_id, model_idx)
            props_box = []
            props_meta = []  # (score, label, label_id, model)
            for m_idx, dets in enumerate(per_model):
                for d in dets:
                    if target_filter_fn(d['label']):
                        props_box.append(np.array(d['box7d'], dtype=np.float32))
                        props_meta.append((float(d['score']), d['label'], int(d.get('label_id', -1)), m_idx))

            if not props_box:
                return

            props_box = np.stack(props_box, axis=0)  # (M, 7)
            order = np.argsort([-pm[0] for pm in props_meta])  # sort by score desc
            used = np.zeros((len(order),), dtype=bool)

            for oi, _ in enumerate(order):
                if used[oi]:
                    continue
                seed_idx = order[oi]
                seed_box = props_box[seed_idx]
                cluster_idx = [oi]  # indices in 'order' space

                # Find others with IoU >= threshold against the seed
                remain_mask = ~used
                remain_mask[:oi+1] = False
                remain_inds = np.where(remain_mask)[0]
                if remain_inds.size > 0:
                    others_boxes = props_box[order[remain_inds]]
                    ious = bev_iou_seed_vs_rest(seed_box, others_boxes)  # (R,)
                    hit_inds = remain_inds[ious >= iou_thr]
                    cluster_idx.extend(list(hit_inds))

                # Count distinct supporting models
                support_models = set()
                for ci in cluster_idx:
                    _, _, _, m = props_meta[order[ci]]
                    support_models.add(m)

                if len(support_models) >= vote_need:
                    # Use the highest-score box in the cluster as representative
                    best_ci = max(cluster_idx, key=lambda ci: props_meta[order[ci]][0])
                    best_score, best_label, best_label_id, _ = props_meta[order[best_ci]]
                    best_box = props_box[order[best_ci]].tolist()
                    fused_dets.append({
                        "box7d": [float(t) for t in best_box],
                        "score": float(best_score),
                        "label_id": int(best_label_id),
                        "label": best_label
                    })

                # Mark the entire cluster as used
                for ci in cluster_idx:
                    used[ci] = True

        # Lock gate
        cluster_and_vote(lambda lab: lab == lock_gate_label)
        len_before_ships = len(fused_dets)
        # Ships (keep specific sub-class labels)
        cluster_and_vote(lambda lab: is_ship_label(lab, ship_labels))
        len_after_ships = len(fused_dets)

        lock_gate_present = any(det['label'] == lock_gate_label for det in fused_dets)
        ship_present = (len_after_ships > len_before_ships)

        fused[fid] = {
            "points_path": points_path,
            "detections": fused_dets,
            "lock_gate_present": lock_gate_present,
            "ship_present": ship_present
        }

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        with open(save_p, "w", encoding="utf-8") as f:
            json.dump(fused, f, ensure_ascii=False, indent=2)
        print(f"[3D:FUSE] Saved to {save_p}")
    return fused


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="path to Water-level-monitoring infer_3d_pcdet.yaml")
    parser.add_argument("--no_fuse", action="store_true", help="only run per-model inference JSON, skip fusion")
    parser.add_argument("--iou_thr", type=float, default=0.7, help="BEV IoU threshold for considering the same object")
    parser.add_argument("--vote_ratio", type=float, default=0.5, help="min ratio of models that must agree (e.g., 0.5 means at least half)")
    parser.add_argument("--ext", type=str, default=".bin", help="points file extension to read (.bin or .npy)")
    parser.add_argument("--pcdet_tools", type=str, default="/home/luxiaodong/PCDet_trout/tools",
                        help="Path to OpenPCDet 'tools' dir so relative _BASE_CONFIG_ resolves correctly")
    args = parser.parse_args()

    # Read Water-level-monitoring YAML config
    with open(args.cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Make input/output paths absolute (avoid side effects from temporary chdir)
    points_dir = Path(cfg["data"]["points_dir"]).expanduser().resolve()
    orig_cwd = Path(os.getcwd()).resolve()
    out_root_base = cfg.get("output", {}).get("root", "outputs")
    out_root_cfg = Path(out_root_base).expanduser()
    out_root = (out_root_cfg if out_root_cfg.is_absolute() else (orig_cwd / out_root_cfg)) / "det3d"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger()
    logger.info("=========== PCDet Inference from config.yaml ===========")

    pcdet_tools = args.pcdet_tools
    if not os.path.isdir(pcdet_tools):
        raise RuntimeError(f"PCDet tools dir not found: {pcdet_tools}")

    model_json_paths: List[str] = []

    # Parse config/build models in the PCDet tools directory (important: _BASE_CONFIG_ is relative to this dir)
    with temp_chdir(pcdet_tools):
        for model_cfg in cfg.get("pcdet_models", []):
            name = model_cfg["name"]
            cfg_file = model_cfg["config"]
            ckpt = model_cfg["checkpoint"]
            score_thr = float(model_cfg.get("score_thr", 0.0))

            logger.info(f"[{name}] Loading config={cfg_file}, ckpt={ckpt}")

            # Load PCDet model config (resolves _BASE_CONFIG_)
            cfg_from_yaml_file(cfg_file, PC_CFG)

            # Build dataset
            demo_dataset = DemoDataset(
                dataset_cfg=PC_CFG.DATA_CONFIG,
                class_names=PC_CFG.CLASS_NAMES,
                root_path=points_dir,
                ext=args.ext,
                logger=logger
            )

            # Build and load model
            model = build_network(model_cfg=PC_CFG.MODEL,
                                  num_class=len(PC_CFG.CLASS_NAMES),
                                  dataset=demo_dataset)
            model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
            model.cuda()
            model.eval()

            id2name = build_label_name_map(PC_CFG.CLASS_NAMES)
            results: Dict[str, Any] = {}

            # Inference per frame
            with torch.no_grad():
                for idx, data_dict in enumerate(tqdm(demo_dataset, desc=f"[3D:{name}]")):
                    frame_id = str(data_dict.get("frame_id", idx))
                    batch = demo_dataset.collate_batch([data_dict])
                    load_data_to_gpu(batch)

                    pred_dicts, _ = model.forward(batch)
                    pred = pred_dicts[0]

                    boxes = pred.get("pred_boxes", torch.empty((0, 7))).cpu().numpy()
                    scores = pred.get("pred_scores", torch.empty((0,))).cpu().numpy()
                    labels = pred.get("pred_labels", torch.empty((0,), dtype=torch.long)).cpu().numpy().astype(int)

                    detections = []
                    for b, s, l in zip(boxes, scores, labels):
                        if s < score_thr:
                            continue
                        detections.append({
                            "box7d": [float(t) for t in b.tolist()],
                            "score": float(s),
                            "label_id": int(l),
                            "label": id2name.get(int(l), str(l))
                        })

                    results[frame_id] = {
                        "points_path": str(demo_dataset.sample_file_list[idx]),
                        "detections": detections
                    }

            # Save this model's results (absolute path to avoid impact from chdir)
            out_path = out_root / f"{name}.det3d.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"[{name}] Saved results to {out_path}")
            model_json_paths.append(str(out_path))

    # Fusion output
    if (not args.no_fuse) and model_json_paths:
        fused_path = out_root / "fused.det3d.json"
        fused_path.parent.mkdir(parents=True, exist_ok=True)
        fuse_3d_jsons(
            json_paths=model_json_paths,
            labels_cfg=cfg["labels"],
            iou_thr=args.iou_thr,
            vote_ratio=args.vote_ratio,
            save_path=str(fused_path)
        )

    logger.info("All done.")


if __name__ == "__main__":
    main()

