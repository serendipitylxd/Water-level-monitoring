# -*- coding: utf-8 -*-
"""
infer_3d_pcdet.py — Inference script based on OpenPCDet

功能：
- 读取 Water-level-monitoring 的 config.yaml
- 支持多个 PCDet 模型顺序推理，逐模型输出 JSON
- 在推理完成后按 BEV 旋转 IoU + 多数投票进行 3D 多模型融合，输出 fused.det3d.json
- 融合结果包含 lock_gate_present 与 ship_present

实现细节：
- 旋转 IoU 计算不依赖 pcdet.ops.iou3d_nms，内联纯 Python/NumPy 的多边形裁剪（Sutherland–Hodgman）与面积计算。
  输入 7D 盒 [x, y, z, dx, dy, dz, heading]（OpenPCDet 约定），在 BEV 上转为 4 点多边形，计算 IoU。
  
运行：
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
    """临时进入目录 d，退出时恢复原工作目录。"""
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


class DemoDataset(DatasetTemplate):
    """简单数据集：支持 .bin 与 .npy 点云文件。"""
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
        """加载一帧点云数据。"""
        fp = self.sample_file_list[index]
        ext = Path(fp).suffix.lower()

        if ext == ".bin":
            points = np.fromfile(fp, dtype=np.float32).reshape(-1, 4)
        elif ext == ".npy":
            points = np.load(fp)
            # 若只有 xyz，则补 1 维强度为 0
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
    """PCDet 的预测标签从 1 开始，这里映射 id → 类名。"""
    return {i + 1: str(n) for i, n in enumerate(class_names)}


# -------------------- 几何 & IoU 工具（纯 Python/NumPy，旋转 IoU） --------------------
EPS = 1e-9

def _box7d_to_bev_corners(box7d: np.ndarray) -> np.ndarray:
    """
    将 7D 盒 [x, y, z, dx, dy, dz, heading] 转为 BEV 上四边形 4x2 顶点（顺时针）。
    约定：dx 为沿局部 x 轴长度（通常“长”），dy 为沿局部 y 轴宽度；heading 绕 z 轴，右手系。
    """
    x, y, _, dx, dy, _, yaw = box7d[:7]
    # 若盒尺寸非法，返回退化多边形（零面积）
    dx = float(dx); dy = float(dy)
    if dx <= 0 or dy <= 0:
        c = np.array([[x, y]] * 4, dtype=np.float32)
        return c

    # 局部坐标系（以盒中心为原点），按顺时针
    # 顺序：(+dx/2,+dy/2) -> (+dx/2,-dy/2) -> (-dx/2,-dy/2) -> (-dx/2,+dy/2)
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
    多边形面积（要求顶点按顺/逆时针，自动闭合），返回非负面积。
    poly: (N,2)
    """
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]; y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """确保多边形为 CCW（便于一致的“左侧”为内侧判断）"""
    if poly.shape[0] < 3:
        return poly
    # 签名面积>0 表示 CCW
    x = poly[:, 0]; y = poly[:, 1]
    signed = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed < 0:
        return poly[::-1].copy()
    return poly

def _inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """
    判断点 p 是否在有向边 a->b 的左侧（CCW 多边形的内部）。
    """
    return ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -EPS

def _compute_intersection(a: np.ndarray, b: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    线段 ab 与 pq 的交点，假定存在交（用于裁剪过程），返回 2D 点。
    使用两线参数式求解。
    """
    # a + t*(b-a) = p + u*(q-p)
    r = b - a
    s = q - p
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < EPS:
        # 平行或重合，返回 a（尽量不中断算法；实际裁剪时一般不走到这里）
        return a.copy()
    t = ((p[0]-a[0])*s[1] - (p[1]-a[1])*s[0]) / rxs
    return a + t * r

def _sutherland_hodgman(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """
    Sutherland–Hodgman 多边形裁剪：裁剪 subject，使其落在 clipper（凸多边形）内。
    顶点顺序：都按 CCW。
    返回裁剪后的多边形顶点（始终为 np.ndarray，可能为空 (0,2)）。
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
        # 👇 如果已经是空，直接返回标准空数组，避免返回 list
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

    # 统一返回 np.ndarray
    if output.ndim == 1:
        output = output.reshape(-1, 2)
    return output


def _iou_poly(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """
    两个多边形（CCW）的 IoU。若任一退化，返回 0。
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
    计算单个 7D 盒与多个 7D 盒在 BEV 的**旋转** IoU。
    boxes: [x, y, z, dx, dy, dz, heading]
    返回：(N,) IoU 数组。
    """
    if others_box7d.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # 预处理为 CCW 多边形
    seed_poly = _ensure_ccw(_box7d_to_bev_corners(seed_box7d))
    ious = np.zeros((others_box7d.shape[0],), dtype=np.float32)

    for i in range(others_box7d.shape[0]):
        other_poly = _ensure_ccw(_box7d_to_bev_corners(others_box7d[i]))
        ious[i] = _iou_poly(seed_poly, other_poly)

    return ious


def is_ship_label(label: str, ship_set: set) -> bool:
    return label in ship_set


# -------------------- FUSION (多数投票 + 旋转 BEV-IoU 聚类) --------------------
def fuse_frame_detections(
    per_model: List[List[Dict[str, Any]]],
    labels_cfg: Dict[str, Any],
    iou_thr: float = 0.7,
    vote_ratio: float = 0.5,
) -> Dict[str, Any]:
    """Fuse one frame in memory using the same deployed voting rule.

    Keeping this logic independent of JSON I/O allows an actual end-to-end
    runtime benchmark to include fusion without estimating it from saved files.
    """
    lock_gate_label = labels_cfg['lock_gate']
    ship_labels = set(labels_cfg['ships'])
    vote_need = ceil(len(per_model) * vote_ratio)
    fused_dets: List[Dict[str, Any]] = []

    def cluster_and_vote(target_filter_fn):
        props_box = []
        props_meta = []  # (score, label, label_id, model)
        for model_index, detections in enumerate(per_model):
            for detection in detections:
                if target_filter_fn(detection['label']):
                    props_box.append(
                        np.asarray(detection['box7d'], dtype=np.float32)
                    )
                    props_meta.append(
                        (
                            float(detection['score']),
                            detection['label'],
                            int(detection.get('label_id', -1)),
                            model_index,
                        )
                    )
        if not props_box:
            return

        props_box_array = np.stack(props_box, axis=0)
        order = np.argsort([-metadata[0] for metadata in props_meta])
        used = np.zeros((len(order),), dtype=bool)
        for ordered_index, _ in enumerate(order):
            if used[ordered_index]:
                continue
            seed_index = order[ordered_index]
            cluster_indices = [ordered_index]
            remaining_mask = ~used
            remaining_mask[:ordered_index + 1] = False
            remaining_indices = np.where(remaining_mask)[0]
            if remaining_indices.size > 0:
                other_boxes = props_box_array[order[remaining_indices]]
                ious = bev_iou_seed_vs_rest(
                    props_box_array[seed_index], other_boxes
                )
                cluster_indices.extend(
                    remaining_indices[ious >= iou_thr].tolist()
                )

            support_models = {
                props_meta[order[cluster_index]][3]
                for cluster_index in cluster_indices
            }
            if len(support_models) >= vote_need:
                best_cluster_index = max(
                    cluster_indices,
                    key=lambda index: props_meta[order[index]][0],
                )
                best_index = order[best_cluster_index]
                score, label, label_id, _ = props_meta[best_index]
                fused_dets.append(
                    {
                        "box7d": [
                            float(value) for value in props_box_array[best_index]
                        ],
                        "score": float(score),
                        "label_id": int(label_id),
                        "label": label,
                    }
                )
            for cluster_index in cluster_indices:
                used[cluster_index] = True

    cluster_and_vote(lambda label: label == lock_gate_label)
    num_gate_detections = len(fused_dets)
    cluster_and_vote(lambda label: is_ship_label(label, ship_labels))
    return {
        "detections": fused_dets,
        "lock_gate_present": any(
            detection['label'] == lock_gate_label for detection in fused_dets
        ),
        "ship_present": len(fused_dets) > num_gate_detections,
    }


def fuse_3d_jsons(
    json_paths: List[str],
    labels_cfg: Dict[str, Any],
    iou_thr: float = 0.7,
    vote_ratio: float = 0.5,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    对多个模型的 3D 检测结果进行 BEV 旋转 IoU 多数投票融合。
    规则：
      - 仅对 'lock_gate' 与 'ships' 两大类进行投票融合；
      - 旋转 BEV IoU >= iou_thr 视为“同一目标”的同一簇；
      - 若同一簇中支持的模型数 >= ceil(num_models * vote_ratio)，则保留该簇，
        并用簇内分数最高的框作为代表写入最终结果；
      - ship_present / lock_gate_present 根据融合后的结果判断。
    """
    # 读取所有模型结果
    all_res: List[Dict[str, Any]] = []
    for p in json_paths:
        with open(p, 'r', encoding='utf-8') as f:
            all_res.append(json.load(f))

    # 帧 ID 并集
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

        frame_result = fuse_frame_detections(
            per_model=per_model,
            labels_cfg=labels_cfg,
            iou_thr=iou_thr,
            vote_ratio=vote_ratio,
        )
        fused[fid] = {"points_path": points_path, **frame_result}

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
    parser.add_argument("--pcdet_tools", type=str, default="path/to/OpenPCDet/tools",
                        help="Path to OpenPCDet 'tools' dir so relative _BASE_CONFIG_ resolves correctly")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.cfg).expanduser().resolve()

    def resolve_repo_path(value: str) -> Path:
        path = Path(os.path.expandvars(value)).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    # 读取 Water-level-monitoring YAML 配置
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 绝对化输入与输出路径（避免临时 chdir 影响）
    points_dir = Path(
        os.path.expandvars(cfg["data"]["points_dir"])
    ).expanduser().resolve()
    out_root_base = cfg.get("output", {}).get("root", "outputs")
    out_root = resolve_repo_path(out_root_base) / "det3d"
    out_root.mkdir(parents=True, exist_ok=True)

    logger = common_utils.create_logger()
    logger.info("=========== PCDet Inference from config.yaml ===========")

    pcdet_tools = Path(os.path.expandvars(args.pcdet_tools)).expanduser().resolve()
    if not pcdet_tools.is_dir():
        raise RuntimeError(f"PCDet tools dir not found: {pcdet_tools}")

    model_json_paths: List[str] = []

    # 在 PCDet tools 目录下进行配置解析与建模（关键：_BASE_CONFIG_ 是相对该目录的）
    with temp_chdir(pcdet_tools):
        for model_cfg in cfg.get("pcdet_models", []):
            name = model_cfg["name"]
            cfg_file = str(resolve_repo_path(model_cfg["config"]))
            ckpt = str(resolve_repo_path(model_cfg["checkpoint"]))
            score_thr = float(model_cfg.get("score_thr", 0.0))

            logger.info(f"[{name}] Loading config={cfg_file}, ckpt={ckpt}")

            # 加载 PCDet 模型配置（会解析 _BASE_CONFIG_）
            cfg_from_yaml_file(cfg_file, PC_CFG)

            # 构建数据集
            demo_dataset = DemoDataset(
                dataset_cfg=PC_CFG.DATA_CONFIG,
                class_names=PC_CFG.CLASS_NAMES,
                root_path=points_dir,
                ext=args.ext,
                logger=logger
            )

            # 构建与加载模型
            model = build_network(model_cfg=PC_CFG.MODEL,
                                  num_class=len(PC_CFG.CLASS_NAMES),
                                  dataset=demo_dataset)
            model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
            model.cuda()
            model.eval()

            id2name = build_label_name_map(PC_CFG.CLASS_NAMES)
            results: Dict[str, Any] = {}

            # 逐帧推理
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
                        # Store a portable reference in exported JSON rather than
                        # leaking the machine-local dataset root.
                        "points_path": (
                            "${TROUT_ROOT}/points_test/"
                            + Path(demo_dataset.sample_file_list[idx]).name
                        ),
                        "detections": detections
                    }

            # 保存该模型的结果（绝对路径，避免 chdir 影响）
            out_path = out_root / f"{name}.det3d.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"[{name}] Saved results to {out_path}")
            model_json_paths.append(str(out_path))

    # 融合输出
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
