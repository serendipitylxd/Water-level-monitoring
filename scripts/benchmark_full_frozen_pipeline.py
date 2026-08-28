#!/usr/bin/env python3
"""Measure the actual one-GPU six-detector-to-filter end-to-end pipeline.

All six frozen detector checkpoints are loaded simultaneously.  Each retained
frame is read once, prepared for every detector, passed through all six models
sequentially on the same RTX 4080, fused in memory, cleaned, converted to the
wall-profile feature, evaluated by the frozen Random Forest, stabilized
causally, and updated by the frozen Kalman filter.  Checkpoint/model loading is
excluded; point-cloud disk I/O is included.  Batch size is one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_3d_pcdet import (  # noqa: E402
    DemoDataset,
    build_label_name_map,
    fuse_frame_detections,
    temp_chdir,
)
from utils.data import (  # noqa: E402
    OBB,
    extract_wall_sequence,
    load_operation_manifest,
    points_in_obb_mask,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def milliseconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1.0e6


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean_ms": float(np.mean(array)),
        "sample_sd_ms": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "minimum_ms": float(np.min(array)),
        "maximum_ms": float(np.max(array)),
    }


class OnlineStates:
    def __init__(self, alpha: float, kalman: dict):
        self.alpha = float(alpha)
        self.base_r = float(kalman["base_R"])
        self.q_pos = float(kalman["q_pos"])
        self.q_vel = float(kalman["q_vel"])
        self.operation = None
        self.causal_value = None
        self.x = None
        self.p = None
        self.previous_time = None

    def reset_operation(self, operation: int) -> None:
        self.operation = int(operation)
        self.causal_value = None
        self.x = None
        self.p = None
        self.previous_time = None

    def causal(self, prediction: float) -> float:
        if self.causal_value is None:
            self.causal_value = float(prediction)
        else:
            self.causal_value = self.alpha * float(prediction) + (
                1.0 - self.alpha
            ) * self.causal_value
        return self.causal_value

    def kalman(self, timestamp: float, observation: float) -> float:
        if self.x is None:
            self.x = np.asarray([[observation], [0.0]], dtype=np.float64)
            self.p = np.diag([0.05 ** 2, 0.01 ** 2])
            self.previous_time = float(timestamp)
            return float(observation)
        delta = float(timestamp) - float(self.previous_time)
        if delta > 10.0:
            self.x = np.asarray([[observation], [0.0]], dtype=np.float64)
            self.p = np.diag([0.05 ** 2, 0.01 ** 2])
            self.previous_time = float(timestamp)
            return float(observation)
        delta = max(1.0e-3, delta)
        self.previous_time = float(timestamp)
        transition = np.asarray([[1.0, delta], [0.0, 1.0]], dtype=np.float64)
        process = np.asarray(
            [[self.q_pos * delta, 0.0], [0.0, self.q_vel * delta]],
            dtype=np.float64,
        )
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + process
        observation_matrix = np.asarray([[1.0, 0.0]], dtype=np.float64)
        observation_noise = np.asarray([[max(1.0e-6, self.base_r ** 2)]])
        innovation = np.asarray([[observation]]) - observation_matrix @ self.x
        innovation_covariance = (
            observation_matrix @ self.p @ observation_matrix.T + observation_noise
        )
        gain = self.p @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(2) - gain @ observation_matrix) @ self.p
        return float(self.x[0, 0])


def load_detector_stack(config: dict, pcdet_tools: Path, logger):
    points_dir = Path(
        os.path.expandvars(config["data"]["points_dir"])
    ).expanduser().resolve()
    stack = []
    with temp_chdir(str(pcdet_tools)):
        for specification in config["pcdet_models"]:
            config_path = Path(
                os.path.expandvars(specification["config"])
            ).expanduser()
            checkpoint_path = Path(
                os.path.expandvars(specification["checkpoint"])
            ).expanduser()
            if not config_path.is_absolute():
                config_path = REPO_ROOT / config_path
            if not checkpoint_path.is_absolute():
                checkpoint_path = REPO_ROOT / checkpoint_path
            config_path = config_path.resolve()
            checkpoint_path = checkpoint_path.resolve()
            local_cfg = EasyDict()
            cfg_from_yaml_file(str(config_path), local_cfg)
            dataset = DemoDataset(
                dataset_cfg=local_cfg.DATA_CONFIG,
                class_names=local_cfg.CLASS_NAMES,
                root_path=points_dir,
                ext=".bin",
                logger=logger,
            )
            model = build_network(
                model_cfg=local_cfg.MODEL,
                num_class=len(local_cfg.CLASS_NAMES),
                dataset=dataset,
            )
            model.load_params_from_file(
                filename=str(checkpoint_path), logger=logger, to_cpu=False
            )
            model.cuda().eval()
            stack.append(
                {
                    "name": specification["name"],
                    "dataset": dataset,
                    "model": model,
                    "score_threshold": float(specification.get("score_thr", 0.0)),
                    "id_to_name": build_label_name_map(local_cfg.CLASS_NAMES),
                    "config": str(config_path),
                    "checkpoint": str(checkpoint_path),
                }
            )
            logger.info("[runtime] loaded %s", specification["name"])
    torch.cuda.synchronize()
    return stack


def detector_predictions(stack: list[dict], raw_points: np.ndarray):
    per_model, preprocess_ms, inference_ms = [], {}, {}
    for item in stack:
        start = time.perf_counter_ns()
        prepared = item["dataset"].prepare_data(
            data_dict={"points": raw_points.copy(), "frame_id": "runtime"}
        )
        batch = item["dataset"].collate_batch([prepared])
        preprocess_end = time.perf_counter_ns()
        torch.cuda.synchronize()
        inference_start = time.perf_counter_ns()
        load_data_to_gpu(batch)
        prediction_dicts, _ = item["model"].forward(batch)
        prediction = prediction_dicts[0]
        boxes = prediction.get("pred_boxes", torch.empty((0, 7), device="cuda")).detach().cpu().numpy()
        scores = prediction.get("pred_scores", torch.empty((0,), device="cuda")).detach().cpu().numpy()
        labels = prediction.get(
            "pred_labels", torch.empty((0,), dtype=torch.long, device="cuda")
        ).detach().cpu().numpy().astype(int)
        torch.cuda.synchronize()
        inference_end = time.perf_counter_ns()
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < item["score_threshold"]:
                continue
            detections.append(
                {
                    "box7d": [float(value) for value in box.tolist()],
                    "score": float(score),
                    "label_id": int(label),
                    "label": item["id_to_name"].get(int(label), str(label)),
                }
            )
        per_model.append(detections)
        preprocess_ms[item["name"]] = milliseconds(start, preprocess_end)
        inference_ms[item["name"]] = milliseconds(inference_start, inference_end)
    return per_model, preprocess_ms, inference_ms


def selected_segments(records: list[dict], warmup: int, measured: int):
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record["operation_id"]), []).append(record)
    output = []
    for operation, values in sorted(grouped.items()):
        values.sort(key=lambda row: (float(row["tsec"]), row["sample_id"]))
        needed = warmup + measured
        if len(values) < needed:
            raise RuntimeError(f"Operation {operation} has fewer than {needed} frames")
        start = (len(values) - needed) // 2
        for local_index, record in enumerate(values[start:start + needed]):
            output.append((record, local_index < warmup))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=REPO_ROOT / "configs/infer_3d_pcdet.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv",
    )
    parser.add_argument(
        "--rf-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs_operation_split/test_only_8000/rf_tuned_robust_anchor"
        ),
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/feature_cache/waterlevel_test_only_default.npz",
    )
    parser.add_argument(
        "--pcdet-tools",
        type=Path,
        default=Path("path/to/OpenPCDet/tools"),
        help="Path to the OpenPCDet tools directory.",
    )
    parser.add_argument("--warmup-per-operation", type=int, default=10)
    parser.add_argument("--measured-per-operation", type=int, default=20)
    parser.add_argument(
        "--feature-equivalence-tolerance-m",
        type=float,
        default=0.005,
        help=(
            "Maximum accepted live-versus-cache wall-profile difference. "
            "This audit allows millimetre-scale GPU detector nondeterminism."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/runtime_full_pipeline",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark")
    config_path = args.cfg.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    manifest_path = args.manifest.expanduser().resolve()
    test_records = load_operation_manifest(str(manifest_path), split="test")
    segments = selected_segments(
        test_records, args.warmup_per_operation, args.measured_per_operation
    )

    rf_dir = args.rf_dir.expanduser().resolve()
    frozen_path = rf_dir / "frozen_selection.json"
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    model_path = Path(frozen["model_path"]).resolve()
    if sha256_file(model_path) != frozen["model_sha256"]:
        raise RuntimeError("Frozen Random Forest hash mismatch")
    with model_path.open("rb") as handle:
        random_forest = pickle.load(handle)
    states = OnlineStates(
        frozen["causal_stabilizer_selection"]["alpha"],
        frozen["kalman_selection"]["parameters"],
    )

    feature_cache_path = args.feature_cache.expanduser().resolve()
    with np.load(str(feature_cache_path), allow_pickle=False) as cache:
        cache_ids = np.asarray(cache["sample_ids"]).astype(str)
        cache_sequences = np.asarray(cache["sequences"], dtype=np.float32)
        cache_masks = np.asarray(cache["masks"], dtype=bool)
    cache_index = {sample_id: index for index, sample_id in enumerate(cache_ids)}

    logger = common_utils.create_logger()
    pcdet_tools = Path(
        os.path.expandvars(str(args.pcdet_tools))
    ).expanduser().resolve()
    stack = load_detector_stack(config, pcdet_tools, logger)
    if len(stack) != 6:
        raise RuntimeError(f"Expected six detectors, loaded {len(stack)}")
    loaded_memory = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    water = config.get("waterlevel") or {
        "wall_band_x_half": 1.0,
        "y_bin": 1.0,
        "quantile_low": 0.12,
        "qc_min_pts": 10,
        "inflate_xy": 0.2,
        "inflate_z": 0.2,
    }
    geometry = config["geometry"]
    rows, maximum_feature_difference = [], 0.0
    current_operation = None
    with torch.no_grad():
        for record, is_warmup in segments:
            operation = int(record["operation_id"])
            if operation != current_operation:
                states.reset_operation(operation)
                current_operation = operation
            total_start = time.perf_counter_ns()

            load_start = time.perf_counter_ns()
            raw = np.fromfile(record["points_path"], dtype=np.float32).reshape(-1, 4)
            load_end = time.perf_counter_ns()

            per_model, preprocess_by_model, inference_by_model = detector_predictions(stack, raw)
            detectors_end = time.perf_counter_ns()

            fusion_start = time.perf_counter_ns()
            fused = fuse_frame_detections(
                per_model, config["labels"], iou_thr=0.70, vote_ratio=0.50
            )
            fusion_end = time.perf_counter_ns()

            removal_start = time.perf_counter_ns()
            boxes = [
                OBB(*detection["box7d"], detection["label"])
                for detection in fused["detections"]
            ]
            xyz = raw[:, :3]
            removal_mask = points_in_obb_mask(
                xyz, boxes, float(water["inflate_xy"]), float(water["inflate_z"])
            )
            cleaned = xyz[~removal_mask]
            removal_end = time.perf_counter_ns()

            feature_start = time.perf_counter_ns()
            sequence, mask = extract_wall_sequence(
                cleaned,
                float(geometry["chamber_x_range"][0]),
                float(geometry["chamber_x_range"][1]),
                float(geometry["chamber_y_range"][0]),
                float(geometry["chamber_y_range"][1]),
                float(water["wall_band_x_half"]),
                float(water["y_bin"]),
                float(water["quantile_low"]),
                int(water["qc_min_pts"]),
            )
            feature_end = time.perf_counter_ns()

            cache_position = cache_index[record["sample_id"]]
            maximum_feature_difference = max(
                maximum_feature_difference,
                float(np.max(np.abs(sequence - cache_sequences[cache_position]))),
            )
            if not np.array_equal(mask, cache_masks[cache_position]):
                raise RuntimeError(f"Live/cache feature mask mismatch: {record['sample_id']}")

            rf_start = time.perf_counter_ns()
            raw_prediction = float(
                random_forest.predict(sequence[None, ...], key_padding_mask=mask[None, ...])[0]
            )
            rf_end = time.perf_counter_ns()
            causal_start = time.perf_counter_ns()
            primary_prediction = states.causal(raw_prediction)
            causal_end = time.perf_counter_ns()
            kalman_start = time.perf_counter_ns()
            filtered_prediction = states.kalman(float(record["tsec"]), primary_prediction)
            kalman_end = time.perf_counter_ns()
            total_end = kalman_end

            if is_warmup:
                continue
            row = {
                "sample_id": record["sample_id"],
                "operation_id": operation,
                "num_input_points": len(raw),
                "num_fused_boxes": len(boxes),
                "num_removed_points": int(removal_mask.sum()),
                "point_loading_ms": milliseconds(load_start, load_end),
                "six_detector_preprocessing_ms": sum(preprocess_by_model.values()),
                "six_detector_inference_ms": sum(inference_by_model.values()),
                "fusion_ms": milliseconds(fusion_start, fusion_end),
                "point_removal_ms": milliseconds(removal_start, removal_end),
                "feature_construction_ms": milliseconds(feature_start, feature_end),
                "random_forest_ms": milliseconds(rf_start, rf_end),
                "causal_stabilizer_ms": milliseconds(causal_start, causal_end),
                "kalman_filter_ms": milliseconds(kalman_start, kalman_end),
                "end_to_end_ms": milliseconds(total_start, total_end),
                "rf_raw_output_m": raw_prediction,
                "primary_output_m": primary_prediction,
                "kf_output_m": filtered_prediction,
            }
            for model_name, value in preprocess_by_model.items():
                row[f"preprocess_{model_name}_ms"] = value
            for model_name, value in inference_by_model.items():
                row[f"inference_{model_name}_ms"] = value
            rows.append(row)
            print(
                f"[measured {len(rows):03d}] {record['sample_id']} "
                f"end-to-end={row['end_to_end_ms']:.2f} ms",
                flush=True,
            )

    expected = 5 * int(args.measured_per_operation)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} measured frames, got {len(rows)}")
    feature_equivalence_passed = maximum_feature_difference <= float(
        args.feature_equivalence_tolerance_m
    )
    if not feature_equivalence_passed:
        print(
            "[audit-warning] live/cache feature difference "
            f"{maximum_feature_difference:.6f} m exceeds "
            f"{args.feature_equivalence_tolerance_m:.6f} m; runtime rows are "
            "retained because live detector outputs define the measured pipeline.",
            flush=True,
        )

    stage_columns = [
        "point_loading_ms",
        "six_detector_preprocessing_ms",
        "six_detector_inference_ms",
        "fusion_ms",
        "point_removal_ms",
        "feature_construction_ms",
        "random_forest_ms",
        "causal_stabilizer_ms",
        "kalman_filter_ms",
        "end_to_end_ms",
    ] + [
        f"inference_{item['name']}_ms" for item in stack
    ]
    summary = {column: describe([float(row[column]) for row in rows]) for column in stage_columns}
    summary["throughput_fps_from_mean_end_to_end"] = 1000.0 / summary["end_to_end_ms"]["mean_ms"]

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "per_frame_runtime.csv", rows)
    audit = {
        "protocol": "measured-full-frozen-six-detector-rf-causal-kf-pipeline",
        "measurement_scope": (
            "point-cloud disk read, six detector input preparations and sequential "
            "inferences on one GPU, fusion, point removal, wall-profile construction, "
            "Random Forest, causal stabilizer, and Kalman update"
        ),
        "excluded": ["process startup", "configuration parsing", "checkpoint/model loading"],
        "batch_size": 1,
        "execution": "six frozen detectors resident together and executed sequentially on one GPU",
        "estimated_parallel_latency_used": False,
        "warmup_frames_per_operation": int(args.warmup_per_operation),
        "measured_frames_per_operation": int(args.measured_per_operation),
        "measured_frames": len(rows),
        "operations": sorted({int(row["operation_id"]) for row in rows}),
        "feature_cache_equivalence_max_abs_m": maximum_feature_difference,
        "feature_cache_equivalence_tolerance_m": float(
            args.feature_equivalence_tolerance_m
        ),
        "feature_cache_equivalence_passed": feature_equivalence_passed,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_loaded_models_memory_mib": loaded_memory / (1024 ** 2),
        "gpu_peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "cpu": platform.processor() or platform.machine(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "feature_cache": {"path": str(feature_cache_path), "sha256": sha256_file(feature_cache_path)},
        "random_forest": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "causal_alpha": states.alpha,
        "kalman_parameters": {
            "base_R": states.base_r,
            "q_pos": states.q_pos,
            "q_vel": states.q_vel,
        },
        "fusion": {"bev_iou_threshold": 0.70, "vote_ratio": 0.50, "votes_required": 3},
        "detectors": [
            {
                "name": item["name"],
                "score_threshold": item["score_threshold"],
                "config": item["config"],
                "config_sha256": sha256_file(Path(item["config"])),
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": sha256_file(Path(item["checkpoint"])),
            }
            for item in stack
        ],
        "summary": summary,
    }
    atomic_json(output / "runtime_summary.json", audit)
    print(
        f"[done] mean={summary['end_to_end_ms']['mean_ms']:.2f} ms "
        f"p95={summary['end_to_end_ms']['p95_ms']:.2f} ms "
        f"fps={summary['throughput_fps_from_mean_end_to_end']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
