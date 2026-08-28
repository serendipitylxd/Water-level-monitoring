#!/usr/bin/env python3
"""Quantify cadence, maximum observed rate, and reference bound for 8k labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/reference_audit",
    )
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 8000 or {row["source"] for row in records} != {"legacy_test"}:
        raise RuntimeError("Reference audit requires the 8,000-frame detector-test shard")

    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        grouped[int(row["operation_id"])].append(
            (
                datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT).timestamp(),
                float(row["water_level"]),
            )
        )
    cadence_rows, all_gaps, all_rates = [], [], []
    for operation, values in sorted(grouped.items()):
        values.sort()
        times = np.asarray([value[0] for value in values], dtype=np.float64)
        levels = np.asarray([value[1] for value in values], dtype=np.float64)
        gaps = np.diff(times)
        rates = np.abs(np.diff(levels)) / gaps
        all_gaps.extend(gaps.tolist())
        all_rates.extend((float(rate), operation) for rate in rates)
        cadence_rows.append(
            {
                "operation_id": operation,
                "num_frames": len(values),
                "mean_gap_sec": float(np.mean(gaps)),
                "median_gap_sec": float(np.median(gaps)),
                "min_gap_sec": float(np.min(gaps)),
                "max_gap_sec": float(np.max(gaps)),
                "fraction_1p95_to_2p05": float(np.mean((gaps >= 1.95) & (gaps <= 2.05))),
                "maximum_abs_rate_m_per_s": float(np.max(rates)),
            }
        )

    gaps = np.asarray(all_gaps, dtype=np.float64)
    max_rate_m_per_s, max_rate_operation = max(all_rates)
    max_rate_cm_per_s = 100.0 * max_rate_m_per_s
    components = [
        {
            "component": "Image interpolation",
            "basis": "half of coarsest calibrated increment: (5 cm / 24 px) / 2",
            "bound_cm": (5.0 / 24.0) / 2.0,
        },
        {
            "component": "Reader agreement",
            "basis": "half of the 0.5 cm two-reader acceptance threshold",
            "bound_cm": 0.5 / 2.0,
        },
        {
            "component": "Timestamp-based video-frame matching",
            "basis": f"{max_rate_cm_per_s:.6f} cm/s times 1/120 s",
            "bound_cm": max_rate_cm_per_s / 120.0,
        },
        {
            "component": "Stored-value rounding",
            "basis": "half of the 0.001 m stored-label increment",
            "bound_cm": 0.05,
        },
        {
            "component": "Gauge-to-LiDAR-ROI separation",
            "basis": "gauge lies inside the LiDAR-observed wall-belt ROI",
            "bound_cm": 0.0,
        },
    ]
    total = float(sum(row["bound_cm"] for row in components))
    components.append(
        {
            "component": "Conservative linear total",
            "basis": "sum of quantified absolute half-widths",
            "bound_cm": total,
        }
    )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_csv(output / "cadence_and_rate_by_operation.csv", cadence_rows)
    atomic_csv(output / "uncertainty_components.csv", components)
    audit = {
        "protocol": "camera-read-staff-gauge-operational-reference",
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "num_water_level_frames": len(records),
        "num_operations": len(grouped),
        "native_lidar_rate_hz": 10.0,
        "revised_protocol_measured_cadence": {
            "median_within_operation_gap_sec": float(np.median(gaps)),
            "minimum_gap_sec": float(np.min(gaps)),
            "maximum_gap_sec": float(np.max(gaps)),
            "fraction_1p95_to_2p05": float(np.mean((gaps >= 1.95) & (gaps <= 2.05))),
            "interpretation": "approximately one retained frame every 2 s; longer gaps reflect missing scans",
        },
        "staff_gauge": {
            "physical_graduation_cm": 5.0,
            "camera_resolution_pixels": [1920, 1080],
            "camera_frame_rate_hz": 60.0,
            "calibrated_pixels_per_local_5cm_interval": [24, 27],
            "reading_method": "interval-specific linear pixel interpolation at the waterline",
            "annotation_platform": "CVAT",
            "annotators": 3,
            "annotation_rule": (
                "two independent readings; mean used when difference <=0.5 cm; "
                "third annotator independently re-read every frame with difference >0.5 cm"
            ),
            "third_annotator_coverage_of_triggered_frames_percent": 100.0,
            "stored_increment_m": 0.001,
        },
        "synchronization": {
            "clock": "camera and LiDAR timestamps use the same network-disciplined host-computer clock",
            "matching": "nearest 60-fps video frame to each retained LiDAR timestamp",
            "conditional_nearest_frame_half_width_sec": 1.0 / 120.0,
            "fixed_device_latency_independently_calibrated": False,
        },
        "spatial_relation": {
            "description": "staff gauge lies on the wall belt inside the LiDAR observation ROI",
            "minimum_gauge_to_roi_separation_m": 0.0,
        },
        "maximum_observed_abs_rate": {
            "m_per_s": max_rate_m_per_s,
            "cm_per_s": max_rate_cm_per_s,
            "operation_id": max_rate_operation,
            "calculation": "adjacent retained labels within each operation divided by actual timestamp gap",
        },
        "operational_bound": {
            "linear_total_cm": total,
            "rounded_reported_bound_cm": round(total, 2),
            "interpretation": "protocol-derived operational estimate, not a traceable calibration certificate",
            "unquantified_terms": [
                "fixed camera-LiDAR device latency",
                "common-mode staff-gauge calibration error",
                "staff-gauge installation error",
            ],
        },
    }
    atomic_json(output / "reference_protocol_audit.json", audit)
    print(
        f"[done] frames=8000 median_gap={np.median(gaps):.6f}s "
        f"max_rate={max_rate_cm_per_s:.3f}cm/s bound=+/-{total:.3f}cm"
    )


if __name__ == "__main__":
    main()
