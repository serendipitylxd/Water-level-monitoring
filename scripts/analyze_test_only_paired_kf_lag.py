#!/usr/bin/env python3
"""Paired primary-output versus Kalman-output lag for the 8,000-frame protocol.

The model evaluation cadence is derived from actual retained timestamps.  The
two series are interpolated only for correlation analysis on a configurable
grid (2 s by default, matching the retained test-shard median); model metrics
remain those calculated at the original irregular timestamps.  Positive lag
means that the Kalman output occurs later than its own input series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"
EXPECTED_MODELS = ("Linear Regression", "MLP", "Random Forest", "HGRN")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def timestamp_seconds(value: str) -> float:
    return datetime.strptime(value, TIMESTAMP_FORMAT).timestamp()


def normalized_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) - float(np.mean(first))
    second = np.asarray(second, dtype=np.float64) - float(np.mean(second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator > 0.0 else np.nan


def paired_lag(
    times: np.ndarray,
    primary: np.ndarray,
    filtered: np.ndarray,
    grid_step_sec: float,
    max_lag_sec: float,
) -> tuple[float, float, int]:
    order = np.argsort(times, kind="stable")
    times, primary, filtered = times[order], primary[order], filtered[order]
    if np.any(np.diff(times) <= 0.0):
        raise RuntimeError("Timestamps are not strictly increasing within an operation")
    start = np.ceil(times[0] / grid_step_sec) * grid_step_sec
    end = np.floor(times[-1] / grid_step_sec) * grid_step_sec
    grid = np.arange(start, end + grid_step_sec * 0.5, grid_step_sec)
    max_steps = int(np.floor(max_lag_sec / grid_step_sec))
    if len(grid) <= 2 * max_steps + 2:
        raise RuntimeError("Operation is too short for the requested lag window")
    primary_grid = np.interp(grid, times, primary)
    filtered_grid = np.interp(grid, times, filtered)
    candidates = []
    for steps in range(-max_steps, max_steps + 1):
        if steps > 0:
            first, second = primary_grid[:-steps], filtered_grid[steps:]
        elif steps < 0:
            first, second = primary_grid[-steps:], filtered_grid[:steps]
        else:
            first, second = primary_grid, filtered_grid
        correlation = normalized_correlation(first, second)
        if np.isfinite(correlation):
            candidates.append((steps * grid_step_sec, correlation))
    peak = max(value for _, value in candidates)
    tied = [item for item in candidates if abs(item[1] - peak) <= 1.0e-12]
    lag, peak = min(tied, key=lambda item: (abs(item[0]), item[0]))
    return float(lag), float(peak), int(len(grid))


def cadence(records: list[dict[str, str]], scope: str) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    for row in records:
        grouped.setdefault(int(row["operation_id"]), []).append(
            timestamp_seconds(row["timestamp"])
        )
    rows, all_gaps = [], []
    for operation, values in sorted(grouped.items()):
        gaps = np.diff(np.sort(values))
        all_gaps.extend(gaps.tolist())
        rows.append(
            {
                "scope": f"operation_{operation}",
                "num_frames": len(values),
                "num_gaps": len(gaps),
                "mean_gap_sec": float(np.mean(gaps)),
                "median_gap_sec": float(np.median(gaps)),
                "min_gap_sec": float(np.min(gaps)),
                "max_gap_sec": float(np.max(gaps)),
                "fraction_1p95_to_2p05": float(np.mean((gaps >= 1.95) & (gaps <= 2.05))),
            }
        )
    gaps = np.asarray(all_gaps, dtype=np.float64)
    rows.append(
        {
            "scope": scope,
            "num_frames": len(records),
            "num_gaps": len(gaps),
            "mean_gap_sec": float(np.mean(gaps)),
            "median_gap_sec": float(np.median(gaps)),
            "min_gap_sec": float(np.min(gaps)),
            "max_gap_sec": float(np.max(gaps)),
            "fraction_1p95_to_2p05": float(np.mean((gaps >= 1.95) & (gaps <= 2.05))),
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict) -> None:
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
        "--input",
        nargs=3,
        action="append",
        required=True,
        metavar=("MODEL_NAME", "PREDICTIONS_CSV", "PRIMARY_COLUMN"),
    )
    parser.add_argument("--grid-step-sec", type=float, default=2.0)
    parser.add_argument("--max-lag-sec", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/kalman_lag_paired",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = read_csv(manifest_path)
    test_manifest = [row for row in manifest if row["split"] == "test"]
    expected_operations = sorted({int(row["operation_id"]) for row in test_manifest})
    expected_keys = sorted(
        (row["sample_id"], int(row["operation_id"]), row["timestamp"], float(row["water_level"]))
        for row in test_manifest
    )
    supplied_names = tuple(item[0] for item in args.input)
    if set(supplied_names) != set(EXPECTED_MODELS) or len(supplied_names) != len(EXPECTED_MODELS):
        raise RuntimeError(f"Expected exactly these models: {EXPECTED_MODELS}; got {supplied_names}")

    per_operation_rows, inputs = [], {}
    for model_name, path_text, primary_column in args.input:
        path = Path(path_text).expanduser().resolve()
        rows = read_csv(path)
        required = {"sample_id", "operation_id", "timestamp", "gt", primary_column, "pred_kf"}
        missing = required - set(rows[0])
        if missing:
            raise RuntimeError(f"{path} lacks columns {sorted(missing)}")
        keys = sorted(
            (row["sample_id"], int(row["operation_id"]), row["timestamp"], float(row["gt"]))
            for row in rows
        )
        if keys != expected_keys:
            raise RuntimeError(f"{model_name}: samples/timestamps/reference labels differ from manifest")
        grouped: dict[int, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(int(row["operation_id"]), []).append(row)
        if sorted(grouped) != expected_operations:
            raise RuntimeError(f"{model_name}: unexpected test operations {sorted(grouped)}")
        for operation, operation_rows in sorted(grouped.items()):
            times = np.asarray([timestamp_seconds(row["timestamp"]) for row in operation_rows])
            primary = np.asarray([float(row[primary_column]) for row in operation_rows])
            filtered = np.asarray([float(row["pred_kf"]) for row in operation_rows])
            lag, peak, points = paired_lag(
                times, primary, filtered, args.grid_step_sec, args.max_lag_sec
            )
            per_operation_rows.append(
                {
                    "model": model_name,
                    "operation_id": operation,
                    "num_frames": len(operation_rows),
                    "num_correlation_grid_points": points,
                    "primary_output_column": primary_column,
                    "primary_output_lag_sec": 0.0,
                    "kf_output_lag_sec": lag,
                    "extra_kf_lag_vs_primary_sec": lag,
                    "peak_normalized_correlation": peak,
                }
            )
        inputs[model_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "primary_column": primary_column,
            "num_rows": len(rows),
        }

    summary_rows = []
    for model_name in supplied_names:
        rows = [row for row in per_operation_rows if row["model"] == model_name]
        primary = np.asarray([row["primary_output_lag_sec"] for row in rows])
        filtered = np.asarray([row["kf_output_lag_sec"] for row in rows])
        extra = np.asarray([row["extra_kf_lag_vs_primary_sec"] for row in rows])
        summary_rows.append(
            {
                "model": model_name,
                "paired_operations": len(rows),
                "primary_output_lag_mean_sec": float(np.mean(primary)),
                "primary_output_lag_sample_sd_sec": float(np.std(primary, ddof=1)),
                "kf_output_lag_mean_sec": float(np.mean(filtered)),
                "kf_output_lag_sample_sd_sec": float(np.std(filtered, ddof=1)),
                "extra_kf_lag_mean_sec": float(np.mean(extra)),
                "extra_kf_lag_sample_sd_sec": float(np.std(extra, ddof=1)),
                "minimum_peak_normalized_correlation": min(
                    row["peak_normalized_correlation"] for row in rows
                ),
            }
        )
    for row in summary_rows:
        if abs(
            row["kf_output_lag_mean_sec"]
            - row["primary_output_lag_mean_sec"]
            - row["extra_kf_lag_mean_sec"]
        ) > 1.0e-12:
            raise RuntimeError("Lag arithmetic audit failed")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "paired_kf_lag_by_operation.csv", per_operation_rows)
    write_csv(output / "paired_kf_lag_summary.csv", summary_rows)
    write_csv(output / "test_timestamp_cadence.csv", cadence(test_manifest, "test_operations_14_18"))
    write_csv(output / "full_timestamp_cadence.csv", cadence(manifest, "all_operations_1_18"))
    write_json(
        output / "analysis_audit.json",
        {
            "protocol": "paired-primary-versus-kalman-output-lag",
            "num_water_level_frames": len(manifest),
            "num_test_frames": len(test_manifest),
            "test_operations": expected_operations,
            "native_lidar_rate_hz": 10.0,
            "retained_cadence": "approximately one frame every 2 s in this 8,000-frame shard",
            "correlation_grid_step_sec": args.grid_step_sec,
            "max_lag_sec": args.max_lag_sec,
            "lag_definition": (
                "Within each operation, interpolate the primary and its KF output on the stated "
                "correlation grid, mean-centre both series, and select the offset maximizing "
                "normalized cross-correlation. Positive lag denotes a delayed KF output."
            ),
            "metrics_evaluated_at_original_timestamps": True,
            "operation_boundary_reset": True,
            "random_forest_primary_output": "pred_causal",
            "arithmetic_identity": "extra KF lag = KF-output lag - primary-output lag",
            "arithmetic_audit_passed": True,
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "inputs": inputs,
        },
    )
    for row in summary_rows:
        print(
            f"[{row['model']}] extra KF lag "
            f"{row['extra_kf_lag_mean_sec']:.2f} +/- "
            f"{row['extra_kf_lag_sample_sd_sec']:.2f} s"
        )
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
