#!/usr/bin/env python3
"""Run 18-fold LOOCV for the frozen anchored Random Forest configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.tune_anchored_random_forest_operation_split import (  # noqa: E402
    causal_exponential_by_operation,
    extended_metrics,
    fit_forest,
    load_rf_inputs,
    macro_rmse,
    predict_forest,
)
from scripts.tune_random_forest_operation_split import filter_by_operation  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_fold_manifest(
    source_rows: list[dict[str, str]], fields: list[str], operation: int, path: Path
) -> None:
    rows = []
    for source in source_rows:
        row = dict(source)
        row["split"] = "test" if int(row["operation_id"]) == operation else "train"
        rows.append(row)
    atomic_csv(path, [{field: row[field] for field in fields} for row in rows])


def metric_columns(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_{key}": metrics[key]
        for key in ("MAE", "RMSE", "Bias", "MaxAE", "Corr")
    }


def summarize(rows: list[dict]) -> dict:
    result = {"num_completed_folds": len(rows)}
    for prefix in ("raw", "primary", "kalman"):
        for metric in ("MAE", "RMSE", "Bias", "Corr"):
            values = np.asarray(
                [float(row[f"{prefix}_{metric}"]) for row in rows], dtype=np.float64
            )
            result[f"{prefix}_{metric}_mean"] = float(np.mean(values))
            result[f"{prefix}_{metric}_std"] = float(np.std(values, ddof=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/feature_cache/waterlevel_test_only_default.npz",
    )
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs_operation_split/test_only_8000/loocv/wl_random_forest",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    cache = args.cache.expanduser().resolve()
    frozen_path = args.frozen.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen["manifest_sha256"] != sha256_file(manifest):
        raise RuntimeError("Frozen selection and LOOCV master manifest differ")
    if frozen["feature_cache_sha256"] != sha256_file(cache):
        raise RuntimeError("Frozen selection and LOOCV feature cache differ")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_rows = list(reader)
        fields = list(reader.fieldnames or [])

    (
        features,
        anchor,
        sequences,
        masks,
        targets,
        splits,
        operations,
        times,
        records,
        provenance,
    ) = load_rf_inputs(
        manifest,
        cache,
        frozen["residual_anchor"],
        frozen.get("feature_mode", "raw_stats"),
    )
    expected_operations = sorted(set(operations.tolist()))
    if expected_operations != list(range(1, 19)):
        raise RuntimeError(f"Unexpected operation IDs: {expected_operations}")
    params = frozen["parameter_selection"]["best_parameters"]
    coefficient = float(frozen["residual_coefficient"])
    alpha = float(frozen["causal_stabilizer_selection"]["alpha"])
    kalman_params = frozen["kalman_selection"]["parameters"]

    output.mkdir(parents=True, exist_ok=True)
    completed_rows = []
    for operation in expected_operations:
        fold_dir = output / f"operation_{operation:02d}"
        metrics_path = fold_dir / "eval/per_operation_metrics.csv"
        if metrics_path.exists() and not args.force:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                completed_rows.extend(csv.DictReader(handle))
            print(f"[skip] operation={operation:02d}")
            continue
        train = operations != operation
        test = operations == operation
        model = fit_forest(
            params,
            4200 + operation,
            features[train],
            targets[train],
            anchor[train],
            coefficient,
            operations[train],
        )
        raw = predict_forest(model, features[test], anchor[test], coefficient)
        primary = causal_exponential_by_operation(
            raw, operations[test], times[test], alpha
        )
        kalman = filter_by_operation(
            times[test], primary, operations[test], kalman_params
        )
        raw_metrics = extended_metrics(targets[test], raw)
        primary_metrics = extended_metrics(targets[test], primary)
        kalman_metrics = extended_metrics(targets[test], kalman)
        date = sorted({records[index]["date"] for index in np.flatnonzero(test)})
        if len(date) != 1:
            raise RuntimeError(f"Operation {operation} spans dates {date}")
        row = {
            "operation_id": operation,
            "date": date[0],
            "count": int(test.sum()),
            **metric_columns("raw", raw_metrics),
            **metric_columns("primary", primary_metrics),
            **metric_columns("kalman", kalman_metrics),
        }
        atomic_csv(metrics_path, [row])
        prediction_rows = []
        for local, global_index in enumerate(np.flatnonzero(test)):
            prediction_rows.append(
                {
                    "sample_id": records[global_index]["sample_id"],
                    "operation_id": operation,
                    "date": date[0],
                    "timestamp": records[global_index]["timestamp"],
                    "gt": targets[test][local],
                    "pred": raw[local],
                    "pred_causal": primary[local],
                    "pred_kf": kalman[local],
                }
            )
        atomic_csv(fold_dir / "eval/predictions.csv", prediction_rows)
        fold_manifest = fold_dir / "manifest.csv"
        write_fold_manifest(source_rows, fields, operation, fold_manifest)
        split_operations = {
            "train": [value for value in expected_operations if value != operation],
            "val": [],
            "test": [operation],
        }
        split_counts = {
            "train": int(train.sum()),
            "test": int(test.sum()),
        }
        atomic_json(
            fold_dir / "eval/split_audit.json",
            {
                "manifest_path": str(fold_manifest),
                "manifest_sha256": sha256_file(fold_manifest),
                "feature_cache": {"path": str(cache), "sha256": sha256_file(cache)},
                "audit": {
                    "num_frames": len(operations),
                    "num_operations": len(expected_operations),
                    "split_counts": split_counts,
                    "split_operations": split_operations,
                    "operation_disjoint": True,
                },
            },
        )
        completed_rows.append({key: str(value) for key, value in row.items()})
        print(
            f"[done] operation={operation:02d} primary_RMSE={primary_metrics['RMSE']:.6f}",
            flush=True,
        )

    rows = []
    for operation in expected_operations:
        metrics_path = output / f"operation_{operation:02d}/eval/per_operation_metrics.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
        rows.append({"fold": f"operation_{operation:02d}", **row})
    atomic_csv(output / "loocv_per_operation.csv", rows)
    summary = summarize(rows)
    summary.update(
        {
            "primary_definition": "anchored Random Forest plus causal first-order stabilizer",
            "raw_definition": "anchored Random Forest before causal stabilization",
            "kalman_definition": "Kalman output after the causal primary output",
            "primary_RMSE_macro": summary["primary_RMSE_mean"],
            "primary_RMSE_macro_sample_sd": summary["primary_RMSE_std"],
        }
    )
    atomic_json(output / "loocv_macro_summary.json", summary)
    atomic_json(
        output / "protocol.json",
        {
            "protocol": "leave-one-operation-out",
            "base_config": str(frozen_path),
            "master_manifest": str(manifest),
            "operations": expected_operations,
            "epochs": 1,
            "num_workers": 0,
            "parallel_jobs": 1,
            "hyperparameters_fixed_before_folds": True,
            "frozen_selection_sha256": sha256_file(frozen_path),
            "residual_anchor": frozen["residual_anchor"],
            "feature_mode": frozen.get("feature_mode", "raw_stats"),
            "residual_coefficient": coefficient,
            "causal_alpha": alpha,
            "kalman_parameters": kalman_params,
            "compatibility_note": "raw_* is pre-stabilizer; primary_* is the deployed RF output",
        },
    )
    print(f"[done] RF LOOCV: {output}")


if __name__ == "__main__":
    main()
