#!/usr/bin/env python3
"""Compose manuscript tables after replacing only the Random Forest result.

The original 13-model outputs remain immutable.  This reporting helper copies
the other 12 models exactly, maps their direct frame-wise values to the
``primary`` columns, and replaces only ``wl_random_forest`` with the current
RF causal inference output.  It also writes a source-hash audit so the overlay
cannot be mistaken for a new run of the other estimators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = REPO_ROOT / "outputs_operation_split/test_only_8000/fixed"
RF_ROOT = REPO_ROOT / "outputs_operation_split/test_only_8000/rf_tuned_robust_anchor"
RF_MODEL = "wl_random_forest"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--rf-root", type=Path, default=RF_ROOT)
    args = parser.parse_args()
    original_root = args.original_root.expanduser().resolve()
    rf_root = args.rf_root.expanduser().resolve()
    original_overall_path = original_root / "fixed_split_overall_all_models.csv"
    original_per_operation_path = (
        original_root / "fixed_split_per_operation_all_models.csv"
    )
    rf_summary_path = rf_root / "test_summary.json"
    rf_per_operation_path = rf_root / "test_per_operation_metrics.csv"

    original_overall = read_csv(original_overall_path)
    original_per_operation = read_csv(original_per_operation_path)
    rf_summary = read_json(rf_summary_path)
    rf_per_operation = read_csv(rf_per_operation_path)

    models = [row["model"] for row in original_overall]
    if len(models) != 13 or len(set(models)) != 13 or models.count(RF_MODEL) != 1:
        raise RuntimeError("Expected 13 unique models with one Random Forest row")
    if len(original_per_operation) != 13 * len(rf_per_operation):
        raise RuntimeError(
            "Expected the same per-operation coverage for all 13 models"
        )

    original_rf_by_operation = {
        int(row["operation_id"]): row
        for row in original_per_operation
        if row["model"] == RF_MODEL
    }
    current_rf_by_operation = {
        int(row["operation_id"]): row for row in rf_per_operation
    }
    expected_operations = sorted(current_rf_by_operation)
    if sorted(original_rf_by_operation) != expected_operations:
        raise RuntimeError("Original RF operation set does not match 14--18")
    if sorted(current_rf_by_operation) != expected_operations:
        raise RuntimeError("Current RF operation set does not match 14--18")

    per_operation_rows: list[dict] = []
    for source in original_per_operation:
        model = source["model"]
        operation = int(source["operation_id"])
        if model == RF_MODEL:
            current = current_rf_by_operation[operation]
            if int(current["count"]) != int(source["count"]):
                raise RuntimeError(f"RF frame count changed for operation {operation}")
            row = {
                "model": model,
                "operation_id": operation,
                "date": source["date"],
                "count": int(current["count"]),
                "primary_MAE": current["causal_MAE"],
                "primary_RMSE": current["causal_RMSE"],
                "primary_Bias": current["causal_Bias"],
                "primary_Corr": current["causal_Corr"],
                "kalman_MAE": current["kalman_MAE"],
                "kalman_RMSE": current["kalman_RMSE"],
                "kalman_Bias": current["kalman_Bias"],
                "kalman_Corr": current["kalman_Corr"],
                "primary_definition": "causal_first_order_stabilized",
                "source_artifact": str(rf_per_operation_path.relative_to(REPO_ROOT)),
            }
        else:
            row = {
                "model": model,
                "operation_id": operation,
                "date": source["date"],
                "count": int(source["count"]),
                "primary_MAE": source["raw_MAE"],
                "primary_RMSE": source["raw_RMSE"],
                "primary_Bias": source["raw_Bias"],
                "primary_Corr": source["raw_Corr"],
                "kalman_MAE": source["kalman_MAE"],
                "kalman_RMSE": source["kalman_RMSE"],
                "kalman_Bias": source["kalman_Bias"],
                "kalman_Corr": source["kalman_Corr"],
                "primary_definition": "direct_frame_wise",
                "source_artifact": str(
                    original_per_operation_path.relative_to(REPO_ROOT)
                ),
            }
        per_operation_rows.append(row)

    rf_primary = rf_summary["causal_stabilized"]
    rf_kalman = rf_summary["kalman_after_causal"]
    rf_operation_rows = [
        row for row in per_operation_rows if row["model"] == RF_MODEL
    ]

    overall_rows: list[dict] = []
    for source in original_overall:
        model = source["model"]
        if model == RF_MODEL:
            primary_mae_values = [
                float(row["primary_MAE"]) for row in rf_operation_rows
            ]
            primary_rmse_values = [
                float(row["primary_RMSE"]) for row in rf_operation_rows
            ]
            primary_bias_values = [
                float(row["primary_Bias"]) for row in rf_operation_rows
            ]
            primary_corr_values = [
                float(row["primary_Corr"]) for row in rf_operation_rows
            ]
            kalman_mae_values = [float(row["kalman_MAE"]) for row in rf_operation_rows]
            kalman_rmse_values = [
                float(row["kalman_RMSE"]) for row in rf_operation_rows
            ]
            kalman_bias_values = [
                float(row["kalman_Bias"]) for row in rf_operation_rows
            ]
            kalman_corr_values = [
                float(row["kalman_Corr"]) for row in rf_operation_rows
            ]
            row = {
                "model": model,
                "test_operations": source["test_operations"],
                "micro_count": int(rf_primary["count"]),
                "micro_primary_MAE": rf_primary["MAE"],
                "micro_primary_RMSE": rf_primary["RMSE"],
                "micro_primary_Bias": rf_primary["Bias"],
                "micro_primary_Corr": rf_primary["Corr"],
                "micro_kalman_MAE": rf_kalman["MAE"],
                "micro_kalman_RMSE": rf_kalman["RMSE"],
                "micro_kalman_Bias": rf_kalman["Bias"],
                "micro_kalman_Corr": rf_kalman["Corr"],
                "macro_num_operations": len(expected_operations),
                "macro_primary_MAE_mean": sample_mean(primary_mae_values),
                "macro_primary_MAE_std": sample_std(primary_mae_values),
                "macro_primary_RMSE_mean": sample_mean(primary_rmse_values),
                "macro_primary_RMSE_std": sample_std(primary_rmse_values),
                "macro_primary_Bias_mean": sample_mean(primary_bias_values),
                "macro_primary_Bias_std": sample_std(primary_bias_values),
                "macro_primary_Corr_mean": sample_mean(primary_corr_values),
                "macro_primary_Corr_std": sample_std(primary_corr_values),
                "macro_kalman_MAE_mean": sample_mean(kalman_mae_values),
                "macro_kalman_MAE_std": sample_std(kalman_mae_values),
                "macro_kalman_RMSE_mean": sample_mean(kalman_rmse_values),
                "macro_kalman_RMSE_std": sample_std(kalman_rmse_values),
                "macro_kalman_Bias_mean": sample_mean(kalman_bias_values),
                "macro_kalman_Bias_std": sample_std(kalman_bias_values),
                "macro_kalman_Corr_mean": sample_mean(kalman_corr_values),
                "macro_kalman_Corr_std": sample_std(kalman_corr_values),
                "primary_definition": "causal_first_order_stabilized",
                "source_artifact": str(rf_summary_path.relative_to(REPO_ROOT)),
            }
        else:
            row = {
                "model": model,
                "test_operations": source["test_operations"],
                "micro_count": source["micro_raw_count"],
                "micro_primary_MAE": source["micro_raw_MAE"],
                "micro_primary_RMSE": source["micro_raw_RMSE"],
                "micro_primary_Bias": source["micro_raw_Bias"],
                "micro_primary_Corr": source["micro_raw_Corr"],
                "micro_kalman_MAE": source["micro_kalman_MAE"],
                "micro_kalman_RMSE": source["micro_kalman_RMSE"],
                "micro_kalman_Bias": source["micro_kalman_Bias"],
                "micro_kalman_Corr": source["micro_kalman_Corr"],
                "macro_num_operations": source["macro_num_operations"],
                "macro_primary_MAE_mean": source["macro_raw_MAE_mean"],
                "macro_primary_MAE_std": source["macro_raw_MAE_std"],
                "macro_primary_RMSE_mean": source["macro_raw_RMSE_mean"],
                "macro_primary_RMSE_std": source["macro_raw_RMSE_std"],
                "macro_primary_Bias_mean": source["macro_raw_Bias_mean"],
                "macro_primary_Bias_std": source["macro_raw_Bias_std"],
                "macro_primary_Corr_mean": source["macro_raw_Corr_mean"],
                "macro_primary_Corr_std": source["macro_raw_Corr_std"],
                "macro_kalman_MAE_mean": source["macro_kalman_MAE_mean"],
                "macro_kalman_MAE_std": source["macro_kalman_MAE_std"],
                "macro_kalman_RMSE_mean": source["macro_kalman_RMSE_mean"],
                "macro_kalman_RMSE_std": source["macro_kalman_RMSE_std"],
                "macro_kalman_Bias_mean": source["macro_kalman_Bias_mean"],
                "macro_kalman_Bias_std": source["macro_kalman_Bias_std"],
                "macro_kalman_Corr_mean": source["macro_kalman_Corr_mean"],
                "macro_kalman_Corr_std": source["macro_kalman_Corr_std"],
                "primary_definition": "direct_frame_wise",
                "source_artifact": str(original_overall_path.relative_to(REPO_ROOT)),
            }
        overall_rows.append(row)

    overall_output = rf_root / "manuscript_fixed_split_all_models_overall.csv"
    per_operation_output = (
        rf_root / "manuscript_fixed_split_all_models_per_operation.csv"
    )
    write_csv(overall_output, overall_rows, list(overall_rows[0]))
    write_csv(
        per_operation_output,
        per_operation_rows,
        list(per_operation_rows[0]),
    )

    audit_output = rf_root / "manuscript_fixed_split_reporting_audit.json"
    write_json(
        audit_output,
        {
            "protocol": "13-model-reporting-overlay-replacing-random-forest-only",
            "rf_replaced_only": True,
            "unchanged_model_count": 12,
            "num_models": len(overall_rows),
            "num_model_operation_rows": len(per_operation_rows),
            "test_operations": expected_operations,
            "total_test_frames": int(rf_primary["count"]),
            "manifest_sha256": rf_summary["manifest_sha256"],
            "feature_cache_sha256": rf_summary["feature_cache_sha256"],
            "sources": {
                str(original_overall_path.relative_to(REPO_ROOT)): sha256_file(
                    original_overall_path
                ),
                str(original_per_operation_path.relative_to(REPO_ROOT)): sha256_file(
                    original_per_operation_path
                ),
                str(rf_summary_path.relative_to(REPO_ROOT)): sha256_file(
                    rf_summary_path
                ),
                str(rf_per_operation_path.relative_to(REPO_ROOT)): sha256_file(
                    rf_per_operation_path
                ),
            },
            "outputs": {
                str(overall_output.relative_to(REPO_ROOT)): sha256_file(overall_output),
                str(per_operation_output.relative_to(REPO_ROOT)): sha256_file(
                    per_operation_output
                ),
            },
        },
    )
    print(overall_output)
    print(per_operation_output)
    print(audit_output)


if __name__ == "__main__":
    main()
