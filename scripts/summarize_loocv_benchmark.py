#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit and summarize a multi-model operation-wise LOOCV benchmark."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "wl_linear_regression",
    "wl_mlp",
    "wl_random_forest",
    "wl_hgrn",
]
METRIC_KEYS = [
    "raw_MAE",
    "raw_RMSE",
    "raw_Bias",
    "raw_Corr",
    "primary_MAE",
    "primary_RMSE",
    "primary_Bias",
    "primary_Corr",
    "kalman_MAE",
    "kalman_RMSE",
    "kalman_Bias",
    "kalman_Corr",
]


def metric_value(mapping, key):
    """Use direct frame-wise output as primary unless a model defines one."""
    value = mapping.get(key)
    if value not in (None, ""):
        return value
    if key.startswith("primary_"):
        return mapping[key.replace("primary_", "raw_", 1)]
    raise KeyError(key)


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            REPO_ROOT / "outputs_operation_split" / "test_only_8000" / "loocv"
        ),
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--expected-operations", type=int, default=18)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    expected_operations = list(range(1, int(args.expected_operations) + 1))
    macro_rows = []
    per_operation_rows = []
    counts_by_operation = {}
    manifest_hashes_by_operation = {}
    cache_hashes = set()
    master_manifests = set()
    audit_failures = []
    model_protocols = {}
    base_config_hashes = {}

    for model_name in args.models:
        model_root = root / model_name
        protocol_path = model_root / "protocol.json"
        summary_path = model_root / "loocv_macro_summary.json"
        table_path = model_root / "loocv_per_operation.csv"
        for required in (protocol_path, summary_path, table_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing required LOOCV artifact: {required}")

        protocol = read_json(protocol_path)
        summary = read_json(summary_path)
        rows = read_csv(table_path)
        model_protocols[model_name] = protocol
        base_config_path = Path(protocol["base_config"]).expanduser().resolve()
        base_config_hashes[model_name] = sha256_file(base_config_path)
        master_manifests.add(str(Path(protocol["master_manifest"]).resolve()))

        observed_operations = sorted(int(row["operation_id"]) for row in rows)
        if observed_operations != expected_operations:
            raise RuntimeError(
                f"{model_name}: expected operations {expected_operations}, "
                f"found {observed_operations}"
            )
        if int(summary.get("num_completed_folds", -1)) != len(expected_operations):
            raise RuntimeError(
                f"{model_name}: macro summary reports "
                f"{summary.get('num_completed_folds')} completed folds"
            )

        macro_row = {
            "model": model_name,
            "epochs": int(protocol["epochs"]),
            "num_folds": len(rows),
        }
        for key in METRIC_KEYS:
            macro_row[f"{key}_mean"] = metric_value(summary, f"{key}_mean")
            macro_row[f"{key}_std"] = metric_value(summary, f"{key}_std")
        macro_rows.append(macro_row)

        for row in rows:
            operation_id = int(row["operation_id"])
            expected_fold = f"operation_{operation_id:02d}"
            if row["fold"] != expected_fold:
                raise RuntimeError(
                    f"{model_name}: fold {row['fold']} does not match "
                    f"operation {operation_id}"
                )
            count = int(row["count"])
            previous_count = counts_by_operation.setdefault(operation_id, count)
            if previous_count != count:
                raise RuntimeError(
                    f"Operation {operation_id} count differs across models: "
                    f"{previous_count} vs {count}"
                )

            audit_path = model_root / expected_fold / "eval" / "split_audit.json"
            audit = read_json(audit_path)
            audit_body = audit["audit"]
            test_operations = [
                int(value)
                for value in audit_body["split_operations"].get("test", [])
            ]
            train_operations = {
                int(value)
                for value in audit_body["split_operations"].get("train", [])
            }
            operation_disjoint = bool(audit_body.get("operation_disjoint"))
            if (
                not operation_disjoint
                or test_operations != [operation_id]
                or operation_id in train_operations
            ):
                audit_failures.append(
                    {
                        "model": model_name,
                        "operation_id": operation_id,
                        "audit_path": str(audit_path),
                    }
                )
            manifest_hashes_by_operation.setdefault(operation_id, set()).add(
                str(audit["manifest_sha256"])
            )
            cache_info = audit.get("feature_cache") or {}
            if cache_info.get("sha256"):
                cache_hashes.add(str(cache_info["sha256"]))

            combined = {
                "model": model_name,
                "epochs": int(protocol["epochs"]),
                "fold": row["fold"],
                "operation_id": operation_id,
                "date": row["date"],
                "count": count,
            }
            for key in METRIC_KEYS:
                combined[key] = metric_value(row, key)
            per_operation_rows.append(combined)

    if audit_failures:
        raise RuntimeError(f"Operation-disjoint audit failures: {audit_failures}")
    inconsistent_manifest_hashes = {
        operation_id: sorted(hashes)
        for operation_id, hashes in manifest_hashes_by_operation.items()
        if len(hashes) != 1
    }
    if inconsistent_manifest_hashes:
        raise RuntimeError(
            "Fold manifests differ across models: "
            f"{inconsistent_manifest_hashes}"
        )
    if len(master_manifests) != 1:
        raise RuntimeError(
            f"Models use different master manifests: {sorted(master_manifests)}"
        )
    if len(cache_hashes) != 1:
        raise RuntimeError(
            f"Models use different feature caches: {sorted(cache_hashes)}"
        )

    macro_fields = ["model", "epochs", "num_folds"]
    for key in METRIC_KEYS:
        macro_fields.extend([f"{key}_mean", f"{key}_std"])
    macro_path = root / "loocv_four_model_macro.csv"
    write_csv(macro_path, macro_fields, macro_rows)

    per_operation_fields = [
        "model",
        "epochs",
        "fold",
        "operation_id",
        "date",
        "count",
    ] + METRIC_KEYS
    per_operation_path = root / "loocv_four_model_per_operation.csv"
    write_csv(per_operation_path, per_operation_fields, per_operation_rows)

    rows_by_model_operation = {
        (row["model"], int(row["operation_id"])): row
        for row in per_operation_rows
    }
    rmse_wide_fields = ["operation_id", "date", "count"] + [
        f"{model_name}_kalman_RMSE" for model_name in args.models
    ]
    rmse_wide_rows = []
    for operation_id in expected_operations:
        first = rows_by_model_operation[(args.models[0], operation_id)]
        wide_row = {
            "operation_id": operation_id,
            "date": first["date"],
            "count": first["count"],
        }
        for model_name in args.models:
            wide_row[f"{model_name}_kalman_RMSE"] = rows_by_model_operation[
                (model_name, operation_id)
            ]["kalman_RMSE"]
        rmse_wide_rows.append(wide_row)
    rmse_wide_path = root / "loocv_four_model_kf_rmse_by_operation.csv"
    write_csv(rmse_wide_path, rmse_wide_fields, rmse_wide_rows)

    master_manifest_path = Path(next(iter(master_manifests)))
    audit_report = {
        "protocol": "four-model-leave-one-operation-out",
        "models": list(args.models),
        "model_epochs": {
            name: int(model_protocols[name]["epochs"]) for name in args.models
        },
        "expected_operations": expected_operations,
        "num_models": len(args.models),
        "num_folds_per_model": len(expected_operations),
        "total_evaluated_folds": len(per_operation_rows),
        "total_test_frames_per_model": sum(counts_by_operation.values()),
        "all_operation_disjoint": True,
        "master_manifest": str(master_manifest_path),
        "master_manifest_sha256": sha256_file(master_manifest_path),
        "base_config_sha256": base_config_hashes,
        "feature_cache_sha256": next(iter(cache_hashes)),
        "fold_manifest_sha256": {
            str(operation_id): next(iter(manifest_hashes_by_operation[operation_id]))
            for operation_id in expected_operations
        },
        "outputs": {
            "macro_csv": str(macro_path),
            "macro_csv_sha256": sha256_file(macro_path),
            "per_operation_csv": str(per_operation_path),
            "per_operation_csv_sha256": sha256_file(per_operation_path),
            "kf_rmse_by_operation_csv": str(rmse_wide_path),
            "kf_rmse_by_operation_csv_sha256": sha256_file(rmse_wide_path),
        },
    }
    audit_path = root / "loocv_four_model_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[done] macro table: {macro_path}")
    print(f"[done] per-operation table: {per_operation_path}")
    print(f"[done] KF RMSE wide table: {rmse_wide_path}")
    print(f"[done] audit: {audit_path}")


if __name__ == "__main__":
    main()
