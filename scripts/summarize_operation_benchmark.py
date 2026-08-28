#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Combine completed fixed-split operation-wise model results into CSV tables."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def flatten_summary(model_name, summary):
    row = {
        "model": model_name,
        "test_operations": ",".join(
            str(value) for value in summary.get("test_operations", [])
        ),
    }
    for section in ("micro_raw", "micro_kalman"):
        metrics = summary.get(section) or {}
        for key in ("count", "MAE", "RMSE", "Bias", "Corr"):
            row[f"{section}_{key}"] = metrics.get(key)
    for key, value in (summary.get("macro_across_operations") or {}).items():
        row[f"macro_{key}"] = value
    return row


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_operation_split"))
    parser.add_argument("--expected-models", type=int, default=13)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional protocol manifest. When supplied, expected split counts, "
            "operations, dates, and the manifest hash are derived from this file "
            "instead of using the legacy 16,000-frame defaults."
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    overall_rows = []
    operation_rows = []
    manifest_hashes = set()
    cache_hashes = set()
    audited_models = []
    expected_split_operations = {
        "train": list(range(1, 11)),
        "val": list(range(11, 14)),
        "test": list(range(14, 19)),
    }
    expected_split_counts = {"train": 9690, "val": 1954, "test": 4356}
    expected_manifest_hash = None
    dates_by_split = {}
    manifest_path = None
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        if not manifest_rows:
            raise RuntimeError(f"Manifest is empty: {manifest_path}")
        split_order = ("train", "val", "test")
        expected_split_counts = {
            split: sum(row["split"] == split for row in manifest_rows)
            for split in split_order
        }
        expected_split_operations = {
            split: sorted(
                {int(row["operation_id"]) for row in manifest_rows if row["split"] == split}
            )
            for split in split_order
        }
        dates_by_split = {
            split: sorted({row["date"] for row in manifest_rows if row["split"] == split})
            for split in split_order
        }
        expected_manifest_hash = sha256_file(manifest_path)
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = model_dir / "eval" / "operation_summary.json"
        metrics_path = model_dir / "eval" / "per_operation_metrics.csv"
        audit_path = model_dir / "eval" / "split_audit.json"
        if not summary_path.is_file() or not metrics_path.is_file():
            continue
        if not audit_path.is_file():
            raise FileNotFoundError(f"Missing split audit: {audit_path}")
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with audit_path.open("r", encoding="utf-8") as handle:
            split_audit = json.load(handle)
        audit_body = split_audit.get("audit") or {}
        if not bool(audit_body.get("operation_disjoint")):
            raise RuntimeError(f"{model_dir.name}: operation-disjoint audit failed")
        observed_split_operations = {
            key: [int(value) for value in values]
            for key, values in (audit_body.get("split_operations") or {}).items()
        }
        if observed_split_operations != expected_split_operations:
            raise RuntimeError(
                f"{model_dir.name}: unexpected split operations: "
                f"{observed_split_operations}"
            )
        observed_split_counts = {
            key: int(value)
            for key, value in (audit_body.get("split_counts") or {}).items()
        }
        if observed_split_counts != expected_split_counts:
            raise RuntimeError(
                f"{model_dir.name}: unexpected split counts: {observed_split_counts}"
            )
        manifest_hashes.add(str(split_audit.get("manifest_sha256")))
        cache_hash = str((split_audit.get("feature_cache") or {}).get("sha256"))
        if cache_hash and cache_hash != "None":
            cache_hashes.add(cache_hash)
        overall_rows.append(flatten_summary(model_dir.name, summary))
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            model_operation_rows = list(csv.DictReader(handle))
            observed_test_operations = sorted(
                int(row["operation_id"]) for row in model_operation_rows
            )
            if observed_test_operations != expected_split_operations["test"]:
                raise RuntimeError(
                    f"{model_dir.name}: per-operation table contains "
                    f"{observed_test_operations}"
                )
            for row in model_operation_rows:
                operation_rows.append({"model": model_dir.name, **row})
        audited_models.append(model_dir.name)

    if not overall_rows:
        print(f"[info] No completed fixed-split results found under {root}")
        return
    if len(overall_rows) != int(args.expected_models):
        raise RuntimeError(
            f"Expected {args.expected_models} completed models, found "
            f"{len(overall_rows)}: {audited_models}"
        )
    if len(manifest_hashes) != 1:
        raise RuntimeError(f"Models use different manifests: {sorted(manifest_hashes)}")
    if expected_manifest_hash is not None and manifest_hashes != {expected_manifest_hash}:
        raise RuntimeError(
            "Model results do not use the requested manifest: "
            f"expected={expected_manifest_hash}, observed={sorted(manifest_hashes)}"
        )
    if len(cache_hashes) != 1:
        raise RuntimeError(f"Models use different feature caches: {sorted(cache_hashes)}")
    overall_path = root / "fixed_split_overall_all_models.csv"
    operation_path = root / "fixed_split_per_operation_all_models.csv"
    write_csv(overall_path, overall_rows)
    write_csv(operation_path, operation_rows)
    audit_report = {
        "protocol": "fixed-operation-and-held-out-day",
        "num_models": len(overall_rows),
        "models": audited_models,
        "num_operations": 18,
        "num_frames": sum(expected_split_counts.values()),
        "split_operations": expected_split_operations,
        "split_counts": expected_split_counts,
        "all_operation_disjoint": True,
        "manifest_sha256": next(iter(manifest_hashes)),
        "feature_cache_sha256": next(iter(cache_hashes)),
        "dates_by_split": dates_by_split,
        "test_day": (
            dates_by_split.get("test", [None])[0]
            if len(dates_by_split.get("test", [])) == 1
            else None
        ),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "per_operation_rows": len(operation_rows),
        "outputs": {
            "overall_csv": str(overall_path),
            "overall_csv_sha256": sha256_file(overall_path),
            "per_operation_csv": str(operation_path),
            "per_operation_csv_sha256": sha256_file(operation_path),
        },
    }
    audit_path = root / "fixed_split_audit.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[done] overall table: {overall_path}")
    print(f"[done] per-operation table: {operation_path}")
    print(f"[done] audit: {audit_path}")


if __name__ == "__main__":
    main()
