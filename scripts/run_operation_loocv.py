#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare or run leave-one-operation-out (LOOCV) water-level evaluation.

Hyperparameters must be fixed before this script is run.  For fold k, all
frames from operation k are assigned to test and all frames from the other 17
operations are assigned to train.  No validation split is used in a fold, so
``train.epochs`` is fixed (or overridden with ``--epochs``).

Examples:
    # Generate 18 auditable fold manifests/configs only.
    python scripts/run_operation_loocv.py --cfg configs/wl_hgrn.yaml

    # Execute selected folds after checking the generated protocol.
    python scripts/run_operation_loocv.py --cfg configs/wl_hgrn.yaml \
        --operations 1-3 --run
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_CANDIDATES = [
    "wl_transformer",
    "wl_retnet",
    "wl_mamba",
    "wl_rwkv",
    "wl_hyena",
    "wl_mega",
    "wl_hgrn",
    "wl_linear_regression",
    "wl_ridge_regression",
    "wl_mlp",
    "wl_1dcnn",
    "wl_svr",
    "wl_random_forest",
    "wl_xgboost",
]


def parse_operation_spec(spec: str) -> List[int]:
    values = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending operation range: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return sorted(values)


def detect_section(cfg: dict) -> str:
    for key in SECTION_CANDIDATES:
        if isinstance(cfg.get(key), dict):
            return key
    raise RuntimeError("Cannot detect the active wl_* section")


def load_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "operation_id" not in fields or "split" not in fields:
        raise RuntimeError(
            f"Manifest {path} must contain operation_id and split columns"
        )
    operations = sorted({int(row["operation_id"]) for row in rows})
    if not rows or not operations:
        raise RuntimeError(f"Manifest {path} contains no usable rows")
    return fields, rows, operations


def write_fold_manifest(
    fields: List[str], rows: List[Dict[str, str]], operation_id: int, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for original in rows:
            row = dict(original)
            row["split"] = (
                "test" if int(row["operation_id"]) == operation_id else "train"
            )
            writer.writerow(row)


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def prepare_fold_config(
    base_cfg: dict,
    section_key: str,
    fold_dir: Path,
    manifest_path: Path,
    epochs: int,
    num_workers: int,
) -> Path:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("data", {})["operation_manifest_path"] = str(
        manifest_path.resolve()
    )
    cfg["data"]["train_split"] = "train"
    cfg["data"]["val_split"] = "val"
    cfg["data"]["test_split"] = "test"

    cfg.setdefault("output", {})["root"] = str(fold_dir.resolve())
    section = cfg[section_key]
    section["out_dir"] = "model"
    section.setdefault("train", {})["epochs"] = int(epochs)
    section["train"]["num_workers"] = int(num_workers)
    section["train"]["val_num_workers"] = int(num_workers)

    eval_cfg = cfg.setdefault("eval", {})
    # train_wl.py saves a compatibility model.pth for both torch and sklearn.
    eval_cfg["model_path"] = str((fold_dir / "model" / "model.pth").resolve())
    eval_cfg["out_dir"] = str((fold_dir / "eval").resolve())
    eval_cfg["num_workers"] = int(num_workers)

    config_path = fold_dir / "config.yaml"
    save_yaml(cfg, config_path)
    return config_path


def run_fold(config_path: Path, fold_dir: Path) -> None:
    log_path = fold_dir / "run.log"
    commands = [
        [sys.executable, str(REPO_ROOT / "scripts" / "train_wl.py"), "--cfg", str(config_path)],
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_wl.py"), "--cfg", str(config_path)],
    ]
    with log_path.open("w", encoding="utf-8") as log_handle:
        for command in commands:
            print("[cmd] " + " ".join(command), flush=True)
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Command failed with exit code {result.returncode}; see {log_path}"
                )


def aggregate_completed_folds(output_root: Path) -> None:
    rows = []
    for metrics_path in sorted(output_root.glob("operation_*/eval/per_operation_metrics.csv")):
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            fold_rows = list(csv.DictReader(handle))
        if len(fold_rows) != 1:
            raise RuntimeError(
                f"Expected exactly one held-out operation in {metrics_path}, "
                f"found {len(fold_rows)}"
            )
        row = dict(fold_rows[0])
        row["fold"] = metrics_path.parents[1].name
        rows.append(row)

    if not rows:
        return
    fieldnames = ["fold"] + [key for key in rows[0] if key != "fold"]
    summary_csv = output_root / "loocv_per_operation.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {"num_completed_folds": len(rows)}
    for key in (
        "raw_MAE", "raw_RMSE", "raw_Bias", "raw_Corr",
        "kalman_MAE", "kalman_RMSE", "kalman_Bias", "kalman_Corr",
    ):
        values = []
        for row in rows:
            value = str(row.get(key, "")).strip()
            if value and value.lower() not in {"none", "nan"}:
                values.append(float(value))
        if values:
            aggregate[f"{key}_mean"] = statistics.mean(values)
            aggregate[f"{key}_std"] = statistics.stdev(values) if len(values) >= 2 else None
    summary_json = output_root / "loocv_macro_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[done] LOOCV table: {summary_csv}")
    print(f"[done] LOOCV macro: {summary_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            REPO_ROOT
            / "splits"
            / "waterlevel_test_only_8000"
            / "manifest.csv"
        ),
    )
    parser.add_argument("--operations", default="1-18")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers per fold. Zero is fastest for the in-memory "
            "validated feature cache and avoids restarting workers each epoch."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT / "outputs_operation_split" / "test_only_8000" / "loocv"
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually train/evaluate. Without this flag only fold files are prepared.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of independent folds to execute concurrently.",
    )
    args = parser.parse_args()

    cfg_path = args.cfg.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)
    section_key = detect_section(base_cfg)
    default_epochs = int(base_cfg[section_key].get("train", {}).get("epochs", 40))
    epochs = int(args.epochs) if args.epochs is not None else default_epochs

    fields, rows, observed_operations = load_manifest(manifest_path)
    requested_operations = parse_operation_spec(args.operations)
    unknown = sorted(set(requested_operations) - set(observed_operations))
    if unknown:
        raise ValueError(f"Requested operations are absent from manifest: {unknown}")

    output_root = args.output_root.expanduser().resolve() / cfg_path.stem
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = output_root / "protocol.json"
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": "leave-one-operation-out",
                "base_config": str(cfg_path),
                "master_manifest": str(manifest_path),
                "operations": requested_operations,
                "epochs": epochs,
                "num_workers": int(args.num_workers),
                "parallel_jobs": int(args.jobs),
                "hyperparameters_fixed_before_folds": True,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    pending_folds = []
    for operation_id in requested_operations:
        fold_dir = output_root / f"operation_{operation_id:02d}"
        fold_manifest = fold_dir / "manifest.csv"
        write_fold_manifest(fields, rows, operation_id, fold_manifest)
        fold_config = prepare_fold_config(
            base_cfg,
            section_key,
            fold_dir,
            fold_manifest,
            epochs,
            int(args.num_workers),
        )
        print(
            f"[prepared] operation={operation_id:02d} "
            f"manifest={fold_manifest} config={fold_config}"
        )
        if args.run:
            result_path = fold_dir / "eval" / "per_operation_metrics.csv"
            if args.skip_existing and result_path.is_file():
                print(f"[skip] completed fold: {operation_id:02d}")
                continue
            pending_folds.append((operation_id, fold_config, fold_dir))

    if args.run and pending_folds:
        jobs = max(1, int(args.jobs))
        if jobs == 1:
            for operation_id, fold_config, fold_dir in pending_folds:
                print(f"[fold-start] operation={operation_id:02d}", flush=True)
                run_fold(fold_config, fold_dir)
                print(f"[fold-done] operation={operation_id:02d}", flush=True)
        else:
            failures = []
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_to_operation = {
                    executor.submit(run_fold, fold_config, fold_dir): operation_id
                    for operation_id, fold_config, fold_dir in pending_folds
                }
                for operation_id in sorted(future_to_operation.values()):
                    print(f"[fold-queued] operation={operation_id:02d}", flush=True)
                for future in as_completed(future_to_operation):
                    operation_id = future_to_operation[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append((operation_id, str(exc)))
                        print(
                            f"[fold-failed] operation={operation_id:02d}: {exc}",
                            flush=True,
                        )
                    else:
                        print(f"[fold-done] operation={operation_id:02d}", flush=True)
            if failures:
                raise RuntimeError(f"LOOCV fold failures: {failures}")

    aggregate_completed_folds(output_root)
    if not args.run:
        print("[done] Fold files prepared. Re-run with --run to train/evaluate.")


if __name__ == "__main__":
    main()
