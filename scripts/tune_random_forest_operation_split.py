#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage-safe Random-Forest tuning for the fixed operation/day split.

The workflow is intentionally split into two commands:

1. ``tune`` only uses operations 1--10 for fitting and operations 11--13 for
   hyperparameter/Kalman selection.  It freezes a model, selected settings,
   and their hashes without evaluating operations 14--18.
2. ``refit`` uses the already selected settings to fit the final model on
   operations 1--13, then replaces and re-hashes the frozen model.  It neither
   selects settings nor evaluates operations 14--18.
3. ``evaluate`` verifies the frozen hashes and evaluates operations 14--18
   once.  No selection is performed in this stage.

All generated artifacts stay under
``outputs_operation_split/test_only_8000/rf_tuning`` by default.  The
deterministic feature cache contains no labels; labels and split
roles are read from the audited operation manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data import load_operation_manifest, validate_operation_manifest  # noqa: E402
from utils.eval_utils import kf_online  # noqa: E402
from utils.metrics import eval_metrics  # noqa: E402


DEFAULT_MANIFEST = (
    REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv"
)
DEFAULT_CACHE = (
    REPO_ROOT
    / "outputs_operation_split/test_only_8000/feature_cache/waterlevel_test_only_default.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs_operation_split/test_only_8000/rf_tuning"
EXPECTED_SPLITS = {
    "train": list(range(1, 11)),
    "val": [11, 12, 13],
    "test": [14, 15, 16, 17, 18],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_arrays(
    manifest_path: Path,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict], dict]:
    records = load_operation_manifest(str(manifest_path), validate_paths=False)
    audit = validate_operation_manifest(records)
    if not audit.get("operation_disjoint"):
        raise RuntimeError("Manifest audit did not confirm operation disjointness")
    observed = {
        split: audit["split_operations"].get(split, [])
        for split in ("train", "val", "test")
    }
    if observed != EXPECTED_SPLITS:
        raise RuntimeError(
            f"Unexpected fixed protocol: observed={observed}, expected={EXPECTED_SPLITS}"
        )

    with np.load(str(cache_path), allow_pickle=False) as cache:
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        sequences = np.asarray(cache["sequences"], dtype=np.float32)
        masks = np.asarray(cache["masks"], dtype=bool)
        cache_metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))

    manifest_sha = sha256_file(manifest_path)
    if cache_metadata.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Feature cache and operation manifest hashes differ")
    if sequences.shape[:2] != masks.shape or sequences.shape[0] != len(sample_ids):
        raise RuntimeError(
            f"Invalid feature-cache shapes: sequences={sequences.shape}, masks={masks.shape}"
        )

    record_by_id = {str(record["sample_id"]): record for record in records}
    if set(sample_ids.tolist()) != set(record_by_id):
        raise RuntimeError("Feature-cache sample IDs differ from manifest sample IDs")
    ordered_records = [record_by_id[sample_id] for sample_id in sample_ids]

    # This is exactly the baseline preprocessing before flattening: invalid
    # bins are zeroed.  The legacy wrapper pads 103 bins to 200 bins with
    # trailing zeros; omitting those constant columns is prediction-equivalent.
    features = sequences.copy()
    features[masks] = 0.0
    features = features.reshape(features.shape[0], -1)
    targets = np.asarray(
        [float(record["water_level"]) for record in ordered_records],
        dtype=np.float64,
    )
    splits = np.asarray([str(record["split"]) for record in ordered_records])
    operations = np.asarray(
        [int(record["operation_id"]) for record in ordered_records], dtype=np.int64
    )
    times = np.asarray([float(record["tsec"]) for record in ordered_records])
    return (
        features,
        targets,
        splits,
        operations,
        times,
        ordered_records,
        {"manifest_audit": audit, "cache_metadata": cache_metadata},
    )


def macro_rmse(y_true: np.ndarray, y_pred: np.ndarray, operations: np.ndarray) -> float:
    values = []
    for operation_id in sorted(set(operations.tolist())):
        selected = operations == operation_id
        values.append(float(np.sqrt(np.mean((y_pred[selected] - y_true[selected]) ** 2))))
    return float(np.mean(values))


def operation_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_kf: np.ndarray,
    operations: np.ndarray,
) -> List[dict]:
    rows = []
    for operation_id in sorted(set(operations.tolist())):
        selected = operations == operation_id
        raw = eval_metrics(y_true[selected], y_pred[selected])
        filtered = eval_metrics(y_true[selected], y_kf[selected])
        rows.append(
            {
                "operation_id": int(operation_id),
                "count": int(selected.sum()),
                **{f"raw_{key}": raw[key] for key in ("MAE", "RMSE", "Bias", "Corr")},
                **{
                    f"kalman_{key}": filtered[key]
                    for key in ("MAE", "RMSE", "Bias", "Corr")
                },
            }
        )
    return rows


def operation_balanced_weights(operations: np.ndarray) -> np.ndarray:
    counts = Counter(int(value) for value in operations.tolist())
    num_operations = len(counts)
    total = len(operations)
    return np.asarray(
        [total / (num_operations * counts[int(value)]) for value in operations],
        dtype=np.float64,
    )


def canonical_params(params: dict) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def candidate_parameters(num_random: int, seed: int) -> List[dict]:
    anchors = [
        {
            "n_estimators": 300,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "bootstrap": True,
            "max_samples": None,
            "ccp_alpha": 0.0,
            "weighting": "none",
        },
        {
            "n_estimators": 100,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "bootstrap": True,
            "max_samples": None,
            "ccp_alpha": 0.0,
            "weighting": "none",
        },
        {
            "n_estimators": 192,
            "criterion": "squared_error",
            "max_depth": 24,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "bootstrap": True,
            "max_samples": 0.9,
            "ccp_alpha": 0.0,
            "weighting": "operation_balanced",
        },
        {
            "n_estimators": 192,
            "criterion": "friedman_mse",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.6,
            "bootstrap": False,
            "max_samples": None,
            "ccp_alpha": 0.0,
            "weighting": "none",
        },
    ]
    rng = np.random.default_rng(seed)
    depths = [None, None, 8, 12, 16, 20, 24, 32, 48]
    tree_counts = [64, 96, 128, 160, 192, 256, 384]
    min_splits = [2, 2, 4, 6, 8, 12, 16, 24]
    min_leaves = [1, 1, 1, 2, 3, 4, 6, 8, 12]
    feature_fractions = [1.0, 1.0, 0.8, 0.6, 0.4, 0.25, "sqrt"]
    alpha_values = [0.0, 0.0, 0.0, 1.0e-7, 1.0e-6, 1.0e-5]
    weighting_values = ["none", "none", "operation_balanced"]
    candidates = list(anchors)
    seen = {canonical_params(item) for item in candidates}
    while len(candidates) < len(anchors) + num_random:
        bootstrap = bool(rng.choice([True, True, True, False]))
        params = {
            "n_estimators": int(rng.choice(tree_counts)),
            "criterion": str(rng.choice(["squared_error", "squared_error", "friedman_mse"])),
            "max_depth": rng.choice(depths),
            "min_samples_split": int(rng.choice(min_splits)),
            "min_samples_leaf": int(rng.choice(min_leaves)),
            "max_features": rng.choice(feature_fractions),
            "bootstrap": bootstrap,
            "max_samples": (
                rng.choice([None, None, 0.7, 0.85, 0.95]) if bootstrap else None
            ),
            "ccp_alpha": float(rng.choice(alpha_values)),
            "weighting": str(rng.choice(weighting_values)),
        }
        if params["max_depth"] is not None:
            params["max_depth"] = int(params["max_depth"])
        if params["max_features"] != "sqrt":
            params["max_features"] = float(params["max_features"])
        if params["max_samples"] is not None:
            params["max_samples"] = float(params["max_samples"])
        key = canonical_params(params)
        if key not in seen:
            seen.add(key)
            candidates.append(params)
    return candidates


def build_forest(params: dict, seed: int):
    from sklearn.ensemble import RandomForestRegressor

    forest_params = {key: value for key, value in params.items() if key != "weighting"}
    return RandomForestRegressor(
        **forest_params,
        random_state=seed,
        n_jobs=-1,
    )


def filter_by_operation(
    times: np.ndarray,
    observations: np.ndarray,
    operations: np.ndarray,
    params: dict,
) -> np.ndarray:
    output = np.full(observations.shape, np.nan, dtype=np.float64)
    for operation_id in sorted(set(operations.tolist())):
        indices = np.flatnonzero(operations == operation_id)
        order = indices[np.argsort(times[indices], kind="stable")]
        filtered = kf_online(
            times=times[order].tolist(),
            obs=observations[order].tolist(),
            base_R=float(params["base_R"]),
            q_pos=float(params["q_pos"]),
            q_vel=float(params["q_vel"]),
            reset_gap=10.0,
            history_len=0,
            init_mode="use_obs",
            default_value=0.0,
            pos_var0=0.05,
            vel_var0=0.01,
            warmup_frames=0,
        )
        output[order] = np.asarray(filtered, dtype=np.float64)
    return output


def tune_kalman(
    y_true: np.ndarray,
    predictions: np.ndarray,
    operations: np.ndarray,
    times: np.ndarray,
) -> Tuple[dict, np.ndarray, List[dict]]:
    rows = []
    best = None
    best_pred = None
    for base_R in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        for q_pos in (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
            for q_vel in (1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4):
                params = {"base_R": base_R, "q_pos": q_pos, "q_vel": q_vel}
                filtered = filter_by_operation(times, predictions, operations, params)
                metrics = eval_metrics(y_true, filtered)
                macro = macro_rmse(y_true, filtered, operations)
                row = {
                    **params,
                    "val_pooled_MAE": metrics["MAE"],
                    "val_pooled_RMSE": metrics["RMSE"],
                    "val_pooled_Bias": metrics["Bias"],
                    "val_macro_RMSE": macro,
                }
                rows.append(row)
                score = (macro, metrics["RMSE"])
                if best is None or score < best[0]:
                    best = (score, params, metrics)
                    best_pred = filtered
    assert best is not None and best_pred is not None
    payload = {
        "parameters": best[1],
        "validation_metrics": best[2],
        "validation_macro_RMSE": best[0][0],
        "selection_objective": "unweighted operation-macro RMSE on operations 11-13",
    }
    return payload, best_pred, rows


SEARCH_FIELDS = [
    "candidate_id",
    "n_estimators",
    "criterion",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "max_samples",
    "ccp_alpha",
    "weighting",
    "val_pooled_MAE",
    "val_pooled_RMSE",
    "val_pooled_Bias",
    "val_pooled_Corr",
    "val_macro_RMSE",
    "fit_seconds",
]


def run_tune(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "frozen_selection.json"
    if frozen_path.exists() and not args.force:
        raise FileExistsError(
            f"Frozen selection already exists: {frozen_path}; use --force to replace it"
        )

    X, y, splits, operations, times, records, provenance = load_arrays(
        args.manifest.resolve(), args.cache.resolve()
    )
    train = splits == "train"
    val = splits == "val"
    # Deliberately do not construct or evaluate a test target array here.
    X_train, y_train, op_train = X[train], y[train], operations[train]
    X_val, y_val, op_val, time_val = X[val], y[val], operations[val], times[val]
    print(
        f"[protocol] tune only: train={len(y_train)} ops={sorted(set(op_train.tolist()))}; "
        f"val={len(y_val)} ops={sorted(set(op_val.tolist()))}; test is not evaluated",
        flush=True,
    )

    candidates = candidate_parameters(args.num_random, args.seed)
    search_rows = []
    best_score = None
    best_params = None
    for index, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        model = build_forest(params, args.seed)
        weights = (
            operation_balanced_weights(op_train)
            if params["weighting"] == "operation_balanced"
            else None
        )
        model.fit(X_train, y_train, sample_weight=weights)
        pred = model.predict(X_val)
        metrics = eval_metrics(y_val, pred)
        macro = macro_rmse(y_val, pred, op_val)
        elapsed = time.perf_counter() - started
        row = {
            "candidate_id": index,
            **params,
            "val_pooled_MAE": metrics["MAE"],
            "val_pooled_RMSE": metrics["RMSE"],
            "val_pooled_Bias": metrics["Bias"],
            "val_pooled_Corr": metrics["Corr"],
            "val_macro_RMSE": macro,
            "fit_seconds": elapsed,
        }
        search_rows.append(row)
        write_csv(output_dir / "search_results.csv", search_rows, SEARCH_FIELDS)
        score = (macro, metrics["RMSE"])
        if best_score is None or score < best_score:
            best_score = score
            best_params = dict(params)
        print(
            f"[search {index:03d}/{len(candidates):03d}] "
            f"macro={macro:.6f} pooled={metrics['RMSE']:.6f} "
            f"best_macro={best_score[0]:.6f} seconds={elapsed:.2f}",
            flush=True,
        )

    assert best_params is not None
    print(f"[freeze] refitting best parameters: {best_params}", flush=True)
    best_model = build_forest(best_params, args.seed)
    best_weights = (
        operation_balanced_weights(op_train)
        if best_params["weighting"] == "operation_balanced"
        else None
    )
    best_model.fit(X_train, y_train, sample_weight=best_weights)
    val_pred = np.asarray(best_model.predict(X_val), dtype=np.float64)
    val_metrics = eval_metrics(y_val, val_pred)
    val_macro = macro_rmse(y_val, val_pred, op_val)
    kalman_selection, val_pred_kf, kalman_rows = tune_kalman(
        y_val, val_pred, op_val, time_val
    )

    model_path = output_dir / "best_model.pkl"
    temporary_model = model_path.with_name(model_path.name + ".building")
    with temporary_model.open("wb") as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_model, model_path)

    validation_rows = []
    val_indices = np.flatnonzero(val)
    for local_index, global_index in enumerate(val_indices):
        record = records[global_index]
        validation_rows.append(
            {
                "sample_id": record["sample_id"],
                "operation_id": int(record["operation_id"]),
                "timestamp": record["timestamp"],
                "gt": y_val[local_index],
                "pred": val_pred[local_index],
                "pred_kf": val_pred_kf[local_index],
            }
        )
    write_csv(
        output_dir / "validation_predictions.csv",
        validation_rows,
        ("sample_id", "operation_id", "timestamp", "gt", "pred", "pred_kf"),
    )
    kf_fields = list(kalman_rows[0].keys())
    write_csv(output_dir / "kalman_search_results.csv", kalman_rows, kf_fields)

    model_sha = sha256_file(model_path)
    frozen = {
        "protocol": "fixed-operation-day-split-rf-tuning",
        "selection_status": "frozen_before_test_evaluation",
        "selection_objective": "unweighted validation-operation macro raw RMSE",
        "train_operations": EXPECTED_SPLITS["train"],
        "validation_operations": EXPECTED_SPLITS["val"],
        "test_operations_not_evaluated_in_tune_stage": EXPECTED_SPLITS["test"],
        "num_candidates": len(candidates),
        "random_seed": args.seed,
        "best_parameters": best_params,
        "validation_raw_metrics": val_metrics,
        "validation_raw_macro_RMSE": val_macro,
        "kalman_selection": kalman_selection,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "feature_cache_path": str(args.cache.resolve()),
        "feature_cache_sha256": sha256_file(args.cache.resolve()),
        "manifest_audit": provenance["manifest_audit"],
    }
    json_dump(frozen_path, frozen)
    print(f"[done] frozen selection: {frozen_path}", flush=True)
    print(
        f"[done] validation raw RMSE={val_metrics['RMSE']:.6f}; "
        f"KF RMSE={kalman_selection['validation_metrics']['RMSE']:.6f}",
        flush=True,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    frozen_path = output_dir / "frozen_selection.json"
    result_path = output_dir / "test_summary.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Run tune stage first: {frozen_path}")
    if result_path.exists() and not args.force:
        raise FileExistsError(
            f"Test was already evaluated: {result_path}; refusing repeated evaluation"
        )
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    model_path = Path(frozen["model_path"])
    checks = {
        "model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "feature_cache_sha256": sha256_file(args.cache.resolve()),
    }
    for key, observed in checks.items():
        if observed != frozen[key]:
            raise RuntimeError(
                f"Frozen artifact changed before test evaluation: {key} "
                f"frozen={frozen[key]} observed={observed}"
            )

    X, y, splits, operations, times, records, provenance = load_arrays(
        args.manifest.resolve(), args.cache.resolve()
    )
    test = splits == "test"
    X_test, y_test = X[test], y[test]
    op_test, time_test = operations[test], times[test]
    if sorted(set(op_test.tolist())) != EXPECTED_SPLITS["test"]:
        raise RuntimeError("Test operations differ from the frozen protocol")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    pred = np.asarray(model.predict(X_test), dtype=np.float64)
    kf_params = frozen["kalman_selection"]["parameters"]
    pred_kf = filter_by_operation(time_test, pred, op_test, kf_params)
    raw_metrics = eval_metrics(y_test, pred)
    kf_metrics = eval_metrics(y_test, pred_kf)
    per_operation = operation_rows(y_test, pred, pred_kf, op_test)

    test_rows = []
    test_indices = np.flatnonzero(test)
    for local_index, global_index in enumerate(test_indices):
        record = records[global_index]
        test_rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "operation_id": int(record["operation_id"]),
                "date": record["date"],
                "timestamp": record["timestamp"],
                "gt": y_test[local_index],
                "pred": pred[local_index],
                "pred_kf": pred_kf[local_index],
            }
        )
    write_csv(
        output_dir / "test_predictions.csv",
        test_rows,
        ("sample_id", "source", "operation_id", "date", "timestamp", "gt", "pred", "pred_kf"),
    )
    operation_fields = list(per_operation[0].keys())
    write_csv(output_dir / "test_per_operation_metrics.csv", per_operation, operation_fields)

    summary = {
        "protocol": "fixed-operation-day-split-rf-tuned-test-once",
        "selection_file": str(frozen_path),
        "selection_file_sha256": sha256_file(frozen_path),
        "test_evaluation_number": 1,
        "test_operations": EXPECTED_SPLITS["test"],
        "num_test_frames": int(test.sum()),
        "raw": raw_metrics,
        "kalman": kf_metrics,
        "raw_operation_macro_RMSE": macro_rmse(y_test, pred, op_test),
        "kalman_operation_macro_RMSE": macro_rmse(y_test, pred_kf, op_test),
        "target_below_0p03": {
            "raw_RMSE": bool(raw_metrics["RMSE"] < 0.03),
            "kalman_RMSE": bool(kf_metrics["RMSE"] < 0.03),
        },
        "frozen_artifact_checks": checks,
        "manifest_audit": provenance["manifest_audit"],
    }
    json_dump(result_path, summary)
    print(f"[done] test evaluated once: {result_path}", flush=True)
    print(
        f"[result] raw RMSE={raw_metrics['RMSE']:.6f}; "
        f"KF RMSE={kf_metrics['RMSE']:.6f}; "
        f"target raw={summary['target_below_0p03']['raw_RMSE']} "
        f"KF={summary['target_below_0p03']['kalman_RMSE']}",
        flush=True,
    )


def run_refit(args: argparse.Namespace) -> None:
    """Refit selected RF settings on train+validation before opening test."""
    output_dir = args.output.resolve()
    frozen_path = output_dir / "frozen_selection.json"
    result_path = output_dir / "test_summary.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Run tune stage first: {frozen_path}")
    if result_path.exists():
        raise RuntimeError(
            "Refit is forbidden after test evaluation; the test set has already been opened"
        )
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("selection_status") == (
        "final_model_refit_on_train_and_validation_frozen_before_test"
    ) and not args.force:
        raise FileExistsError("Final train+validation refit is already frozen")

    model_path = Path(frozen["model_path"])
    checks = {
        "model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "feature_cache_sha256": sha256_file(args.cache.resolve()),
    }
    for key, observed in checks.items():
        if observed != frozen[key]:
            raise RuntimeError(
                f"Artifact changed before final refit: {key} "
                f"frozen={frozen[key]} observed={observed}"
            )

    X, y, splits, operations, _, _, provenance = load_arrays(
        args.manifest.resolve(), args.cache.resolve()
    )
    final_fit = np.isin(splits, ["train", "val"])
    X_fit = X[final_fit]
    y_fit = y[final_fit]
    op_fit = operations[final_fit]
    expected_fit_operations = EXPECTED_SPLITS["train"] + EXPECTED_SPLITS["val"]
    if sorted(set(op_fit.tolist())) != expected_fit_operations:
        raise RuntimeError("Final refit operations differ from the declared protocol")

    params = dict(frozen["best_parameters"])
    model = build_forest(params, int(frozen["random_seed"]))
    weights = (
        operation_balanced_weights(op_fit)
        if params["weighting"] == "operation_balanced"
        else None
    )
    started = time.perf_counter()
    model.fit(X_fit, y_fit, sample_weight=weights)
    elapsed = time.perf_counter() - started

    temporary_model = model_path.with_name(model_path.name + ".building")
    with temporary_model.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_model, model_path)
    pre_refit_sha = frozen["model_sha256"]
    frozen.update(
        {
            "selection_status": (
                "final_model_refit_on_train_and_validation_frozen_before_test"
            ),
            "final_fit_operations": expected_fit_operations,
            "final_fit_num_frames": int(final_fit.sum()),
            "final_fit_used_test_operations": False,
            "final_fit_seconds": elapsed,
            "pre_refit_train_only_model_sha256": pre_refit_sha,
            "model_sha256": sha256_file(model_path),
            "manifest_audit": provenance["manifest_audit"],
            "validation_artifacts_note": (
                "Validation predictions and selection metrics were produced by the "
                "train-only selection model before this final train+validation refit."
            ),
        }
    )
    json_dump(frozen_path, frozen)
    print(
        f"[done] final RF refit: frames={int(final_fit.sum())} "
        f"ops={expected_fit_operations} seconds={elapsed:.2f}",
        flush=True,
    )
    print(f"[done] frozen model sha256={frozen['model_sha256']}", flush=True)
    print("[protocol] test operations 14-18 were not evaluated", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "refit", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-random", type=int, default=44)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_random < 0:
        parser.error("--num-random must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.stage == "tune":
        run_tune(args)
    elif args.stage == "refit":
        run_refit(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
