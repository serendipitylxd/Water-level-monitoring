#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune the Random Forest without changing data, splits, or other models.

The RF-specific workflow has three auditable stages:

1. ``tune`` selects a physics-anchor coefficient by leave-one-operation-out
   validation on development operations 1--13, then selects ordinary forest
   hyperparameters by fitting operations 1--10 and validating on 11--13.
   Test operations 14--18 are not evaluated.
2. ``refit`` freezes the selected settings and fits operations 1--13.
3. ``evaluate`` verifies all frozen hashes and evaluates operations 14--18.

Only the Random Forest representation is changed.  It augments the unchanged
wall-profile cache with robust summary statistics and predicts a residual
relative to a fixed fraction of the lowest wall-belt height.  This gives the
forest limited extrapolation ability across collection days while retaining
RandomForestRegressor as the only learned estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.tune_random_forest_operation_split import (  # noqa: E402
    EXPECTED_SPLITS,
    build_forest,
    candidate_parameters,
    filter_by_operation,
    load_arrays,
    macro_rmse,
    operation_balanced_weights,
    sha256_file,
    tune_kalman,
)
from utils.metrics import eval_metrics  # noqa: E402
from utils.models import RandomForestWL  # noqa: E402


DEFAULT_MANIFEST = (
    REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv"
)
DEFAULT_CACHE = (
    REPO_ROOT
    / "outputs_operation_split/test_only_8000/feature_cache/waterlevel_test_only_default.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs_operation_split/test_only_8000/rf_tuned_robust_anchor"
)
FEATURE_MODE = "raw_stats"
RESIDUAL_ANCHOR = "min_both"
COEFFICIENT_CANDIDATES = (
    0.0,
    0.15,
    0.25,
    0.35,
    0.40,
    0.425,
    0.45,
    0.475,
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.70,
    0.75,
)
COEFFICIENT_SELECTION_PARAMS = {
    "n_estimators": 96,
    "criterion": "squared_error",
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.25,
    "bootstrap": True,
    "max_samples": None,
    "ccp_alpha": 0.0,
    "weighting": "none",
}
CURRENT_RF_PARAMS = {
    "n_estimators": 160,
    "criterion": "squared_error",
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.25,
    "bootstrap": True,
    "max_samples": None,
    "ccp_alpha": 0.0,
    "weighting": "none",
}


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_rf_inputs(
    manifest: Path, cache: Path, residual_anchor: str, feature_mode: str = FEATURE_MODE
):
    _, targets, splits, operations, times, records, provenance = load_arrays(
        manifest, cache
    )
    with np.load(str(cache), allow_pickle=False) as payload:
        sequences = np.asarray(payload["sequences"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=bool)
    if sequences.shape[:2] != masks.shape or len(sequences) != len(targets):
        raise RuntimeError(
            f"RF input mismatch: sequences={sequences.shape}, masks={masks.shape}, "
            f"targets={targets.shape}"
        )
    transformer = RandomForestWL(
        n_estimators=1,
        feature_mode=feature_mode,
        residual_anchor=residual_anchor,
        residual_coefficient=0.0,
    )
    features, anchor = transformer._engineered_features(sequences, masks)
    return (
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
    )


def fit_forest(
    params: dict,
    seed: int,
    features: np.ndarray,
    targets: np.ndarray,
    anchor: np.ndarray,
    coefficient: float,
    operations: np.ndarray,
):
    model = build_forest(params, seed)
    weights = (
        operation_balanced_weights(operations)
        if params["weighting"] == "operation_balanced"
        else None
    )
    model.fit(
        features,
        targets - float(coefficient) * anchor,
        sample_weight=weights,
    )
    return model


def predict_forest(
    model,
    features: np.ndarray,
    anchor: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    return (
        np.asarray(model.predict(features), dtype=np.float64)
        + float(coefficient) * anchor
    )


def wrap_forest(
    model,
    params: dict,
    coefficient: float,
    residual_anchor: str,
    feature_mode: str = FEATURE_MODE,
) -> RandomForestWL:
    estimator_params = {key: value for key, value in params.items() if key != "weighting"}
    wrapper = RandomForestWL(
        **estimator_params,
        random_state=getattr(model, "random_state", 42),
        n_jobs=getattr(model, "n_jobs", -1),
        feature_mode=feature_mode,
        residual_anchor=residual_anchor,
        residual_coefficient=float(coefficient),
    )
    wrapper.estimator = model
    return wrapper


def parameter_candidates(
    num_random: int, seed: int, weighting_policy: str = "any"
) -> List[dict]:
    candidates = [dict(CURRENT_RF_PARAMS)] + candidate_parameters(num_random, seed)
    if weighting_policy != "any":
        candidates = [
            params
            for params in candidates
            if params.get("weighting", "none") == weighting_policy
        ]
    unique = []
    seen = set()
    for params in candidates:
        key = json.dumps(params, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def extended_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = eval_metrics(y_true, y_pred)
    metrics["MaxAE"] = float(np.max(np.abs(y_pred - y_true)))
    return metrics


def causal_exponential_by_operation(
    predictions: np.ndarray,
    operations: np.ndarray,
    times: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply a causal first-order stabilizer and reset it at each operation."""
    output = np.empty_like(predictions, dtype=np.float64)
    for operation in sorted(set(operations.tolist())):
        indices = np.flatnonzero(operations == operation)
        order = indices[np.argsort(times[indices], kind="stable")]
        state = float(predictions[order[0]])
        output[order[0]] = state
        for index in order[1:]:
            state = float(alpha) * float(predictions[index]) + (
                1.0 - float(alpha)
            ) * state
            output[index] = state
    return output


def tune_causal_stabilizer(
    y_true: np.ndarray,
    predictions: np.ndarray,
    operations: np.ndarray,
    times: np.ndarray,
) -> Tuple[dict, np.ndarray, List[dict]]:
    rows = []
    best = None
    for alpha in (1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1):
        stabilized = causal_exponential_by_operation(
            predictions, operations, times, alpha
        )
        metrics = extended_metrics(y_true, stabilized)
        macro = macro_rmse(y_true, stabilized, operations)
        row = {
            "alpha": alpha,
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
            "val_Bias": metrics["Bias"],
            "val_MaxAE": metrics["MaxAE"],
            "val_macro_RMSE": macro,
        }
        rows.append(row)
        score = (macro, metrics["RMSE"])
        if best is None or score < best[0]:
            best = (score, float(alpha), metrics, stabilized)
    assert best is not None
    selection = {
        "alpha": best[1],
        "validation_metrics": best[2],
        "validation_macro_RMSE": best[0][0],
        "selection_objective": (
            "validation operation-macro RMSE; pooled RMSE tie-break"
        ),
        "causal": True,
        "reset_at_operation_boundary": True,
        "used_test_operations": False,
    }
    return selection, best[3], rows


def per_operation_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_causal: np.ndarray,
    y_kf: np.ndarray,
    operations: np.ndarray,
) -> List[dict]:
    rows = []
    for operation in sorted(set(operations.tolist())):
        selected = operations == operation
        raw = extended_metrics(y_true[selected], y_pred[selected])
        causal = extended_metrics(y_true[selected], y_causal[selected])
        filtered = extended_metrics(y_true[selected], y_kf[selected])
        rows.append(
            {
                "operation_id": int(operation),
                "count": int(selected.sum()),
                **{f"raw_{key}": raw[key] for key in ("MAE", "RMSE", "Bias", "MaxAE", "Corr")},
                **{f"causal_{key}": causal[key] for key in ("MAE", "RMSE", "Bias", "MaxAE", "Corr")},
                **{f"kalman_{key}": filtered[key] for key in ("MAE", "RMSE", "Bias", "MaxAE", "Corr")},
            }
        )
    return rows


def run_tune(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "frozen_selection.json"
    if frozen_path.exists() and not args.force:
        raise FileExistsError(f"Selection already exists: {frozen_path}")

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
        args.manifest.resolve(),
        args.cache.resolve(),
        args.residual_anchor,
        args.feature_mode,
    )
    development = np.isin(operations, np.arange(1, 14))

    coefficient_rows = []
    coefficient_best = None
    for coefficient in COEFFICIENT_CANDIDATES:
        fold_rmses = []
        fold_biases = []
        for operation in range(1, 14):
            test = operations == operation
            train = development & ~test
            model = fit_forest(
                COEFFICIENT_SELECTION_PARAMS,
                4200 + operation,
                features[train],
                targets[train],
                anchor[train],
                coefficient,
                operations[train],
            )
            prediction = predict_forest(
                model, features[test], anchor[test], coefficient
            )
            error = prediction - targets[test]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            bias = float(np.mean(error))
            fold_rmses.append(rmse)
            fold_biases.append(bias)
            coefficient_rows.append(
                {
                    "coefficient": coefficient,
                    "held_out_operation": operation,
                    "num_test_frames": int(test.sum()),
                    "RMSE": rmse,
                    "Bias": bias,
                }
            )
        if args.coefficient_objective == "macro_absolute_bias":
            score = (
                float(np.mean(np.abs(fold_biases))),
                max(fold_rmses),
                float(np.mean(fold_rmses)),
            )
        elif args.coefficient_objective == "robust_average_rmse":
            worst_rmse = max(fold_rmses)
            mean_rmse = float(np.mean(fold_rmses))
            score = (0.5 * (worst_rmse + mean_rmse), worst_rmse, mean_rmse)
        else:
            score = (max(fold_rmses), float(np.mean(fold_rmses)))
        print(
            f"[anchor {coefficient:.2f}] worst={max(fold_rmses):.6f} "
            f"macro={float(np.mean(fold_rmses)):.6f} "
            f"macro_abs_bias={float(np.mean(np.abs(fold_biases))):.6f}",
            flush=True,
        )
        if coefficient_best is None or score < coefficient_best[0]:
            coefficient_best = (
                score,
                float(coefficient),
                max(fold_rmses),
                float(np.mean(fold_rmses)),
                float(np.std(fold_rmses, ddof=1)),
                float(np.mean(np.abs(fold_biases))),
            )
    assert coefficient_best is not None
    coefficient = coefficient_best[1]
    atomic_csv(
        output / "coefficient_operation_cv.csv",
        coefficient_rows,
        ("coefficient", "held_out_operation", "num_test_frames", "RMSE", "Bias"),
    )

    train = splits == "train"
    val = splits == "val"
    search_rows = []
    best = None
    candidates = parameter_candidates(
        args.num_random, args.seed, args.weighting_policy
    )
    for candidate_id, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        model = fit_forest(
            params,
            args.seed,
            features[train],
            targets[train],
            anchor[train],
            coefficient,
            operations[train],
        )
        prediction = predict_forest(
            model, features[val], anchor[val], coefficient
        )
        metrics = extended_metrics(targets[val], prediction)
        macro = macro_rmse(targets[val], prediction, operations[val])
        row = {
            "candidate_id": candidate_id,
            **params,
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
            "val_Bias": metrics["Bias"],
            "val_MaxAE": metrics["MaxAE"],
            "val_macro_RMSE": macro,
            "fit_seconds": time.perf_counter() - started,
        }
        search_rows.append(row)
        atomic_csv(output / "parameter_search.csv", search_rows, tuple(row.keys()))
        score = (macro, metrics["RMSE"])
        if best is None or score < best[0]:
            best = (score, dict(params), model, prediction, metrics)
        print(
            f"[forest {candidate_id:03d}/{len(candidates):03d}] "
            f"macro={macro:.6f} pooled={metrics['RMSE']:.6f} "
            f"best={best[0][0]:.6f}",
            flush=True,
        )
    assert best is not None

    causal_selection, val_causal, causal_rows = tune_causal_stabilizer(
        targets[val], best[3], operations[val], times[val]
    )
    atomic_csv(
        output / "causal_stabilizer_search.csv",
        causal_rows,
        tuple(causal_rows[0].keys()),
    )
    kalman_selection, val_kf, kalman_rows = tune_kalman(
        targets[val], val_causal, operations[val], times[val]
    )
    atomic_csv(
        output / "kalman_search.csv", kalman_rows, tuple(kalman_rows[0].keys())
    )
    val_rows = []
    for local, global_index in enumerate(np.flatnonzero(val)):
        record = records[global_index]
        val_rows.append(
            {
                "sample_id": record["sample_id"],
                "operation_id": int(record["operation_id"]),
                "timestamp": record["timestamp"],
                "gt": targets[val][local],
                "pred": best[3][local],
                "pred_causal": val_causal[local],
                "pred_kf": val_kf[local],
            }
        )
    atomic_csv(
        output / "validation_predictions.csv",
        val_rows,
        (
            "sample_id",
            "operation_id",
            "timestamp",
            "gt",
            "pred",
            "pred_causal",
            "pred_kf",
        ),
    )

    model_path = output / "best_model.pkl"
    wrapper = wrap_forest(
        best[2], best[1], coefficient, args.residual_anchor, args.feature_mode
    )
    temporary = model_path.with_name(model_path.name + ".building")
    with temporary.open("wb") as handle:
        pickle.dump(wrapper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, model_path)

    frozen = {
        "protocol": "rf-only-operation-disjoint-tuning",
        "selection_status": "train-only-model-frozen-before-test",
        "feature_mode": args.feature_mode,
        "residual_anchor": args.residual_anchor,
        "residual_coefficient": coefficient,
        "coefficient_selection": {
            "candidate_values": list(COEFFICIENT_CANDIDATES),
            "operations": list(range(1, 14)),
            "objective": (
                "minimum operation-macro absolute bias; worst-operation and "
                "macro RMSE tie-breaks"
                if args.coefficient_objective == "macro_absolute_bias"
                else (
                    "minimum mean of worst-operation and operation-macro RMSE; "
                    "component RMSE tie-breaks"
                    if args.coefficient_objective == "robust_average_rmse"
                    else "minimum worst held-out-operation RMSE; macro RMSE tie-break"
                )
            ),
            "objective_key": args.coefficient_objective,
            "worst_operation_RMSE": coefficient_best[2],
            "macro_RMSE": coefficient_best[3],
            "sample_sd_RMSE": coefficient_best[4],
            "macro_absolute_bias": coefficient_best[5],
            "used_test_operations": False,
        },
        "parameter_selection": {
            "train_operations": EXPECTED_SPLITS["train"],
            "validation_operations": EXPECTED_SPLITS["val"],
            "objective": "validation operation-macro RMSE; pooled RMSE tie-break",
            "num_candidates": len(candidates),
            "weighting_policy": args.weighting_policy,
            "best_parameters": best[1],
            "validation_metrics": best[4],
            "validation_macro_RMSE": best[0][0],
            "used_test_operations": False,
        },
        "causal_stabilizer_selection": causal_selection,
        "kalman_selection": kalman_selection,
        "random_seed": args.seed,
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "feature_cache_path": str(args.cache.resolve()),
        "feature_cache_sha256": sha256_file(args.cache.resolve()),
        "manifest_audit": provenance["manifest_audit"],
        "test_operations_not_evaluated": EXPECTED_SPLITS["test"],
    }
    atomic_json(frozen_path, frozen)
    print(
        f"[done] coefficient={coefficient:.2f} val_RMSE={best[4]['RMSE']:.6f} "
        f"val_macro={best[0][0]:.6f}; "
        f"causal_alpha={causal_selection['alpha']:.2f} "
        f"causal_val_RMSE={causal_selection['validation_metrics']['RMSE']:.6f}",
        flush=True,
    )


def verify_frozen(frozen: dict, manifest: Path, cache: Path) -> Path:
    model_path = Path(frozen["model_path"])
    checks = {
        "model_sha256": sha256_file(model_path),
        "manifest_sha256": sha256_file(manifest),
        "feature_cache_sha256": sha256_file(cache),
    }
    for key, observed in checks.items():
        if frozen[key] != observed:
            raise RuntimeError(
                f"Frozen RF artifact changed: {key} expected={frozen[key]} "
                f"observed={observed}"
            )
    return model_path


def run_refit(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    frozen_path = output / "frozen_selection.json"
    test_path = output / "test_summary.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Run tune first: {frozen_path}")
    if test_path.exists():
        raise RuntimeError("Refit is forbidden after test evaluation")
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen["selection_status"] == "train-validation-refit-frozen-before-test" and not args.force:
        raise FileExistsError("Final RF refit already exists")
    model_path = verify_frozen(
        frozen, args.manifest.resolve(), args.cache.resolve()
    )
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
        args.manifest.resolve(),
        args.cache.resolve(),
        frozen["residual_anchor"],
        frozen["feature_mode"],
    )
    fit = np.isin(splits, ("train", "val"))
    params = frozen["parameter_selection"]["best_parameters"]
    coefficient = float(frozen["residual_coefficient"])
    model = fit_forest(
        params,
        int(frozen["random_seed"]),
        features[fit],
        targets[fit],
        anchor[fit],
        coefficient,
        operations[fit],
    )
    wrapper = wrap_forest(
        model,
        params,
        coefficient,
        frozen["residual_anchor"],
        frozen["feature_mode"],
    )
    temporary = model_path.with_name(model_path.name + ".building")
    with temporary.open("wb") as handle:
        pickle.dump(wrapper, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, model_path)
    frozen.update(
        {
            "selection_status": "train-validation-refit-frozen-before-test",
            "final_fit_operations": list(range(1, 14)),
            "final_fit_frames": int(fit.sum()),
            "final_fit_used_test_operations": False,
            "model_sha256": sha256_file(model_path),
            "manifest_audit": provenance["manifest_audit"],
        }
    )
    atomic_json(frozen_path, frozen)
    print(
        f"[done] RF refit on {int(fit.sum())} frames; test not evaluated; "
        f"sha256={frozen['model_sha256']}",
        flush=True,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    frozen_path = output / "frozen_selection.json"
    result_path = output / "test_summary.json"
    if result_path.exists() and not args.force:
        raise FileExistsError("RF test already evaluated; refusing repeated evaluation")
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen["selection_status"] != "train-validation-refit-frozen-before-test":
        raise RuntimeError("Run the RF refit stage before evaluation")
    model_path = verify_frozen(
        frozen, args.manifest.resolve(), args.cache.resolve()
    )
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
        args.manifest.resolve(),
        args.cache.resolve(),
        frozen["residual_anchor"],
        frozen["feature_mode"],
    )
    test = splits == "test"
    if sorted(set(operations[test].tolist())) != EXPECTED_SPLITS["test"]:
        raise RuntimeError("RF test operations differ from the fixed protocol")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    prediction = np.asarray(
        model.predict(sequences[test], key_padding_mask=masks[test]), dtype=np.float64
    )
    causal_alpha = float(frozen["causal_stabilizer_selection"]["alpha"])
    prediction_causal = causal_exponential_by_operation(
        prediction, operations[test], times[test], causal_alpha
    )
    kf_params = frozen["kalman_selection"]["parameters"]
    prediction_kf = filter_by_operation(
        times[test], prediction_causal, operations[test], kf_params
    )
    raw = extended_metrics(targets[test], prediction)
    causal = extended_metrics(targets[test], prediction_causal)
    filtered = extended_metrics(targets[test], prediction_kf)
    rows = per_operation_rows(
        targets[test], prediction, prediction_causal, prediction_kf, operations[test]
    )
    atomic_csv(output / "test_per_operation_metrics.csv", rows, tuple(rows[0].keys()))

    prediction_rows = []
    for local, global_index in enumerate(np.flatnonzero(test)):
        record = records[global_index]
        prediction_rows.append(
            {
                "sample_id": record["sample_id"],
                "source": record["source"],
                "operation_id": int(record["operation_id"]),
                "date": record["date"],
                "timestamp": record["timestamp"],
                "gt": targets[test][local],
                "pred": prediction[local],
                "pred_causal": prediction_causal[local],
                "pred_kf": prediction_kf[local],
            }
        )
    atomic_csv(
        output / "test_predictions.csv",
        prediction_rows,
        (
            "sample_id",
            "source",
            "operation_id",
            "date",
            "timestamp",
            "gt",
            "pred",
            "pred_causal",
            "pred_kf",
        ),
    )
    summary = {
        "protocol": "posthoc-rf-only-fixed-operation-and-held-out-day-test",
        "test_status_note": (
            "The final-day labels had already been opened for earlier RF variants; "
            "this optimized RF result is post-hoc and is not an untouched confirmatory test."
        ),
        "test_operations": EXPECTED_SPLITS["test"],
        "num_test_frames": int(test.sum()),
        "raw": raw,
        "causal_stabilized": causal,
        "kalman_after_causal": filtered,
        "raw_operation_macro_RMSE": macro_rmse(
            targets[test], prediction, operations[test]
        ),
        "causal_operation_macro_RMSE": macro_rmse(
            targets[test], prediction_causal, operations[test]
        ),
        "kalman_operation_macro_RMSE": macro_rmse(
            targets[test], prediction_kf, operations[test]
        ),
        "target_below_0p03": {
            "raw_RMSE": bool(raw["RMSE"] < 0.03),
            "causal_RMSE": bool(causal["RMSE"] < 0.03),
            "kalman_RMSE": bool(filtered["RMSE"] < 0.03),
        },
        "feature_mode": frozen["feature_mode"],
        "residual_anchor": frozen["residual_anchor"],
        "residual_coefficient": frozen["residual_coefficient"],
        "model_sha256": frozen["model_sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "feature_cache_sha256": frozen["feature_cache_sha256"],
        "manifest_audit": provenance["manifest_audit"],
    }
    atomic_json(result_path, summary)
    print(
        f"[result] raw MAE={raw['MAE']:.6f} RMSE={raw['RMSE']:.6f} "
        f"Bias={raw['Bias']:+.6f} MaxAE={raw['MaxAE']:.6f}; "
        f"causal RMSE={causal['RMSE']:.6f}; KF RMSE={filtered['RMSE']:.6f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "refit", "evaluate"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-random", type=int, default=44)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--residual-anchor",
        choices=("min_both", "left_min"),
        default=RESIDUAL_ANCHOR,
        help="RF-only physics anchor representation considered before test evaluation.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=("raw_stats", "relative_stats", "relative_raw_stats"),
        default=FEATURE_MODE,
        help="Random-Forest-only wall-profile representation.",
    )
    parser.add_argument(
        "--weighting-policy",
        choices=("any", "none", "operation_balanced"),
        default="any",
        help=(
            "Restrict RF candidates by sample-weighting policy. 'none' is the "
            "ordinary unweighted RandomForestRegressor used in the main comparison."
        ),
    )
    parser.add_argument(
        "--coefficient-objective",
        choices=("worst_rmse", "robust_average_rmse", "macro_absolute_bias"),
        default="worst_rmse",
        help=(
            "Development-only objective for selecting the fixed physics-anchor "
            "coefficient across leave-one-operation-out fits on operations 1--13."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
