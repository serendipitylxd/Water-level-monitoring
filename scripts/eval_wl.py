# scripts/eval_wl.py
# -*- coding: utf-8 -*-
import os, sys, argparse, json, inspect, pickle, time, shutil, hashlib
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repository root is at the front to avoid name clashes with third-party packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.io import load_yaml, resolve_path, ensure_dir
from utils.trainer import build_model as build_existing_model
from utils import models as wl_models
from utils.metrics import eval_metrics
from utils.data import (
    load_operation_manifest,
    validate_operation_manifest,
    ManifestWLFrames,
    collate_manifest,
)
from utils.eval_utils import (
    parse_add_info_strict, load_det3d_fused_json, EvalFrames, collate_eval, kf_online
)


# ---------------------------
# Section / model detection
# ---------------------------
def detect_section(cfg: dict) -> Tuple[str, str]:
    """Detect active water-level config section.

    Compatible with:
      1) Original sections: wl_transformer / wl_retnet / wl_mamba / ...
      2) Optional baseline sections: wl_mlp / wl_svr / wl_xgboost / ...
      3) Baseline YAMLs that still use top-level wl_transformer but select the
         actual model by `section["model"]["name"]`.
    """
    if "wl_transformer" in cfg: return "wl_transformer", "transformer"
    if "wl_retnet"      in cfg: return "wl_retnet", "retnet"
    if "wl_mamba"       in cfg: return "wl_mamba", "mamba"
    if "wl_rwkv"        in cfg: return "wl_rwkv", "rwkv"
    if "wl_hyena"       in cfg: return "wl_hyena", "hyena"
    if "wl_mega"        in cfg: return "wl_mega", "mega"
    if "wl_hgrn"        in cfg: return "wl_hgrn", "hgrn"

    # Reviewer-requested simple baselines, if written as explicit sections.
    if "wl_linear_regression" in cfg: return "wl_linear_regression", "linear_regression"
    if "wl_ridge_regression"  in cfg: return "wl_ridge_regression", "ridge_regression"
    if "wl_mlp"               in cfg: return "wl_mlp", "mlp"
    if "wl_1dcnn"             in cfg: return "wl_1dcnn", "1dcnn"
    if "wl_svr"               in cfg: return "wl_svr", "svr"
    if "wl_random_forest"     in cfg: return "wl_random_forest", "random_forest"

    raise RuntimeError(
        "Config must contain one of: wl_transformer / wl_retnet / wl_mamba / "
        "wl_rwkv / wl_hyena / wl_mega / wl_hgrn / wl_linear_regression / "
        "wl_ridge_regression / wl_mlp / wl_1dcnn / wl_svr / "
        "wl_random_forest / wl_xgboost"
    )


_SKLEARN_MODEL_NAMES = {"SVRWL", "RandomForestWL", "XGBoostWL"}
_BASELINE_KIND_TO_NAME = {
    "linear_regression": "LinearRegressionWL",
    "ridge_regression": "RidgeRegressionWL",
    "mlp": "MLPWL",
    "1dcnn": "CNN1DWL",
    "svr": "SVRWL",
    "random_forest": "RandomForestWL",
    "xgboost": "XGBoostWL",
}
# YAML-only keys that should not be passed into torch/sklearn constructors.
_MODEL_CFG_CONTROL_KEYS = {"name", "flatten", "activation", "pooling", "standardize"}


def _model_name_from_cfg(kind: str, model_cfg: dict) -> str:
    model_cfg = model_cfg or {}
    return model_cfg.get("name") or _BASELINE_KIND_TO_NAME.get(kind, "")


def _is_sklearn_name(name: str) -> bool:
    return name in _SKLEARN_MODEL_NAMES


def _filter_model_kwargs(cls, model_cfg: dict) -> dict:
    """Keep only constructor-compatible kwargs."""
    raw = {k: v for k, v in (model_cfg or {}).items() if k not in _MODEL_CFG_CONTROL_KEYS}
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return raw
    allowed = {k for k in params.keys() if k != "self"}
    return {k: v for k, v in raw.items() if k in allowed}


def build_wl_model(kind: str, model_cfg: dict):
    """Build torch-style WL model from YAML.

    If `model.name` is given and the class exists in `utils.models`, instantiate
    it directly. Otherwise fall back to the original `utils.trainer.build_model`.
    Sklearn-style models are normally loaded from pickle during evaluation and
    do not need this function.
    """
    model_cfg = dict(model_cfg or {})
    name = _model_name_from_cfg(kind, model_cfg)

    if name and hasattr(wl_models, name):
        cls = getattr(wl_models, name)
        kwargs = _filter_model_kwargs(cls, model_cfg)
        return cls(**kwargs)

    if name and name in _BASELINE_KIND_TO_NAME.values():
        raise RuntimeError(
            f"Model class '{name}' is not found in utils.models. "
            f"Please update utils/models.py to include class {name}."
        )

    return build_existing_model(kind, model_cfg)


def is_sklearn_like_model(model) -> bool:
    return (not isinstance(model, nn.Module)) and hasattr(model, "predict")


def _load_pickle_model(model_path: str):
    """Load sklearn-style model saved by train_wl.py.

    The modified training script saves classical baselines as pickle. The file
    may be named model.pkl, or model.pth for backward path compatibility.
    """
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_err:
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as joblib_err:
            raise RuntimeError(
                f"Failed to load sklearn-style model from {model_path}. "
                f"pickle error: {pickle_err}; joblib error: {joblib_err}"
            ) from pickle_err


def _load_torch_state(model: nn.Module, model_path: str):
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    sd = torch.load(model_path, map_location="cpu")
    try:
        model.load_state_dict(sd, strict=False)  # Allow minor suffix/key differences
    except Exception:
        # If saved as {'model': state_dict}
        if isinstance(sd, dict) and "model" in sd:
            model.load_state_dict(sd["model"], strict=False)
        else:
            raise
    return model


def _seq_to_flat_numpy(seq, mask=None):
    """Convert padded [N,T,C] wall-profile sequence into [N,T*C] features.

    This fallback is used when a loaded classical ML object is a raw sklearn
    estimator/pipeline rather than the custom SVRWL/RandomForestWL/XGBoostWL
    wrapper. Masked bins are set to zero before flattening.
    """
    if torch.is_tensor(seq):
        x = seq.detach().cpu().float().clone()
        if mask is not None:
            m = mask.detach().cpu().bool()
            x[m] = 0.0
        return x.reshape(x.shape[0], -1).numpy()
    x = np.asarray(seq, dtype=np.float32).copy()
    if mask is not None:
        x[np.asarray(mask, dtype=bool)] = 0.0
    return x.reshape(x.shape[0], -1)


def _predict_sklearn(model, seq, mask):
    """Predict with custom wrapper or raw sklearn estimator."""
    # Custom wrappers generated for this project usually support this signature.
    try:
        pred = model.predict(seq, key_padding_mask=mask)
        return np.asarray(pred, dtype=np.float64).reshape(-1)
    except TypeError:
        pass
    except Exception:
        # Try raw flattened sklearn input below.
        pass

    X = _seq_to_flat_numpy(seq, mask)
    pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def build_or_load_eval_model(kind: str, model_cfg: dict, model_path: str, device: torch.device):
    """Return (model, is_sklearn)."""
    name = _model_name_from_cfg(kind, model_cfg)

    # Classical ML baselines: load whole estimator/wrapper from pickle.
    if _is_sklearn_name(name):
        model = _load_pickle_model(model_path)
        return model, True

    # Torch models: build architecture and load state_dict.
    model = build_wl_model(kind, model_cfg)
    if is_sklearn_like_model(model):
        # Extra robustness if a classical model was built explicitly.
        model = _load_pickle_model(model_path)
        return model, True

    model = _load_torch_state(model, model_path)
    model.to(device).eval()
    return model, False


def _build_manifest_eval_dataset(
    records,
    geom: dict,
    wl_cfg: dict,
    ev_cfg: dict,
    feature_cache_path: str = None,
):
    x_min, x_max = geom["chamber_x_range"]
    y_min, y_max = geom["chamber_y_range"]
    return ManifestWLFrames(
        records,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"],
        wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        remove_labels=wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=ev_cfg.get("ablation", {}).get(
            "use_point_removal", True
        ),
        feature_cache_path=feature_cache_path,
    )


def _kf_by_operation(
    times: np.ndarray,
    observations: np.ndarray,
    operation_ids: np.ndarray,
    **kf_kwargs,
) -> np.ndarray:
    """Run independent online filters and reset explicitly at each operation."""
    output = np.full(observations.shape, np.nan, dtype=np.float64)
    for operation_id in sorted(set(operation_ids.tolist())):
        indices = np.flatnonzero(operation_ids == operation_id)
        local_order = indices[np.argsort(times[indices], kind="stable")]
        filtered = kf_online(
            times=times[local_order].tolist(),
            obs=observations[local_order].tolist(),
            **kf_kwargs,
        )
        output[local_order] = np.asarray(filtered, dtype=np.float64)
    return output


def _prefixed_metrics(prefix: str, metrics: dict) -> dict:
    return {
        f"{prefix}_{name}": metrics.get(name)
        for name in ("MAE", "RMSE", "Bias", "Corr")
    }


def _operation_metric_rows(
    ys: np.ndarray,
    yp: np.ndarray,
    yp_kf: np.ndarray,
    operation_ids: np.ndarray,
    dates: list,
) -> list:
    rows = []
    for operation_id in sorted(set(operation_ids.tolist())):
        indices = np.flatnonzero(operation_ids == operation_id)
        raw_metrics = eval_metrics(ys[indices], yp[indices])
        row = {
            "operation_id": int(operation_id),
            "date": ",".join(sorted({str(dates[index]) for index in indices})),
            "count": int(len(indices)),
            **_prefixed_metrics("raw", raw_metrics),
        }
        if yp_kf is not None:
            row.update(
                _prefixed_metrics(
                    "kalman", eval_metrics(ys[indices], yp_kf[indices])
                )
            )
        rows.append(row)
    return rows


def _macro_operation_summary(rows: list) -> dict:
    summary = {"num_operations": len(rows)}
    metric_columns = [
        key
        for prefix in ("raw", "kalman")
        for key in (f"{prefix}_MAE", f"{prefix}_RMSE", f"{prefix}_Bias", f"{prefix}_Corr")
        if any(key in row for row in rows)
    ]
    for key in metric_columns:
        values = np.asarray(
            [row.get(key) for row in rows if row.get(key) is not None],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        summary[f"{key}_mean"] = float(values.mean()) if values.size else None
        summary[f"{key}_std"] = (
            float(values.std(ddof=1)) if values.size >= 2 else None
        )
    return summary


# ---------------------------
# Main evaluation flow
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to configs/wl_xxx.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.cfg)
    cfg_dir = os.path.dirname(os.path.abspath(args.cfg))

    section_key, model_kind = detect_section(cfg)
    sec = cfg[section_key]
    data_cfg = cfg["data"]
    wl_cfg   = cfg["waterlevel"]
    geom     = cfg["geometry"]
    ev_cfg   = cfg.get("eval", {})

    # Resolve paths
    model_path = resolve_path(cfg_dir, ev_cfg.get("model_path"))
    out_dir = resolve_path(cfg_dir, ev_cfg.get("out_dir", os.path.join("outputs", sec["out_dir"], "eval")))
    ensure_dir(out_dir)
    batch = int(ev_cfg.get("batch", sec.get("train", {}).get("batch", 60)))
    manifest_cfg = data_cfg.get("operation_manifest_path")
    manifest_mode = bool(manifest_cfg)
    split_audit = None
    manifest_path = None
    feature_cache_path = None
    if manifest_mode:
        manifest_path = resolve_path(cfg_dir, manifest_cfg)
        all_records = load_operation_manifest(
            manifest_path,
            validate_paths=bool(data_cfg.get("manifest_validate_paths", True)),
        )
        split_audit = validate_operation_manifest(all_records)
        test_name = str(data_cfg.get("test_split", "test"))
        test_records = [
            record for record in all_records if record["split"] == test_name
        ]
        eval_source = data_cfg.get("manifest_eval_source")
        if eval_source:
            test_records = [
                record
                for record in test_records
                if record["source"] == str(eval_source)
            ]
        override_dir_cfg = data_cfg.get("manifest_points_override_dir")
        if override_dir_cfg:
            override_dir = resolve_path(cfg_dir, override_dir_cfg)
            overridden_records = []
            for original in test_records:
                record = dict(original)
                original_suffix = os.path.splitext(str(record["points_path"]))[1]
                replacement = os.path.join(
                    override_dir, str(record["frame_id"]) + original_suffix
                )
                if not os.path.isfile(replacement):
                    raise FileNotFoundError(
                        f"Missing manifest point override for {record['sample_id']}: "
                        f"{replacement}"
                    )
                record["points_path"] = replacement
                overridden_records.append(record)
            test_records = overridden_records
        feature_cache_cfg = data_cfg.get("feature_cache_path")
        if feature_cache_cfg and not override_dir_cfg:
            feature_cache_path = resolve_path(cfg_dir, feature_cache_cfg)
        elif feature_cache_cfg and override_dir_cfg:
            print(
                "[feature-cache] bypassed because manifest_points_override_dir "
                "changes the point-cloud inputs"
            )
        if not test_records:
            raise RuntimeError(
                f"Operation manifest has no records for test split {test_name!r}"
                + (f" and source {eval_source!r}" if eval_source else "")
            )
        ds = _build_manifest_eval_dataset(
            test_records,
            geom,
            wl_cfg,
            ev_cfg,
            feature_cache_path=feature_cache_path,
        )
        active_collate_fn = collate_manifest
        print(
            f"[split] test frames={len(test_records)} "
            f"operations={sorted({r['operation_id'] for r in test_records})} "
            f"source={eval_source or 'all'} "
            f"annotation_types={sorted({r['annotation_type'] for r in test_records})}"
        )
    else:
        print(
            "[WARNING] No data.operation_manifest_path configured; using the "
            "legacy frame-wise test set. Do not use this mode for revised paper results."
        )
        add_te = resolve_path(cfg_dir, ev_cfg["add_info_testing_path"])
        pts_dir = resolve_path(
            cfg_dir,
            data_cfg.get("points_testing_dir", data_cfg["points_training_dir"]),
        )
        det_json = resolve_path(cfg_dir, ev_cfg.get("fused_det3d_path"))
        gt_map = parse_add_info_strict(add_te)
        frames_sorted = sorted(
            [
                (fid, tsec, ts_raw)
                for fid, (_, tsec, ts_raw) in gt_map.items()
            ],
            key=lambda item: (item[1], item[0]),
        )
        det_boxes = load_det3d_fused_json(det_json) if det_json else {}
        x_min, x_max = geom["chamber_x_range"]
        y_min, y_max = geom["chamber_y_range"]
        ds = EvalFrames(
            frames_sorted, gt_map, pts_dir, det_boxes,
            x_min, x_max, y_min, y_max,
            wl_cfg["wall_band_x_half"], wl_cfg["y_bin"],
            wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
            wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
            rm_labels=wl_cfg.get("extra_remove_3d_labels", []),
            use_point_removal=ev_cfg.get("ablation", {}).get(
                "use_point_removal", True
            ),
        )
        active_collate_fn = collate_eval
    num_workers = int(ev_cfg.get("num_workers", 4))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=False, num_workers=num_workers,
        collate_fn=active_collate_fn, pin_memory=True
    )

    # Build/load model. Torch models load state_dict from .pth; classical ML
    # baselines load the whole fitted object from pickle (.pkl or pickle-form .pth).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_is_sklearn = build_or_load_eval_model(model_kind, sec.get("model", {}), model_path, device)

    # Evaluation inference
    ys_all, yp_all, metadata_all, ts_all = [], [], [], []
    num_eval_frames = 0

    # Total evaluation timing: includes DataLoader feature construction,
    # model inference, result collection, and CPU/GPU synchronization.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_start_time = time.perf_counter()

    grad_context = torch.no_grad() if not model_is_sklearn else torch.inference_mode()
    with grad_context:
        pbar = tqdm(
            total=len(dl),
            desc="Eval",
            ncols=100,
            mininterval=0.5,
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
            leave=False,
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        for batch_data in dl:
            if manifest_mode:
                seq, mask, gt, metadata = batch_data
                batch_tsec = np.asarray(
                    [item["tsec"] for item in metadata], dtype=np.float64
                )
            else:
                seq, mask, gt, fids, batch_tsec, ts_raw = batch_data
                metadata = [
                    {
                        "sample_id": str(fid),
                        "source": "legacy_test",
                        "frame_id": str(fid),
                        "timestamp": str(raw_timestamp),
                        "date": "",
                        "operation_id": -1,
                        "tsec": float(seconds),
                        "split": "legacy_test",
                    }
                    for fid, seconds, raw_timestamp in zip(
                        fids, batch_tsec, ts_raw
                    )
                ]
            if model_is_sklearn:
                pred = _predict_sklearn(model, seq, mask)
            else:
                pred = model(seq.to(device), key_padding_mask=mask.to(device)).detach().cpu().numpy()
            ys_all.append(gt.numpy())
            yp_all.append(np.asarray(pred, dtype=np.float64).reshape(-1))
            metadata_all.extend(metadata)
            ts_all.append(batch_tsec)
            num_eval_frames += len(metadata)
            pbar.update(1)
        pbar.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_elapsed_sec = time.perf_counter() - eval_start_time
    eval_fps = num_eval_frames / eval_elapsed_sec if eval_elapsed_sec > 0 else 0.0
    eval_ms_per_frame = 1000.0 / eval_fps if eval_fps > 0 else 0.0

    print("Eval done.")
    print(
        f"[TIMING] evaluated {num_eval_frames} frames in {eval_elapsed_sec:.4f} s | "
        f"FPS = {eval_fps:.2f} frames/s | {eval_ms_per_frame:.4f} ms/frame"
    )

    ys = np.concatenate(ys_all, 0)
    yp = np.concatenate(yp_all, 0)
    tsec = np.concatenate(ts_all, 0)
    operation_ids = np.asarray(
        [int(item["operation_id"]) for item in metadata_all], dtype=np.int64
    )

    # Metrics (raw)
    met_raw = eval_metrics(ys, yp)  # MAE/RMSE/Bias/Corr

    # (Optional) Kalman smoothing
    use_kf = ev_cfg.get("ablation", {}).get("use_kalman", True)
    if use_kf:
        base_R = float(ev_cfg.get("kalman_obs_noise", 0.10))
        q_pos  = float(ev_cfg.get("kalman", {}).get("q_pos", 1.0e-4))
        q_vel  = float(ev_cfg.get("kalman", {}).get("q_vel", 1.0e-6))
        reset_gap = float(ev_cfg.get("prev_gap_sec", 10.0))
        history_len = int(ev_cfg.get("kalman", {}).get("history_len", 0))
        init_mode   = str(ev_cfg.get("kalman", {}).get("init_mode", "use_obs"))
        default_val = float(ev_cfg.get("kalman", {}).get("default_value", 0.0))
        pos_var0    = float(ev_cfg.get("kalman", {}).get("pos_var0", 0.05))
        vel_var0    = float(ev_cfg.get("kalman", {}).get("vel_var0", 0.01))
        warmup      = int(ev_cfg.get("kalman", {}).get("warmup_frames", 0))

        yp_kf = _kf_by_operation(
            times=tsec,
            observations=yp,
            operation_ids=operation_ids,
            base_R=base_R, q_pos=q_pos, q_vel=q_vel, reset_gap=reset_gap,
            history_len=history_len, init_mode=init_mode, default_value=default_val,
            pos_var0=pos_var0, vel_var0=vel_var0, warmup_frames=warmup
        )
        met_kf = eval_metrics(ys, yp_kf)
    else:
        yp_kf = None
        met_kf = None

    # Output CSV
    import pandas as pd
    df = pd.DataFrame({
        "sample_id": [item["sample_id"] for item in metadata_all],
        "source": [item["source"] for item in metadata_all],
        "fid": [item["frame_id"] for item in metadata_all],
        "operation_id": operation_ids,
        "date": [item["date"] for item in metadata_all],
        "timestamp": [item["timestamp"] for item in metadata_all],
        "tsec": tsec,
        "gt": ys,
        "pred": yp,
        **({"pred_kf": yp_kf} if yp_kf is not None else {})
    })
    csv_path = os.path.join(out_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)

    operation_metrics_path = None
    operation_summary_path = None
    if manifest_mode:
        operation_rows = _operation_metric_rows(
            ys, yp, yp_kf, operation_ids,
            [item["date"] for item in metadata_all],
        )
        operation_metrics_path = os.path.join(
            out_dir, "per_operation_metrics.csv"
        )
        pd.DataFrame(operation_rows).to_csv(operation_metrics_path, index=False)
        operation_summary = {
            "protocol": "operation-wise",
            "test_operations": sorted(set(operation_ids.tolist())),
            "micro_raw": met_raw,
            "micro_kalman": met_kf,
            "macro_across_operations": _macro_operation_summary(operation_rows),
        }
        operation_summary_path = os.path.join(
            out_dir, "operation_summary.json"
        )
        with open(operation_summary_path, "w", encoding="utf-8") as handle:
            json.dump(operation_summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        copied_manifest = os.path.join(out_dir, "split_manifest.csv")
        if os.path.abspath(manifest_path) != os.path.abspath(copied_manifest):
            shutil.copy2(manifest_path, copied_manifest)
        with open(manifest_path, "rb") as handle:
            manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
        audit_path = os.path.join(out_dir, "split_audit.json")
        with open(audit_path, "w", encoding="utf-8") as handle:
            audit_payload = {
                "manifest_path": os.path.abspath(manifest_path),
                "manifest_sha256": manifest_sha256,
                "audit": split_audit,
            }
            if feature_cache_path:
                with open(feature_cache_path, "rb") as cache_handle:
                    audit_payload["feature_cache"] = {
                        "path": os.path.abspath(feature_cache_path),
                        "sha256": hashlib.sha256(cache_handle.read()).hexdigest(),
                    }
            json.dump(
                audit_payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

    # === Save eval_results(test_dataset).txt ===
    def _format_block(title: str, met: dict) -> str:
        def _fmt(v):
            return "None" if v is None else f"{v:.4f}"
        lines = [
            f"===== EVAL ({title}) =====",
            f"count: {met.get('count', 0):.4f}",
            f"MAE: {_fmt(met.get('MAE'))}",
            f"RMSE: {_fmt(met.get('RMSE'))}",
            f"Bias: {_fmt(met.get('Bias'))}",
            f"Corr: {_fmt(met.get('Corr'))}",
            ""
        ]
        return "\n".join(lines).rstrip()

    results_txt_path = os.path.join(out_dir, "eval_results(test_dataset).txt")
    txt_parts = [_format_block("raw", met_raw)]
    if met_kf is not None:
        txt_parts.append(_format_block("kalman", met_kf))

    timing_block = "\n".join([
        "===== TIMING =====",
        f"num_frames: {num_eval_frames}",
        f"elapsed_sec: {eval_elapsed_sec:.4f}",
        f"fps: {eval_fps:.4f}",
        f"ms_per_frame: {eval_ms_per_frame:.4f}",
    ])
    txt_parts.append(timing_block)

    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(txt_parts) + "\n")

    # Console output (print once)
    print("[EVAL] raw:", {k: (None if v is None else round(v, 6)) for k, v in met_raw.items()})
    if met_kf is not None:
        print("[EVAL] kf :", {k: (None if v is None else round(v, 6)) for k, v in met_kf.items()})
    print(f"[done] saved: {csv_path}")
    print(f"[done] saved: {results_txt_path}")
    if operation_metrics_path:
        print(f"[done] saved: {operation_metrics_path}")
    if operation_summary_path:
        print(f"[done] saved: {operation_summary_path}")


if __name__ == "__main__":
    main()
