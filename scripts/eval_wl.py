# scripts/eval_wl.py
# -*- coding: utf-8 -*-
import os, sys, argparse, json, inspect, pickle, time
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
    add_te = resolve_path(cfg_dir, ev_cfg["add_info_testing_path"])
    pts_dir = resolve_path(cfg_dir, data_cfg.get("points_testing_dir", data_cfg["points_training_dir"]))
    det_json = resolve_path(cfg_dir, ev_cfg.get("fused_det3d_path"))
    model_path = resolve_path(cfg_dir, ev_cfg.get("model_path"))
    out_dir = resolve_path(cfg_dir, ev_cfg.get("out_dir", os.path.join("outputs", sec["out_dir"], "eval")))
    ensure_dir(out_dir)

    # Read strict timestamps & sort
    gt_map = parse_add_info_strict(add_te)  # {fid: (wl, tsec, ts_raw)}
    frames_sorted = sorted([(fid, tsec, ts_raw) for fid, (_, tsec, ts_raw) in gt_map.items()],
                           key=lambda x: (x[1], x[0]))

    # Fused det3d boxes
    det_boxes = load_det3d_fused_json(det_json) if det_json else {}

    # Dataset / DataLoader
    x_min, x_max = geom["chamber_x_range"]
    y_min, y_max = geom["chamber_y_range"]
    batch = int(ev_cfg.get("batch", sec.get("train", {}).get("batch", 60)))
    ds = EvalFrames(
        frames_sorted, gt_map, pts_dir, det_boxes,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"],
        wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        rm_labels=wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=ev_cfg.get("ablation", {}).get("use_point_removal", True),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, collate_fn=collate_eval, pin_memory=True)

    # Build/load model. Torch models load state_dict from .pth; classical ML
    # baselines load the whole fitted object from pickle (.pkl or pickle-form .pth).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_is_sklearn = build_or_load_eval_model(model_kind, sec.get("model", {}), model_path, device)

    # Evaluation inference
    ys_all, yp_all, fids_all, ts_all = [], [], [], []
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
        for seq, mask, gt, fids, tsec, _ in dl:
            if model_is_sklearn:
                pred = _predict_sklearn(model, seq, mask)
            else:
                pred = model(seq.to(device), key_padding_mask=mask.to(device)).detach().cpu().numpy()
            ys_all.append(gt.numpy())
            yp_all.append(np.asarray(pred, dtype=np.float64).reshape(-1))
            fids_all.extend(fids)
            ts_all.append(tsec)
            num_eval_frames += len(fids)
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

        order = np.argsort(tsec, kind="stable")
        yp_kf = np.array(kf_online(
            times=tsec[order].tolist(),
            obs=yp[order].tolist(),
            base_R=base_R, q_pos=q_pos, q_vel=q_vel, reset_gap=reset_gap,
            history_len=history_len, init_mode=init_mode, default_value=default_val,
            pos_var0=pos_var0, vel_var0=vel_var0, warmup_frames=warmup
        ), dtype=np.float64)
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        yp_kf = yp_kf[inv]
        met_kf = eval_metrics(ys, yp_kf)
    else:
        yp_kf = None
        met_kf = None

    # Output CSV
    import pandas as pd
    df = pd.DataFrame({
        "fid": fids_all,
        "tsec": tsec,
        "gt": ys,
        "pred": yp,
        **({"pred_kf": yp_kf} if yp_kf is not None else {})
    })
    csv_path = os.path.join(out_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)

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


if __name__ == "__main__":
    main()
