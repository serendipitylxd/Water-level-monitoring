# scripts/train_wl.py
# -*- coding: utf-8 -*-
import os, sys, re, argparse, inspect, pickle, json, shutil, hashlib
from collections import Counter
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repository root takes precedence
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Reuse common utilities
from utils.io import load_yaml, ensure_dir, resolve_path, seed_everything
from utils.data import (
    parse_add_info,
    WLFrames,
    collate_fn,
    load_operation_manifest,
    validate_operation_manifest,
    ManifestWLFrames,
    collate_manifest,
)
from utils.metrics import eval_metrics
from utils.trainer import build_model as build_existing_model
from utils import models as wl_models

# ---------------------------
# Training-only: section detection / dataset building
# ---------------------------
def detect_section(cfg: dict) -> Tuple[str, str]:
    """Detect the active water-level configuration section.

    Backward compatible:
      - old configs: wl_transformer / wl_retnet / ...
      - new baseline configs: wl_linear_regression / wl_mlp / ...

    Note: the baseline YAML files generated earlier still use the top-level
    section `wl_transformer`; in that case the actual model is selected by
    `section["model"]["name"]`.
    """
    if "wl_transformer" in cfg: return "wl_transformer", "transformer"
    if "wl_retnet"      in cfg: return "wl_retnet", "retnet"
    if "wl_mamba"       in cfg: return "wl_mamba", "mamba"
    if "wl_rwkv"        in cfg: return "wl_rwkv", "rwkv"
    if "wl_hyena"       in cfg: return "wl_hyena", "hyena"
    if "wl_mega"        in cfg: return "wl_mega", "mega"
    if "wl_hgrn"        in cfg: return "wl_hgrn", "hgrn"

    # Optional explicit sections for the reviewer-requested baselines.
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

def _numeric_key(fid: str):
    m = re.search(r"\d+", fid)
    return (int(m.group()) if m else 0, fid)


# ---------------------------
# Model building: support both original deep models and 7 reviewer baselines
# ---------------------------
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
# Keys used by YAML only; they should not be passed into model constructors.
_MODEL_CFG_CONTROL_KEYS = {"name", "flatten", "activation", "pooling", "standardize"}


def _filter_model_kwargs(cls, model_cfg: dict) -> dict:
    """Keep constructor-compatible kwargs and discard YAML-only control keys."""
    raw = {k: v for k, v in model_cfg.items() if k not in _MODEL_CFG_CONTROL_KEYS}
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return raw
    allowed = {k for k in params.keys() if k != "self"}
    return {k: v for k, v in raw.items() if k in allowed}


def _maybe_standardize_sklearn(model, standardize: bool):
    """Optionally wrap sklearn estimator with StandardScaler.

    This is mainly useful for SVR. RandomForest/XGBoost do not need it, but the
    wrapper is harmless if explicitly requested.
    """
    if not standardize:
        return model
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("standardize=true requires scikit-learn") from e
    if hasattr(model, "estimator"):
        model.estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model.estimator),
        ])
    return model


def build_wl_model(kind: str, model_cfg: dict):
    """Build a water-level model from YAML.

    Priority:
      1) If `model.name` is provided, instantiate that class from utils.models.
      2) Otherwise use the original utils.trainer.build_model(kind, model_cfg).

    Supported new names:
      LinearRegressionWL, RidgeRegressionWL, MLPWL, CNN1DWL,
      SVRWL, RandomForestWL, XGBoostWL.
    """
    model_cfg = dict(model_cfg or {})
    name = model_cfg.get("name") or _BASELINE_KIND_TO_NAME.get(kind)

    if name:
        if not hasattr(wl_models, name):
            raise RuntimeError(
                f"Model class '{name}' is not found in utils.models. "
                f"Please make sure models.py contains class {name}."
            )
        cls = getattr(wl_models, name)
        kwargs = _filter_model_kwargs(cls, model_cfg)
        model = cls(**kwargs)
        if name in _SKLEARN_MODEL_NAMES:
            model = _maybe_standardize_sklearn(
                model, bool(model_cfg.get("standardize", False))
            )
        return model

    # Old configs without model.name still use the original builder.
    return build_existing_model(kind, model_cfg)


def is_sklearn_like_model(model) -> bool:
    """True for SVRWL / RandomForestWL / XGBoostWL wrappers."""
    return (not isinstance(model, nn.Module)) and hasattr(model, "fit") and hasattr(model, "predict")


def _collect_loader_tensors(loader, desc="Collect"):
    """Collect a DataLoader into tensors for sklearn-style one-shot fitting."""
    seqs, masks, gts = [], [], []
    pbar = tqdm(total=len(loader), desc=desc, ncols=100, dynamic_ncols=True,
                disable=not sys.stdout.isatty(), mininterval=0.5)
    for seq, mask, gt, _ in loader:
        seqs.append(seq.cpu())
        masks.append(mask.cpu())
        gts.append(gt.cpu())
        pbar.update(1)
    pbar.close()
    if not seqs:
        raise RuntimeError("Empty dataloader; no samples available for training/evaluation.")
    return torch.cat(seqs, 0), torch.cat(masks, 0), torch.cat(gts, 0)


def _fmt_metric(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "None"


def _write_metrics(out_dir: str, met: dict, filename: str = "eval_results(val_dataset).txt"):
    cnt = float(met.get("count", 0) or 0)
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write("\n".join([
            "===== Test Metrics =====",
            f"count: {cnt:.0f}",
            f"MAE: {_fmt_metric(met.get('MAE'))}",
            f"RMSE: {_fmt_metric(met.get('RMSE'))}",
            f"Bias: {_fmt_metric(met.get('Bias'))}",
            f"Corr: {_fmt_metric(met.get('Corr'))}",
            ""
        ]))


def train_sklearn_main(model, dl_tr, dl_va, out_dir: str):
    """Training branch for SVRWL / RandomForestWL / XGBoostWL."""
    seq_tr, mask_tr, gt_tr = _collect_loader_tensors(dl_tr, desc="Collect train")
    print(f"[sklearn] fitting {model.__class__.__name__} on {len(gt_tr)} samples ...")
    model.fit(seq_tr, gt_tr, key_padding_mask=mask_tr)

    # Save as pickle. We also create model.pth as a pickle file for compatibility
    # with older YAML paths; evaluation code should load classical baselines with
    # pickle/joblib rather than torch.load(state_dict).
    pkl_path = os.path.join(out_dir, "model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    legacy_path = os.path.join(out_dir, "model.pth")
    with open(legacy_path, "wb") as f:
        pickle.dump(model, f)

    if dl_va is not None:
        seq_va, mask_va, gt_va = _collect_loader_tensors(dl_va, desc="Collect val")
        yps = model.predict(seq_va, key_padding_mask=mask_va)
        ys = gt_va.numpy()
        met = eval_metrics(ys, yps)
        _write_metrics(out_dir, met)
        cnt = float(met.get("count", 0) or 0)
        print(
            f"[eval] count={cnt:.0f}  "
            f"MAE={_fmt_metric(met.get('MAE'))}  "
            f"RMSE={_fmt_metric(met.get('RMSE'))}  "
            f"Bias={_fmt_metric(met.get('Bias'))}  "
            f"Corr={_fmt_metric(met.get('Corr'))}"
        )

    print(f"[done] saved sklearn model: {pkl_path}")
    print(f"[done] legacy copy: {legacy_path}")

def _build_manifest_dataset(
    records, geom: dict, wl_cfg: dict, feature_cache_path: str = None
):
    x_min, x_max = geom["chamber_x_range"]
    y_min, y_max = geom["chamber_y_range"]
    return ManifestWLFrames(
        records,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"],
        wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=wl_cfg.get("ablation", {}).get(
            "use_point_removal", True
        ),
        feature_cache_path=feature_cache_path,
    )


def _manifest_provenance(records, manifest_path: str, audit: dict) -> dict:
    split_details = {}
    for split in ("train", "val", "test"):
        selected = [record for record in records if record["split"] == split]
        split_details[split] = {
            "num_frames": len(selected),
            "operations": sorted({int(record["operation_id"]) for record in selected}),
            "dates": sorted({str(record["date"]) for record in selected}),
            "sources": dict(sorted(Counter(str(record["source"]) for record in selected).items())),
            "annotation_types": dict(
                sorted(Counter(str(record["annotation_type"]) for record in selected).items())
            ),
        }
    with open(manifest_path, "rb") as handle:
        manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        "protocol": "operation-wise",
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": manifest_sha256,
        "audit": audit,
        "splits": split_details,
    }


def _save_manifest_provenance(out_dir: str, context: dict) -> None:
    manifest_path = context["manifest_path"]
    copied_manifest = os.path.join(out_dir, "split_manifest.csv")
    if os.path.abspath(manifest_path) != os.path.abspath(copied_manifest):
        shutil.copy2(manifest_path, copied_manifest)
    summary_path = os.path.join(out_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(context["provenance"], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[split] manifest={manifest_path}")
    for split, item in context["provenance"]["splits"].items():
        print(
            f"[split] {split}: frames={item['num_frames']} "
            f"operations={item['operations']} dates={item['dates']}"
        )
    annotation_types = sorted(
        {
            annotation_type
            for item in context["provenance"]["splits"].values()
            for annotation_type in item["annotation_types"]
        }
    )
    if len(annotation_types) > 1:
        print(
            "[WARNING] Manifest mixes annotation backends "
            f"{annotation_types}. Before final paper runs, generate frozen "
            "detector outputs for legacy_train and rebuild the manifest with "
            "--train-det3d-json."
        )
    print(f"[split] saved provenance: {summary_path}")


def build_datasets(cfg_dir: str, sub: dict, geom: dict, wl_cfg: dict, data_cfg: dict):
    """Build manifest-based operation splits, with a legacy fallback."""
    manifest_cfg = data_cfg.get("operation_manifest_path")
    if manifest_cfg:
        manifest_path = resolve_path(cfg_dir, manifest_cfg)
        records = load_operation_manifest(
            manifest_path,
            validate_paths=bool(data_cfg.get("manifest_validate_paths", True)),
        )
        audit = validate_operation_manifest(records)
        train_name = str(data_cfg.get("train_split", "train"))
        val_name = str(data_cfg.get("val_split", "val"))
        train_records = [record for record in records if record["split"] == train_name]
        val_records = [record for record in records if record["split"] == val_name]
        if not train_records:
            raise RuntimeError(
                f"Operation manifest has no records for train split {train_name!r}"
            )
        feature_cache_cfg = data_cfg.get("feature_cache_path")
        feature_cache_path = (
            resolve_path(cfg_dir, feature_cache_cfg) if feature_cache_cfg else None
        )
        ds_tr = _build_manifest_dataset(
            train_records, geom, wl_cfg, feature_cache_path=feature_cache_path
        )
        ds_va = (
            _build_manifest_dataset(
                val_records, geom, wl_cfg, feature_cache_path=feature_cache_path
            )
            if val_records
            else None
        )
        provenance = _manifest_provenance(records, manifest_path, audit)
        if feature_cache_path:
            with open(feature_cache_path, "rb") as handle:
                cache_sha256 = hashlib.sha256(handle.read()).hexdigest()
            provenance["feature_cache"] = {
                "path": os.path.abspath(feature_cache_path),
                "sha256": cache_sha256,
            }
        context = {
            "manifest_path": manifest_path,
            "provenance": provenance,
        }
        return ds_tr, ds_va, collate_manifest, context

    # Legacy frame-wise path retained only for old-result reproducibility.
    pts_dir = resolve_path(cfg_dir, data_cfg["points_training_dir"])
    lbl_dir = resolve_path(cfg_dir, data_cfg.get("labels_training_dir"))
    add_tr  = resolve_path(cfg_dir, sub["add_info_training_path"])

    gt_map = parse_add_info(add_tr)
    fids = sorted(list(gt_map.keys()), key=_numeric_key)

    n = len(fids)
    n_train = min(6000, n)
    n_val   = min(2000, max(0, n - n_train))
    fids_train = fids[:n_train]
    fids_val   = fids[-n_val:] if n_val > 0 else []

    x_min, x_max = geom["chamber_x_range"]
    y_min, y_max = geom["chamber_y_range"]
    ds_tr = WLFrames(
        fids_train, gt_map, pts_dir, lbl_dir,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"], wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=wl_cfg.get("ablation", {}).get("use_point_removal", True)
    )
    ds_va = WLFrames(
        fids_val, gt_map, pts_dir, lbl_dir,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"], wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=wl_cfg.get("ablation", {}).get("use_point_removal", True)
    ) if n_val > 0 else None
    return ds_tr, ds_va, collate_fn, None

# ---------------------------
# Main training flow 
# ---------------------------
def train_main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

    # Section detection (training only)
    section_key, kind = detect_section(cfg)
    section = cfg[section_key]

    data_cfg = cfg["data"]
    wl_cfg   = cfg["waterlevel"]
    geom     = cfg["geometry"]

    # Output path
    out_root = resolve_path(
        cfg_dir, cfg.get("output", {}).get("root", "./outputs")
    )
    out_dir  = os.path.join(out_root, section["out_dir"])
    ensure_dir(out_dir)

    # Hyperparameters
    tr_cfg = section.get("train", {})
    epochs = int(tr_cfg.get("epochs", 40))
    batch  = int(tr_cfg.get("batch", 60))
    lr     = float(tr_cfg.get("lr", 1e-3))
    wd     = float(tr_cfg.get("wd", 0.0))
    seed   = int(tr_cfg.get("seed", 42))
    seed_everything(seed)

    # DataLoaders
    ds_tr, ds_va, active_collate_fn, split_context = build_datasets(
        cfg_dir, section, geom, wl_cfg, data_cfg
    )
    if split_context is not None:
        _save_manifest_provenance(out_dir, split_context)
    else:
        print(
            "[WARNING] No data.operation_manifest_path configured; using the "
            "legacy frame-wise split. Do not use this mode for revised paper results."
        )
    num_workers_train = int(tr_cfg.get("num_workers", 4))
    num_workers_val = int(tr_cfg.get("val_num_workers", 2))
    dl_tr = DataLoader(
        ds_tr, batch_size=batch, shuffle=True,
        num_workers=num_workers_train, collate_fn=active_collate_fn, pin_memory=True
    )
    dl_va = DataLoader(
        ds_va, batch_size=batch, shuffle=False,
        num_workers=num_workers_val, collate_fn=active_collate_fn, pin_memory=True
    ) if ds_va is not None else None

    # Model
    model = build_wl_model(kind, section["model"])

    # Classical ML baselines are trained with fit()/predict(), not with torch optimizer.
    if is_sklearn_like_model(model):
        train_sklearn_main(model, dl_tr, dl_va, out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer / scheduler
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    ckpt_path = os.path.join(out_dir, "model.pth")
    best_rmse = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tot_loss = 0.0
        pbar = tqdm(total=len(dl_tr), desc=f"Train {ep}/{epochs}", ncols=100, dynamic_ncols=True,
                    disable=not sys.stdout.isatty(), mininterval=0.5)
        for seq, mask, gt, _ in dl_tr:
            seq = seq.to(device); mask = mask.to(device); gt = gt.to(device)
            pred = model(seq, key_padding_mask=mask)
            loss = loss_fn(pred, gt)
            # RidgeRegressionWL can provide an explicit L2 term. If wd is already
            # used in Adam, set ridge_alpha=0 or wd=0 to avoid double regularization.
            if hasattr(model, "regularization_loss"):
                loss = loss + model.regularization_loss()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot_loss += float(loss.item())
            pbar.update(1)
        avg_loss = tot_loss / max(1, len(dl_tr))
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        pbar.close()

        # Validation
        if dl_va is not None:
            model.eval()
            ys, yps = [], []
            with torch.no_grad():
                for seq, mask, gt, _ in dl_va:
                    pred = model(seq.to(device), key_padding_mask=mask.to(device))
                    ys.append(gt.numpy()); yps.append(pred.cpu().numpy())

            # Validation set may be empty; handle robustly
            met = {"count": 0, "MAE": None, "RMSE": None, "Bias": None, "Corr": None}
            if ys and yps:
                ys  = np.concatenate(ys, 0)
                yps = np.concatenate(yps, 0)
                met = eval_metrics(ys, yps)

            mae  = met.get("MAE", None)
            rmse = met.get("RMSE", None)
            bias = met.get("Bias", None)
            corr = met.get("Corr", None)
            cnt  = float(met.get("count", 0))

            # —— Save policy: always save on first epoch; thereafter only when RMSE improves ——
            def _is_num(x):
                try:
                    return np.isfinite(float(x))
                except Exception:
                    return False
            score = float(rmse) if _is_num(rmse) else float("inf")
            should_save = (not os.path.exists(ckpt_path)) or (score < best_rmse)
            if should_save:
                if np.isfinite(score):
                    best_rmse = score
                torch.save(model.state_dict(), ckpt_path)

            # Write val metrics (safe formatting)
            def _fmt(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "None"
            with open(os.path.join(out_dir, "eval_results(val_dataset).txt"), "w", encoding="utf-8") as f:
                f.write("\n".join([
                    "===== Test Metrics =====",
                    f"count: {cnt:.0f}",
                    f"MAE: {_fmt(mae)}",
                    f"RMSE: {_fmt(rmse)}",
                    f"Bias: {_fmt(bias)}",
                    f"Corr: {_fmt(corr)}",
                    ""
                ]))

            # Safe console print (won't crash on None)
            print(f"[eval] count={cnt:.0f}  MAE={_fmt(mae)}  RMSE={_fmt(rmse)}  Bias={_fmt(bias)}  Corr={_fmt(corr)}")

            # Scheduler monitoring: if RMSE unavailable, fall back to avg_loss to avoid passing inf/None
            rmse_for_sched = score if np.isfinite(score) else avg_loss
            sched.step(rmse_for_sched)
        else:
            # No validation set: overwrite-save every epoch
            torch.save(model.state_dict(), ckpt_path)


    print(f"[done] saved: {ckpt_path}")

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to configs/wl_xxx.yaml")
    args = ap.parse_args()
    train_main(args.cfg)

if __name__ == "__main__":
    main()
