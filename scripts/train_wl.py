# scripts/train_wl.py
# -*- coding: utf-8 -*-
import os, sys, re, argparse
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
from utils.data import parse_add_info, WLFrames, collate_fn
from utils.metrics import eval_metrics
from utils.trainer import build_model  

# ---------------------------
# Training-only: section detection / dataset building
# ---------------------------
def detect_section(cfg: dict) -> Tuple[str, str]:
    if "wl_transformer" in cfg: return "wl_transformer", "transformer"
    if "wl_retnet"      in cfg: return "wl_retnet", "retnet"
    if "wl_mamba"       in cfg: return "wl_mamba", "mamba"
    if "wl_rwkv"        in cfg: return "wl_rwkv", "rwkv"
    if "wl_hyena"       in cfg: return "wl_hyena", "hyena"
    if "wl_mega"        in cfg: return "wl_mega", "mega"
    if "wl_hgrn"        in cfg: return "wl_hgrn", "hgrn"
    raise RuntimeError("Config must contain one of: wl_transformer / wl_retnet / wl_mamba / wl_rwkv / wl_hyena / wl_mega / wl_hgrn")

def _numeric_key(fid: str):
    m = re.search(r"\d+", fid)
    return (int(m.group()) if m else 0, fid)

def build_datasets(cfg_dir: str, sub: dict, geom: dict, wl_cfg: dict, data_cfg: dict):
    """First 6000 for training, last 2000 for validation (adaptive if insufficient), same as original logic."""
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
    return ds_tr, ds_va

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
    out_root = cfg.get("output", {}).get("root", "./outputs")
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
    ds_tr, ds_va = build_datasets(cfg_dir, section, geom, wl_cfg, data_cfg)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=4, collate_fn=collate_fn, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True) if ds_va is not None else None

    # Model 
    model = build_model(kind, section["model"])
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

