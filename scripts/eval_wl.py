# scripts/eval_wl.py
# -*- coding: utf-8 -*-
import os, sys, argparse, json
import numpy as np
import torch
from tqdm import tqdm

# Ensure repository root is at the front to avoid name clashes with third-party packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.io import load_yaml, resolve_path, ensure_dir
from utils.trainer import build_model                   
from utils.metrics import eval_metrics                  
from utils.eval_utils import (
    parse_add_info_strict, load_det3d_fused_json, EvalFrames, collate_eval, kf_online
)  

def detect_section(cfg: dict):
    if "wl_transformer" in cfg: return "wl_transformer", "transformer"
    if "wl_retnet"      in cfg: return "wl_retnet", "retnet"
    if "wl_mamba"       in cfg: return "wl_mamba", "mamba"
    if "wl_rwkv"        in cfg: return "wl_rwkv", "rwkv"
    if "wl_hyena"       in cfg: return "wl_hyena", "hyena"
    if "wl_mega"        in cfg: return "wl_mega", "mega"
    if "wl_hgrn"        in cfg: return "wl_hgrn", "hgrn"       
    raise RuntimeError("Config must contain one of: wl_transformer / wl_retnet / wl_mamba / wl_rwkv / wl_hyena / wl_mega/ wl_hgrn")


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
    ds = EvalFrames(
        frames_sorted, gt_map, pts_dir, det_boxes,
        x_min, x_max, y_min, y_max,
        wl_cfg["wall_band_x_half"], wl_cfg["y_bin"],
        wl_cfg["quantile_low"], wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"], wl_cfg["inflate_z"],
        rm_labels=wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=ev_cfg.get("ablation", {}).get("use_point_removal", True),
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=60, shuffle=False, num_workers=4, collate_fn=collate_eval, pin_memory=True)

    # Build model (reuse training-side build_model); hyperparameters: prefer section.model from YAML
    model = build_model(model_kind, sec["model"])    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load weights
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

    # Evaluation inference
    ys_all, yp_all, fids_all, ts_all = [], [], [], []
    with torch.no_grad():
        # Use a single progress bar; force writing to stdout (default is stderr, which may clash and break line wrapping)
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
            pred = model(seq.to(device), key_padding_mask=mask.to(device)).cpu().numpy()
            ys_all.append(gt.numpy()); yp_all.append(pred)
            fids_all.extend(fids); ts_all.append(tsec)
            pbar.update(1)
        pbar.close()
        # After closing the progress bar, print a completion line to avoid triggering tqdm redraw
        print("Eval done.")

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

        # Sort times using strict timestamps (seconds)
        order = np.argsort(tsec, kind="stable")
        yp_kf = np.array(kf_online(
            times=tsec[order].tolist(),
            obs=yp[order].tolist(),
            base_R=base_R, q_pos=q_pos, q_vel=q_vel, reset_gap=reset_gap,
            history_len=history_len, init_mode=init_mode, default_value=default_val,
            pos_var0=pos_var0, vel_var0=vel_var0, warmup_frames=warmup
        ), dtype=np.float64)
        # Restore original order
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
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

