# utils/trainer.py
# -*- coding: utf-8 -*-
from . import models as M

def build_model(kind: str, model_cfg: dict):
    """
    A shared model factory for training and evaluation.
    kind in {"transformer","retnet","mamba","rwkv","hyena","mega","hgrn"}。
    """
    if kind == "transformer":
        return M.TransformerWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("layers", 2)),
            ffn=int(model_cfg.get("ffn", 128)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if kind == "retnet":
        return M.RetNetWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            layers=int(model_cfg.get("layers", 4)),
            mlp_ratio=int(model_cfg.get("mlp_ratio", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            conv_kernel=int(model_cfg.get("conv_kernel", 5)),
        )
    if kind == "mamba":
        return M.MambaWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            layers=int(model_cfg.get("layers", 4)),
            d_state=int(model_cfg.get("d_state", 16)),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if kind == "rwkv":
        return M.RWKVWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            layers=int(model_cfg.get("layers", 4)),
            n_head=int(model_cfg.get("n_head", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if kind == "hyena":
        return M.HyenaWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            layers=int(model_cfg.get("layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if kind == "mega":
        return M.MEGAWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("layers", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            ffn=int(model_cfg.get("ffn", 256)),
        )
    if kind == "hgrn":
        return M.HGRNWL(
            in_dim=2,
            d_model=int(model_cfg.get("d_model", 64)),
            layers=int(model_cfg.get("layers", 4)),
            mlp_ratio=int(model_cfg.get("mlp_ratio", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            conv_kernel=int(model_cfg.get("conv_kernel", 5)),
        )
    raise ValueError(f"Unknown model kind: {kind}")

