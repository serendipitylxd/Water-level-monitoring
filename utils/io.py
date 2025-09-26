# utils/io.py
# -*- coding: utf-8 -*-
import os, argparse, random
from typing import Optional
import numpy as np
try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. pip install pyyaml") from e
import torch

def load_yaml(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def resolve_path(cfg_dir: str, p: Optional[str]) -> Optional[str]:
    """
    Path resolution priority:
    1) Absolute path
    2) Relative to the config.yaml directory
    3) Relative to the repository root (parent of config directory)
    4) Relative to the current working directory (CWD)
    """
    if p is None:
        return None
    p = os.path.expanduser(str(p))
    if os.path.isabs(p):
        return p
    p1 = os.path.join(cfg_dir, p)
    if os.path.exists(p1):
        return p1
    repo_root = os.path.abspath(os.path.join(cfg_dir, os.pardir))
    p2 = os.path.join(repo_root, p)
    if os.path.exists(p2):
        return p2
    return os.path.abspath(p)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to config yaml")
    return ap.parse_args()

