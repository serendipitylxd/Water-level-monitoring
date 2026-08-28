#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute deterministic wall-profile features for manifest experiments.

The cache changes only I/O cost.  It stores the exact float32 sequence and
boolean padding mask returned by ``ManifestWLFrames`` before any model sees the
sample.  Runtime loading verifies the preprocessing signature, sample IDs, and
point/annotation mapping before using cached values.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data import (  # noqa: E402
    ManifestWLFrames,
    collate_manifest,
    feature_spec_sha256,
    load_operation_manifest,
    manifest_record_source_key,
)
from utils.io import load_yaml, resolve_path  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset(records, geom, wl_cfg):
    x_min, x_max = geom["chamber_x_range"]
    y_min, y_max = geom["chamber_y_range"]
    return ManifestWLFrames(
        records,
        x_min,
        x_max,
        y_min,
        y_max,
        wl_cfg["wall_band_x_half"],
        wl_cfg["y_bin"],
        wl_cfg["quantile_low"],
        wl_cfg["qc_min_pts"],
        wl_cfg["inflate_xy"],
        wl_cfg["inflate_z"],
        remove_labels=wl_cfg.get("extra_remove_3d_labels", []),
        use_point_removal=wl_cfg.get("ablation", {}).get(
            "use_point_removal", True
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.cfg).expanduser().resolve()
    cfg = load_yaml(str(cfg_path))
    cfg_dir = str(cfg_path.parent)
    data_cfg = cfg["data"]

    manifest_cfg = args.manifest or data_cfg.get("operation_manifest_path")
    if not manifest_cfg:
        raise RuntimeError("Config has no data.operation_manifest_path")
    manifest_path = Path(resolve_path(cfg_dir, manifest_cfg)).resolve()

    output_cfg = args.output or data_cfg.get("feature_cache_path")
    if not output_cfg:
        output_cfg = (
            "outputs_operation_split/test_only_8000/feature_cache/"
            "waterlevel_test_only_default.npz"
        )
    output_path = Path(resolve_path(cfg_dir, output_cfg)).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"Feature-cache output must end in .npz: {output_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(
            f"Feature cache already exists: {output_path}; pass --force to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_operation_manifest(str(manifest_path), validate_paths=True)
    dataset = build_dataset(records, cfg["geometry"], cfg["waterlevel"])
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_manifest,
        pin_memory=False,
    )

    sequence_batches = []
    mask_batches = []
    observed_sample_ids = []
    for sequences, masks, _, metadata in tqdm(
        loader, total=len(loader), desc="Build feature cache"
    ):
        sequence_batches.append(sequences.numpy().astype(np.float32, copy=False))
        mask_batches.append(masks.numpy().astype(bool, copy=False))
        observed_sample_ids.extend(str(item["sample_id"]) for item in metadata)

    sequences = np.concatenate(sequence_batches, axis=0)
    masks = np.concatenate(mask_batches, axis=0)
    expected_sample_ids = [str(record["sample_id"]) for record in records]
    if observed_sample_ids != expected_sample_ids:
        raise RuntimeError("DataLoader order changed while building the feature cache")
    if sequences.shape[0] != len(records) or masks.shape != sequences.shape[:2]:
        raise RuntimeError(
            f"Invalid collected shapes: sequences={sequences.shape}, masks={masks.shape}"
        )

    sample_ids = np.asarray(expected_sample_ids, dtype=np.str_)
    source_keys = np.asarray(
        [manifest_record_source_key(record) for record in records], dtype=np.str_
    )
    spec = dataset.feature_spec
    arrays_digest = hashlib.sha256()
    arrays_digest.update(np.ascontiguousarray(sequences).tobytes())
    arrays_digest.update(np.ascontiguousarray(masks).tobytes())
    metadata = {
        "format_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(str(manifest_path)),
        "feature_spec": spec,
        "feature_spec_sha256": feature_spec_sha256(spec),
        "sample_count": len(records),
        "sequence_shape": list(sequences.shape[1:]),
        "feature_arrays_sha256": arrays_digest.hexdigest(),
    }
    metadata_json = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )

    temporary_path = output_path.with_name(output_path.stem + ".building.npz")
    np.savez_compressed(
        str(temporary_path),
        sample_ids=sample_ids,
        source_keys=source_keys,
        sequences=sequences,
        masks=masks,
        metadata_json=np.asarray(metadata_json),
    )
    os.replace(str(temporary_path), str(output_path))

    metadata["cache_path"] = str(output_path)
    metadata["cache_sha256"] = sha256_file(str(output_path))
    sidecar_path = Path(str(output_path) + ".metadata.json")
    temporary_sidecar = Path(str(sidecar_path) + ".building")
    with temporary_sidecar.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temporary_sidecar), str(sidecar_path))

    print(f"[done] cache: {output_path}")
    print(f"[done] metadata: {sidecar_path}")
    print(
        f"[done] samples={len(records)} sequence_shape={tuple(sequences.shape[1:])} "
        f"sha256={metadata['cache_sha256']}"
    )


if __name__ == "__main__":
    main()
