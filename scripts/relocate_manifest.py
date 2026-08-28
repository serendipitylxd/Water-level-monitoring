#!/usr/bin/env python3
"""Create a machine-local manifest without editing the public split manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "splits/waterlevel_test_only_8000/manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--points-dir", type=Path, required=True)
    parser.add_argument(
        "--annotation",
        type=Path,
        default=REPO_ROOT / "outputs/det3d/fused.det3d.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "splits/waterlevel_test_only_8000/manifest.local.csv"
        ),
    )
    parser.add_argument(
        "--point-extension", choices=(".bin", ".npy"), default=".bin"
    )
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    points_dir = args.points_dir.expanduser().resolve()
    annotation = args.annotation.expanduser().resolve()
    output = args.output.expanduser().resolve()

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows or "frame_id" not in fields:
        raise RuntimeError(f"Invalid manifest: {manifest}")

    for row in rows:
        row["points_path"] = str(
            points_dir / f"{row['frame_id']}{args.point_extension}"
        )
        if row.get("annotation_type", "none") != "none":
            row["annotation_path"] = str(annotation)

    if args.check_files:
        missing = [row["points_path"] for row in rows if not Path(row["points_path"]).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} point clouds; first missing file: {missing[0]}"
            )
        if not annotation.is_file():
            raise FileNotFoundError(f"Missing fused detection file: {annotation}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
