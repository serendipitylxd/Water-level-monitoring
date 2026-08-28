#!/usr/bin/env python3
"""Fail when public text files contain machine-specific absolute paths."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cff", ".csv", ".json", ".md", ".py", ".sh", ".tex", ".txt", ".yaml", ".yml"
}
FORBIDDEN = (
    "/" + "home" + "/",
    "/" + "Users" + "/",
    "C:" + "\\",
    "luxiao" + "dong",
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / value for value in result.stdout.splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.paths] if args.paths else tracked_files()
    failures = []
    for path in paths:
        if path == Path(__file__).resolve() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in FORBIDDEN):
                failures.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    if failures:
        raise SystemExit("Machine-specific paths found:\n" + "\n".join(failures))
    print(f"Public-path audit passed for {len(paths)} tracked files.")


if __name__ == "__main__":
    main()
