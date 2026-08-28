#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all primary water-level models under the fixed operation-wise split."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "configs/operation_split/wl_linear_regression.yaml",
    "configs/operation_split/wl_ridge_regression.yaml",
    "configs/operation_split/wl_svr.yaml",
    "configs/operation_split/wl_random_forest.yaml",
    "configs/operation_split/wl_mlp.yaml",
    "configs/operation_split/wl_1dcnn.yaml",
    "configs/operation_split/wl_transformer.yaml",
    "configs/operation_split/wl_retnet.yaml",
    "configs/operation_split/wl_mamba.yaml",
    "configs/operation_split/wl_rwkv.yaml",
    "configs/operation_split/wl_hyena.yaml",
    "configs/operation_split/wl_mega.yaml",
    "configs/operation_split/wl_hgrn.yaml",
]


def resolve_config(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def result_path_for_config(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    eval_dir = Path(cfg.get("eval", {}).get("out_dir", ""))
    if not eval_dir.is_absolute():
        eval_dir = REPO_ROOT / eval_dir
    return eval_dir.resolve() / "per_operation_metrics.csv"


def run_command(command, log_handle):
    return subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    ).returncode


def save_state(path, state):
    temporary = Path(str(path) + ".building")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs_operation_split"
            / "test_only_8000"
            / "fixed"
            / "benchmark_logs"
        ),
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=(
            REPO_ROOT / "outputs_operation_split" / "test_only_8000" / "fixed"
        ),
        help="Root containing model/eval result directories to summarize.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Protocol manifest passed to the fixed-split result auditor.",
    )
    parser.add_argument("--expected-models", type=int, default=13)
    args = parser.parse_args()

    log_root = args.log_root.expanduser().resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = log_root / "run_state.json"
    state = {
        "protocol": "waterlevel-test-only-8000-operation-split",
        "started_at": datetime.now().astimezone().isoformat(),
        "models": [],
    }
    failures = []

    for config_text in args.configs:
        config_path = resolve_config(config_text)
        result_path = result_path_for_config(config_path)
        model_state = {
            "config": str(config_path),
            "result": str(result_path),
            "status": "pending",
        }
        state["models"].append(model_state)
        save_state(state_path, state)

        if args.skip_existing and result_path.is_file():
            model_state["status"] = "skipped_existing"
            print(f"[skip] {config_path.stem}: {result_path}", flush=True)
            save_state(state_path, state)
            continue

        log_path = log_root / f"{config_path.stem}.log"
        model_state["log"] = str(log_path)
        model_state["status"] = "running"
        model_state["started_at"] = datetime.now().astimezone().isoformat()
        save_state(state_path, state)
        print(f"[run] {config_path.stem}; log={log_path}", flush=True)

        commands = [
            [sys.executable, str(REPO_ROOT / "scripts" / "train_wl.py"), "--cfg", str(config_path)],
            [sys.executable, str(REPO_ROOT / "scripts" / "eval_wl.py"), "--cfg", str(config_path)],
        ]
        return_code = 0
        with log_path.open("w", encoding="utf-8") as log_handle:
            for command in commands:
                log_handle.write("[cmd] " + " ".join(command) + "\n")
                log_handle.flush()
                return_code = run_command(command, log_handle)
                if return_code != 0:
                    break

        model_state["finished_at"] = datetime.now().astimezone().isoformat()
        model_state["return_code"] = return_code
        if return_code == 0 and result_path.is_file():
            model_state["status"] = "completed"
            print(f"[done] {config_path.stem}", flush=True)
        else:
            model_state["status"] = "failed"
            failures.append(config_path.stem)
            print(
                f"[failed] {config_path.stem}; return_code={return_code}; "
                f"see {log_path}",
                flush=True,
            )
        save_state(state_path, state)
        if failures and not args.continue_on_error:
            break

    summary_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "summarize_operation_benchmark.py"),
        "--root",
        str(args.summary_root.expanduser().resolve()),
        "--expected-models",
        str(args.expected_models),
    ]
    if args.manifest is not None:
        summary_command.extend(
            ["--manifest", str(args.manifest.expanduser().resolve())]
        )
    summary_code = subprocess.run(summary_command, cwd=str(REPO_ROOT), check=False).returncode
    state["finished_at"] = datetime.now().astimezone().isoformat()
    state["summary_return_code"] = summary_code
    state["failures"] = failures
    save_state(state_path, state)
    print(f"[state] {state_path}", flush=True)
    if failures or summary_code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
