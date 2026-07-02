from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+):.*MAE_E_per_atom=\s*(?P<mae_e>nan|[0-9.]+) meV, "
    r"MAE_F=\s*(?P<mae_f>nan|[0-9.]+)"
)
EDGE_SUMMARY_RE = re.compile(
    r"Edge-force compile epoch (?P<epoch>\d+) summary: .*?"
    r"steps=(?P<steps>\d+), .*?"
    r"compile_setup_seconds=(?P<compile>[0-9.]+), "
    r"opt_step_seconds=(?P<opt>[0-9.]+)"
)
TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<config>[^|]+?)\s*\|\s*(?P<mae_e>[0-9.]+)\s*\|"
    r"\s*(?P<mae_f>[0-9.]+)\s*\|\s*(?P<rel_f>[0-9.]+)\s*\|"
)
OOM_RE = re.compile(
    r"outofmemory|out of memory|cuda out of memory|oom-kill|oom killed|oom",
    re.IGNORECASE,
)


def _metric(value: str) -> float:
    return float("nan") if value == "nan" else float(value)


def _read_text_files(case_dir: Path) -> str:
    chunks = []
    patterns = ("*.out", "*.err", "slurm-*.out", "slurm-*.err", "logs/*.log")
    for pattern in patterns:
        for path in sorted(case_dir.glob(pattern)):
            chunks.append(path.read_text(errors="ignore"))
    return "\n".join(chunks)


def _parse_epoch_metrics(text: str) -> dict[str, Any]:
    epochs = []
    for match in EPOCH_RE.finditer(text):
        epochs.append(
            {
                "epoch": int(match.group("epoch")),
                "mae_e_mev_atom": _metric(match.group("mae_e")),
                "mae_f_mev_a": _metric(match.group("mae_f")),
            }
        )
    if not epochs:
        return {}
    last = epochs[-1]
    return {
        "last_valid_epoch": last["epoch"],
        "last_valid_mae_e_mev_atom": last["mae_e_mev_atom"],
        "last_valid_mae_f_mev_a": last["mae_f_mev_a"],
        "has_nan": any(
            math.isnan(epoch["mae_e_mev_atom"]) or math.isnan(epoch["mae_f_mev_a"])
            for epoch in epochs
        ),
    }


def _parse_edge_summaries(text: str) -> dict[str, Any]:
    summaries = []
    for match in EDGE_SUMMARY_RE.finditer(text):
        summaries.append(
            {
                "epoch": int(match.group("epoch")),
                "steps": int(match.group("steps")),
                "compile_setup_seconds": float(match.group("compile")),
                "opt_step_seconds": float(match.group("opt")),
            }
        )
    if not summaries:
        return {}
    hot_epoch_seconds = [item["opt_step_seconds"] for item in summaries if item["epoch"] > 0]
    hot_batch_seconds = [
        item["opt_step_seconds"] / item["steps"]
        for item in summaries
        if item["epoch"] > 0 and item["steps"] > 0
    ]
    mean_hot_epoch = mean(hot_epoch_seconds) if hot_epoch_seconds else None
    return {
        "first_compile_setup_seconds": summaries[0]["compile_setup_seconds"],
        "mean_hot_epoch_opt_seconds": mean_hot_epoch,
        "mean_hot_batch_step_seconds": mean(hot_batch_seconds) if hot_batch_seconds else None,
        "mean_hot_opt_step_seconds": mean_hot_epoch,
        "edge_summary_epochs": len(summaries),
    }


def _parse_error_tables(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in text.splitlines():
        match = TABLE_ROW_RE.match(line)
        if not match:
            continue
        config = match.group("config").strip()
        mae_e = float(match.group("mae_e"))
        mae_f = float(match.group("mae_f"))
        rel_f = float(match.group("rel_f"))
        if config == "valid_Default":
            result.update(
                {
                    "final_valid_mae_e_mev_atom": mae_e,
                    "final_valid_mae_f_mev_a": mae_f,
                    "final_valid_relative_f_mae_percent": rel_f,
                }
            )
        elif config == "train_Default":
            result.update(
                {
                    "final_train_mae_e_mev_atom": mae_e,
                    "final_train_mae_f_mev_a": mae_f,
                    "final_train_relative_f_mae_percent": rel_f,
                }
            )
        elif config.startswith("Default_"):
            result.update(
                {
                    "final_test_mae_e_mev_atom": mae_e,
                    "final_test_mae_f_mev_a": mae_f,
                    "final_test_relative_f_mae_percent": rel_f,
                }
            )
    return result


def parse_nvdmon(path: Path) -> dict[str, Any] | None:
    samples = []
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 16:
            continue
        try:
            samples.append(
                {
                    "power_w": float(fields[2]),
                    "sm_util_percent": float(fields[5]),
                    "mem_util_percent": float(fields[6]),
                    "fb_memory_mb": int(fields[15]),
                }
            )
        except ValueError:
            continue
    if not samples:
        return None
    return {
        "max_fb_memory_mb": max(sample["fb_memory_mb"] for sample in samples),
        "mean_sm_util_percent": mean(sample["sm_util_percent"] for sample in samples),
        "mean_mem_util_percent": mean(sample["mem_util_percent"] for sample in samples),
        "mean_power_w": mean(sample["power_w"] for sample in samples),
        "nvdmon_samples": len(samples),
    }


def _status(text: str) -> str:
    if OOM_RE.search(text):
        return "oom"
    if "Training complete" in text and "Done" in text:
        return "completed"
    if text.strip():
        return "failed"
    return "pending"


def summarize_case(case_dir: Path) -> dict[str, Any]:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    text = _read_text_files(case_dir)
    result: dict[str, Any] = {
        "case": case_dir.name,
        "path": str(case_dir),
        "status": _status(text),
    }
    for key in ("sweep", "batch_size", "max_num_epochs", "start_swa", "eval_interval"):
        if key in manifest:
            result[key] = manifest[key]
    if "params" in manifest:
        params = manifest["params"]
        for key in ("num_channels", "max_L", "num_interactions", "r_max", "correlation"):
            if key in params:
                result[key] = params[key]

    result.update(_parse_epoch_metrics(text))
    result.update(_parse_edge_summaries(text))
    result.update(_parse_error_tables(text))

    nvdmon_files = sorted(case_dir.glob("nvdmon_job-*.log"))
    if nvdmon_files:
        nvdmon = parse_nvdmon(nvdmon_files[-1])
        if nvdmon is not None:
            result.update(nvdmon)
    return result


def summarize_root(root: Path) -> list[dict[str, Any]]:
    case_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    return [summarize_case(case_dir) for case_dir in case_dirs]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "case",
        "status",
        "batch_size",
        "num_channels",
        "max_L",
        "num_interactions",
        "r_max",
        "correlation",
        "max_num_epochs",
        "start_swa",
        "eval_interval",
        "max_fb_memory_mb",
        "first_compile_setup_seconds",
        "mean_hot_epoch_opt_seconds",
        "mean_hot_batch_step_seconds",
        "mean_hot_opt_step_seconds",
        "last_valid_mae_e_mev_atom",
        "last_valid_mae_f_mev_a",
        "final_valid_mae_e_mev_atom",
        "final_valid_mae_f_mev_a",
        "final_test_mae_e_mev_atom",
        "final_test_mae_f_mev_a",
        "path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RECIO8k scaling runs.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        rows.extend(summarize_root(root))
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
