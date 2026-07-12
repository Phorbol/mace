#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.recio8k_accel.parse_metrics import parse_log, parse_nvdmon


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text())


def _bool_from_case(case_name: str, token: str) -> bool:
    return token in case_name.split("_") or token in case_name


def _steps_per_epoch(manifest: dict[str, Any]) -> int | None:
    train_size = manifest.get("train_size")
    batch_size = manifest.get("batch_size")
    if train_size is None or batch_size in (None, 0):
        return None
    return int(math.ceil(int(train_size) / int(batch_size)))


def _case_log(case_dir: Path) -> Path | None:
    logs = sorted((case_dir / "logs").glob("*.log"))
    return logs[-1] if logs else None


def _case_nvdmon(case_dir: Path) -> dict[str, Any] | None:
    nvdmon_files = sorted(case_dir.glob("nvdmon_job-*.log"))
    if not nvdmon_files:
        return None
    return parse_nvdmon(nvdmon_files[-1])


def summarize_case(root: Path, case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    case_name = case_dir.name
    steps_per_epoch = _steps_per_epoch(manifest)
    max_epochs = manifest.get("max_num_epochs")
    max_updates = manifest.get("max_num_updates")
    stage_two_epoch = manifest.get("stage_two_start_epoch")
    stage_two_update = manifest.get("stage_two_start_update")

    row: dict[str, Any] = {
        "root": str(root),
        "case": case_name,
        "compile": "compile" in case_name,
        "cueq": "cueq" in case_name,
        "hybrid_muon": "hybrid_muon" in case_name,
        "target_steps": manifest.get("target_steps"),
        "train_size": manifest.get("train_size"),
        "batch_size": manifest.get("batch_size"),
        "max_num_epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "effective_updates": max_updates,
        "stage_two_start_epoch": stage_two_epoch,
        "stage_two_start_update": stage_two_update,
        "stage1_energy_weight": manifest.get("stage1_energy_weight"),
        "stage1_forces_weight": manifest.get("stage1_forces_weight"),
        "stage2_energy_weight": manifest.get("stage2_energy_weight"),
        "stage2_forces_weight": manifest.get("stage2_forces_weight"),
        "scheduler": manifest.get("scheduler"),
        "lr_scheduler_interval": manifest.get("lr_scheduler_interval"),
        "lr_wsd_warmup_ratio": manifest.get("lr_wsd_warmup_ratio"),
        "lr_wsd_stop_lr_ratio": manifest.get("lr_wsd_stop_lr_ratio"),
        "lr_wsd_decay_phase_ratio": manifest.get("lr_wsd_decay_phase_ratio"),
        "lr_wsd_decay_type": manifest.get("lr_wsd_decay_type"),
        "lr": manifest.get("lr"),
        "weight_decay": manifest.get("weight_decay"),
        "hybrid_muon_mode": manifest.get("hybrid_muon_mode"),
        "hybrid_muon_routing": manifest.get("hybrid_muon_routing"),
        "hybrid_muon_lr_factor": manifest.get("hybrid_muon_lr_factor"),
        "hybrid_muon_weight_decay": manifest.get("hybrid_muon_weight_decay"),
        "hybrid_muon_adam_variant": manifest.get("hybrid_muon_adam_variant"),
        "hybrid_muon_lr_scale_mode": manifest.get("hybrid_muon_lr_scale_mode"),
        "compile_setup_gate": manifest.get("compile_setup_gate"),
        "compile_max_cache_entries": manifest.get("compile_max_cache_entries"),
        "compile_parity_gradients": manifest.get("compile_parity_gradients"),
        "compile_parity_check_strict": manifest.get("compile_parity_check_strict"),
        "log": None,
        "final_epoch": None,
        "mae_e_mev_atom": None,
        "mae_f_mev_a": None,
        "rmse_e_mev_atom": None,
        "rmse_f_mev_a": None,
        "mean_seconds_per_epoch": None,
        "train_compile_fallback": None,
        "train_compile_fallback_count": None,
        "has_nan": None,
        "max_fb_memory_mb": None,
        "mean_sm_util_percent": None,
        "mean_mem_util_percent": None,
    }
    if row["effective_updates"] is None and steps_per_epoch is not None and max_epochs is not None:
        row["effective_updates"] = steps_per_epoch * int(max_epochs)
    if row["stage_two_start_update"] is None and steps_per_epoch is not None and stage_two_epoch is not None:
        row["stage_two_start_update"] = steps_per_epoch * int(stage_two_epoch)

    log_path = _case_log(case_dir)
    if log_path is not None:
        parsed = parse_log(log_path)
        row["log"] = str(log_path)
        row["train_compile_fallback"] = parsed.get("train_compile_fallback")
        row["train_compile_fallback_count"] = parsed.get("train_compile_fallback_count")
        row["has_nan"] = parsed.get("has_nan")
        last = parsed.get("last") or {}
        row["final_epoch"] = last.get("epoch")
        row["mae_e_mev_atom"] = last.get("mae_e_mev_atom")
        row["mae_f_mev_a"] = last.get("mae_f_mev_a")
        row["rmse_e_mev_atom"] = last.get("rmse_e_mev_atom")
        row["rmse_f_mev_a"] = last.get("rmse_f_mev_a")
        timing = parsed.get("timing") or {}
        row["mean_seconds_per_epoch"] = timing.get("mean_seconds_per_epoch")

    nvdmon = _case_nvdmon(case_dir)
    if nvdmon is not None:
        row["max_fb_memory_mb"] = nvdmon.get("max_fb_memory_mb")
        row["mean_sm_util_percent"] = nvdmon.get("mean_sm_util_percent")
        row["mean_mem_util_percent"] = nvdmon.get("mean_mem_util_percent")

    return row


def summarize_roots(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        manifest = _load_manifest(root)
        for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if (case_dir / "logs").is_dir():
                rows.append(summarize_case(root, case_dir, manifest))
    return rows


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        output.write_text("")
        return
    fieldnames = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize_roots(args.roots)
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.csv is not None:
        _write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
