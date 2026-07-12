#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmarks.recio8k_accel.parse_metrics import parse_log, parse_nvdmon


INITIAL_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) .*"
    r"Initial: update=(?P<update>\d+),"
)


def _parse_initial_record(log_path: Path) -> dict[str, Any] | None:
    for line in log_path.read_text(errors="ignore").splitlines():
        match = INITIAL_RE.search(line)
        if match:
            timestamp = datetime.strptime(
                match.group("timestamp"), "%Y-%m-%d %H:%M:%S.%f"
            )
            return {
                "timestamp": timestamp.isoformat(),
                "update": int(match.group("update")),
            }
    return None


def _early_update_timing(
    log_path: Path, parsed: dict[str, Any]
) -> dict[str, float] | None:
    initial = _parse_initial_record(log_path)
    if initial is None:
        return None
    epochs = parsed.get("epochs") or []
    first_update = next(
        (epoch for epoch in epochs if epoch.get("update") is not None), None
    )
    if first_update is None:
        return None
    update_delta = int(first_update["update"]) - int(initial["update"])
    if update_delta <= 0:
        return None
    initial_time = datetime.fromisoformat(str(initial["timestamp"]))
    first_time = datetime.fromisoformat(str(first_update["timestamp"]))
    seconds = (first_time - initial_time).total_seconds()
    if seconds <= 0.0:
        return None
    return {
        "early_seconds_per_update": seconds / update_delta,
        "early_updates_per_second": update_delta / seconds,
    }


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


def _best_metric_from_log(parsed: dict[str, Any], key: str) -> float | None:
    records = parsed.get("records") or parsed.get("epochs") or []
    values = [record.get(key) for record in records if record.get(key) is not None]
    if values:
        return float(min(values))
    last = parsed.get("last") or {}
    value = last.get(key)
    return None if value is None else float(value)


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
        "final_update": None,
        "mae_e_mev_atom": None,
        "mae_f_mev_a": None,
        "rmse_e_mev_atom": None,
        "rmse_f_mev_a": None,
        "final_mae_e_mev_atom": None,
        "final_mae_f_mev_a": None,
        "final_rmse_e_mev_atom": None,
        "final_rmse_f_mev_a": None,
        "best_mae_e_mev_atom": None,
        "best_mae_f_mev_a": None,
        "best_rmse_e_mev_atom": None,
        "best_rmse_f_mev_a": None,
        "mean_seconds_per_epoch": None,
        "seconds_per_update": None,
        "updates_per_second": None,
        "early_seconds_per_update": None,
        "early_updates_per_second": None,
        "total_train_seconds_estimate": None,
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
        row["final_update"] = last.get("update")
        row["mae_e_mev_atom"] = last.get("mae_e_mev_atom")
        row["mae_f_mev_a"] = last.get("mae_f_mev_a")
        row["rmse_e_mev_atom"] = last.get("rmse_e_mev_atom")
        row["rmse_f_mev_a"] = last.get("rmse_f_mev_a")
        row["final_mae_e_mev_atom"] = row["mae_e_mev_atom"]
        row["final_mae_f_mev_a"] = row["mae_f_mev_a"]
        row["final_rmse_e_mev_atom"] = row["rmse_e_mev_atom"]
        row["final_rmse_f_mev_a"] = row["rmse_f_mev_a"]
        row["best_mae_e_mev_atom"] = _best_metric_from_log(parsed, "mae_e_mev_atom")
        row["best_mae_f_mev_a"] = _best_metric_from_log(parsed, "mae_f_mev_a")
        row["best_rmse_e_mev_atom"] = _best_metric_from_log(parsed, "rmse_e_mev_atom")
        row["best_rmse_f_mev_a"] = _best_metric_from_log(parsed, "rmse_f_mev_a")
        timing = parsed.get("timing") or {}
        row["mean_seconds_per_epoch"] = timing.get("mean_seconds_per_epoch")
        mean_epoch = row.get("mean_seconds_per_epoch")
        if mean_epoch is not None and steps_per_epoch:
            row["seconds_per_update"] = float(mean_epoch) / int(steps_per_epoch)
            row["updates_per_second"] = int(steps_per_epoch) / float(mean_epoch)
        early_timing = _early_update_timing(log_path, parsed)
        if early_timing is not None:
            row.update(early_timing)
        if (
            row.get("seconds_per_update") is not None
            and row.get("effective_updates") is not None
        ):
            row["total_train_seconds_estimate"] = (
                float(row["seconds_per_update"]) * int(row["effective_updates"])
            )

    nvdmon = _case_nvdmon(case_dir)
    if nvdmon is not None:
        row["max_fb_memory_mb"] = nvdmon.get("max_fb_memory_mb")
        row["mean_sm_util_percent"] = nvdmon.get("mean_sm_util_percent")
        row["mean_mem_util_percent"] = nvdmon.get("mean_mem_util_percent")

    return row



PAIRWISE_COMPARISONS = (
    ("adamw", "cueq_adamw"),
    ("adamw", "hybrid_muon"),
    ("adamw", "hybrid_muon_tace"),
    ("hybrid_muon", "cueq_hybrid_muon"),
    ("hybrid_muon_tace", "cueq_hybrid_muon_tace"),
    ("eager", "hybrid_muon"),
    ("cueq_adamw", "cueq_hybrid_muon"),
    ("cueq_adamw", "cueq_hybrid_muon_tace"),
    ("cueq", "cueq_hybrid_muon"),
    ("compile", "hybrid_muon_compile"),
    ("cueq_compile", "cueq_hybrid_muon_compile"),
)


def _numeric_delta(candidate: Any, baseline: Any) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def _numeric_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _speedup_vs_baseline(baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    update_speedup = _numeric_ratio(
        baseline.get("seconds_per_update"), candidate.get("seconds_per_update")
    )
    if update_speedup is not None:
        return update_speedup
    return _numeric_ratio(
        baseline.get("mean_seconds_per_epoch"), candidate.get("mean_seconds_per_epoch")
    )


def _comparison_row(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": baseline.get("root"),
        "baseline_case": baseline.get("case"),
        "candidate_case": candidate.get("case"),
        "target_steps": candidate.get("target_steps") or baseline.get("target_steps"),
        "effective_updates": candidate.get("effective_updates") or baseline.get("effective_updates"),
        "stage_two_start_update": candidate.get("stage_two_start_update") or baseline.get("stage_two_start_update"),
        "scheduler": candidate.get("scheduler") or baseline.get("scheduler"),
        "lr_scheduler_interval": candidate.get("lr_scheduler_interval") or baseline.get("lr_scheduler_interval"),
        "hybrid_muon_mode": candidate.get("hybrid_muon_mode"),
        "hybrid_muon_routing": candidate.get("hybrid_muon_routing"),
        "hybrid_muon_lr_factor": candidate.get("hybrid_muon_lr_factor"),
        "baseline_mae_e_mev_atom": baseline.get("mae_e_mev_atom"),
        "candidate_mae_e_mev_atom": candidate.get("mae_e_mev_atom"),
        "mae_e_delta_mev_atom": _numeric_delta(candidate.get("mae_e_mev_atom"), baseline.get("mae_e_mev_atom")),
        "mae_e_ratio": _numeric_ratio(candidate.get("mae_e_mev_atom"), baseline.get("mae_e_mev_atom")),
        "baseline_mae_f_mev_a": baseline.get("mae_f_mev_a"),
        "candidate_mae_f_mev_a": candidate.get("mae_f_mev_a"),
        "mae_f_delta_mev_a": _numeric_delta(candidate.get("mae_f_mev_a"), baseline.get("mae_f_mev_a")),
        "mae_f_ratio": _numeric_ratio(candidate.get("mae_f_mev_a"), baseline.get("mae_f_mev_a")),
        "baseline_final_mae_e_mev_atom": baseline.get("final_mae_e_mev_atom"),
        "candidate_final_mae_e_mev_atom": candidate.get("final_mae_e_mev_atom"),
        "final_mae_e_delta_mev_atom": _numeric_delta(candidate.get("final_mae_e_mev_atom"), baseline.get("final_mae_e_mev_atom")),
        "final_mae_e_ratio": _numeric_ratio(candidate.get("final_mae_e_mev_atom"), baseline.get("final_mae_e_mev_atom")),
        "baseline_final_mae_f_mev_a": baseline.get("final_mae_f_mev_a"),
        "candidate_final_mae_f_mev_a": candidate.get("final_mae_f_mev_a"),
        "final_mae_f_delta_mev_a": _numeric_delta(candidate.get("final_mae_f_mev_a"), baseline.get("final_mae_f_mev_a")),
        "final_mae_f_ratio": _numeric_ratio(candidate.get("final_mae_f_mev_a"), baseline.get("final_mae_f_mev_a")),
        "baseline_best_mae_e_mev_atom": baseline.get("best_mae_e_mev_atom"),
        "candidate_best_mae_e_mev_atom": candidate.get("best_mae_e_mev_atom"),
        "best_mae_e_delta_mev_atom": _numeric_delta(candidate.get("best_mae_e_mev_atom"), baseline.get("best_mae_e_mev_atom")),
        "best_mae_e_ratio": _numeric_ratio(candidate.get("best_mae_e_mev_atom"), baseline.get("best_mae_e_mev_atom")),
        "baseline_best_mae_f_mev_a": baseline.get("best_mae_f_mev_a"),
        "candidate_best_mae_f_mev_a": candidate.get("best_mae_f_mev_a"),
        "best_mae_f_delta_mev_a": _numeric_delta(candidate.get("best_mae_f_mev_a"), baseline.get("best_mae_f_mev_a")),
        "best_mae_f_ratio": _numeric_ratio(candidate.get("best_mae_f_mev_a"), baseline.get("best_mae_f_mev_a")),
        "baseline_seconds_per_epoch": baseline.get("mean_seconds_per_epoch"),
        "candidate_seconds_per_epoch": candidate.get("mean_seconds_per_epoch"),
        "seconds_per_epoch_delta": _numeric_delta(candidate.get("mean_seconds_per_epoch"), baseline.get("mean_seconds_per_epoch")),
        "seconds_per_epoch_ratio": _numeric_ratio(candidate.get("mean_seconds_per_epoch"), baseline.get("mean_seconds_per_epoch")),
        "speedup_vs_baseline": _speedup_vs_baseline(baseline, candidate),
        "baseline_seconds_per_update": baseline.get("seconds_per_update"),
        "candidate_seconds_per_update": candidate.get("seconds_per_update"),
        "seconds_per_update_delta": _numeric_delta(candidate.get("seconds_per_update"), baseline.get("seconds_per_update")),
        "seconds_per_update_ratio": _numeric_ratio(candidate.get("seconds_per_update"), baseline.get("seconds_per_update")),
        "baseline_updates_per_second": baseline.get("updates_per_second"),
        "candidate_updates_per_second": candidate.get("updates_per_second"),
        "updates_per_second_delta": _numeric_delta(candidate.get("updates_per_second"), baseline.get("updates_per_second")),
        "updates_per_second_ratio": _numeric_ratio(candidate.get("updates_per_second"), baseline.get("updates_per_second")),
        "baseline_max_fb_memory_mb": baseline.get("max_fb_memory_mb"),
        "candidate_max_fb_memory_mb": candidate.get("max_fb_memory_mb"),
        "max_fb_memory_delta_mb": _numeric_delta(candidate.get("max_fb_memory_mb"), baseline.get("max_fb_memory_mb")),
    }


def summarize_pairwise_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    roots = sorted({str(row.get("root")) for row in rows})
    for root in roots:
        by_case = {row.get("case"): row for row in rows if str(row.get("root")) == root}
        for baseline_case, candidate_case in PAIRWISE_COMPARISONS:
            baseline = by_case.get(baseline_case)
            candidate = by_case.get(candidate_case)
            if baseline is not None and candidate is not None:
                comparisons.append(_comparison_row(baseline, candidate))
    return comparisons


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
    parser.add_argument("--comparisons-csv", type=Path, default=None)
    parser.add_argument("--comparisons-json", type=Path, default=None)
    args = parser.parse_args()

    rows = summarize_roots(args.roots)
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.csv is not None:
        _write_csv(rows, args.csv)
    comparisons = summarize_pairwise_comparisons(rows)
    if args.comparisons_json is not None:
        args.comparisons_json.write_text(json.dumps(comparisons, indent=2, sort_keys=True))
    if args.comparisons_csv is not None:
        _write_csv(comparisons, args.comparisons_csv)


if __name__ == "__main__":
    main()
