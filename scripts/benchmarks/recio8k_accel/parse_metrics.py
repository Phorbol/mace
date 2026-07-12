from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean


EPOCH_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) .*"
    r"Epoch (?P<epoch>\d+):(?: update=(?P<update>\d+),)?.*(?P<metric>MAE|RMSE)_E(?P<per_atom>_per_atom)?=\s*"
    r"(?P<energy>nan|[0-9.]+) meV, (?P=metric)_F=\s*"
    r"(?P<forces>nan|[0-9.]+)"
    r"(?: meV / A, (?P=metric)_(?P<extra>stress|virials(?:_per_atom)?)=\s*"
    r"(?P<extra_value>nan|[0-9.]+))?"
)
TRAIN_COMPILE_FALLBACK_RE = re.compile(
    r"training torch\.compile failed during backward; disabling compiled "
    r"training model and retrying eager: (?P<reason>.*)$"
)


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


def _parse_metric(value: str) -> float:
    return float("nan") if value == "nan" else float(value)


def _summarize_timing(epochs: list[dict]) -> dict | None:
    timed_epochs = [epoch for epoch in epochs if "timestamp" in epoch]
    if len(timed_epochs) < 2:
        return None

    intervals = []
    for prev, curr in zip(timed_epochs, timed_epochs[1:]):
        epoch_delta = curr["epoch"] - prev["epoch"]
        if epoch_delta <= 0:
            continue
        prev_time = datetime.fromisoformat(prev["timestamp"])
        curr_time = datetime.fromisoformat(curr["timestamp"])
        seconds = (curr_time - prev_time).total_seconds()
        intervals.append(
            {
                "from_epoch": prev["epoch"],
                "to_epoch": curr["epoch"],
                "seconds": seconds,
                "seconds_per_epoch": seconds / epoch_delta,
            }
        )

    if not intervals:
        return None
    return {
        "report_intervals": intervals,
        "mean_seconds_per_epoch": mean(
            interval["seconds_per_epoch"] for interval in intervals
        ),
    }


def parse_log(path: Path) -> dict:
    epochs = []
    train_compile_fallback_reasons = []
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            timestamp = _parse_timestamp(match.group("timestamp"))
            metric_prefix = match.group("metric").lower()
            energy_key = (
                f"{metric_prefix}_e_mev_atom"
                if match.group("per_atom")
                else f"{metric_prefix}_e_mev"
            )
            epoch = {
                "epoch": int(match.group("epoch")),
                energy_key: _parse_metric(match.group("energy")),
                f"{metric_prefix}_f_mev_a": _parse_metric(match.group("forces")),
                "timestamp": timestamp.isoformat(),
            }
            if match.group("update") is not None:
                epoch["update"] = int(match.group("update"))
            extra = match.group("extra")
            if extra is not None:
                extra_key = (
                    f"{metric_prefix}_virials_mev_atom"
                    if extra == "virials_per_atom"
                    else f"{metric_prefix}_{extra}_mev_a3"
                    if extra == "stress"
                    else f"{metric_prefix}_{extra}_mev"
                )
                epoch[extra_key] = _parse_metric(match.group("extra_value"))
            epochs.append(epoch)
            continue
        fallback_match = TRAIN_COMPILE_FALLBACK_RE.search(line)
        if fallback_match:
            train_compile_fallback_reasons.append(
                fallback_match.group("reason").strip()
            )

    nan_epochs = [
        epoch["epoch"]
        for epoch in epochs
        if any(
            math.isnan(value)
            for key, value in epoch.items()
            if key.endswith(("_e_mev_atom", "_e_mev", "_f_mev_a"))
        )
    ]
    summary = {
        "log": str(path),
        "epochs": epochs,
        "last": epochs[-1] if epochs else None,
        "has_nan": bool(nan_epochs),
        "train_compile_fallback": bool(train_compile_fallback_reasons),
        "train_compile_fallback_count": len(train_compile_fallback_reasons),
    }
    if train_compile_fallback_reasons:
        summary["train_compile_fallback_reasons"] = train_compile_fallback_reasons
    if nan_epochs:
        summary["first_nan_epoch"] = nan_epochs[0]
    timing = _summarize_timing(epochs)
    if timing is not None:
        summary["timing"] = timing
    return summary


def parse_nvdmon(path: Path) -> dict | None:
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
        "path": str(path),
        "samples": len(samples),
        "max_fb_memory_mb": max(sample["fb_memory_mb"] for sample in samples),
        "mean_sm_util_percent": mean(sample["sm_util_percent"] for sample in samples),
        "mean_mem_util_percent": mean(sample["mem_util_percent"] for sample in samples),
        "mean_power_w": mean(sample["power_w"] for sample in samples),
    }


def _case_summary(log: Path) -> dict:
    summary = parse_log(log)
    nvdmon_files = sorted(log.parents[1].glob("nvdmon_job-*.log"))
    if nvdmon_files:
        nvdmon = parse_nvdmon(nvdmon_files[-1])
        if nvdmon is not None:
            summary["nvdmon"] = nvdmon
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()

    summaries = {}
    for root_name in args.roots:
        root = Path(root_name)
        for log in root.glob("*/logs/*.log"):
            summaries[f"{root.name}/{log.parents[1].name}"] = _case_summary(log)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
