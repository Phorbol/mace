from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean


EPOCH_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) .*"
    r"Epoch (?P<epoch>\d+):.*MAE_E_per_atom=\s*(?P<mae_e>[0-9.]+) meV, "
    r"MAE_F=\s*(?P<mae_f>[0-9.]+)"
)


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


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
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            timestamp = _parse_timestamp(match.group("timestamp"))
            epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "mae_e_mev_atom": float(match.group("mae_e")),
                    "mae_f_mev_a": float(match.group("mae_f")),
                    "timestamp": timestamp.isoformat(),
                }
            )

    summary = {
        "log": str(path),
        "epochs": epochs,
        "last": epochs[-1] if epochs else None,
    }
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
