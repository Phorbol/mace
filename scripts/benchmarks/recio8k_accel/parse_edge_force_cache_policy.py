from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_bool_count(text: str, key: str, value: str) -> int:
    patterns = (
        rf"{re.escape(key)}[=: ]+{value}",
        rf"'{re.escape(key)}': {value}",
        rf'"{re.escape(key)}": {value}',
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _parse_float_values(text: str, key: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(rf"{re.escape(key)}[=: ]+([-+0-9.eE]+)", text):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _summarize_jsonl_metrics(path: Path) -> dict[str, object] | None:
    rows: list[dict[str, object]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "edge_force_compile" in payload:
            rows.append(payload)
    if not rows:
        return None

    opt_rows = [row for row in rows if row.get("mode") == "opt"]
    compile_true = sum(row.get("edge_force_compile") is True for row in opt_rows)
    compile_false = sum(row.get("edge_force_compile") is False for row in opt_rows)
    cache_hit_true = sum(row.get("edge_force_cache_hit") is True for row in opt_rows)
    cache_hit_false = sum(row.get("edge_force_cache_hit") is False for row in opt_rows)
    setup_values = [
        float(row["edge_force_compile_setup_seconds"])
        for row in opt_rows
        if isinstance(row.get("edge_force_compile_setup_seconds"), (int, float))
    ]
    step_times = [
        float(row["time"])
        for row in opt_rows
        if isinstance(row.get("time"), (int, float))
    ]
    disabled_reasons = sorted(
        {
            str(row.get("edge_force_compile_disabled_reason"))
            for row in opt_rows
            if row.get("edge_force_compile_disabled_reason") is not None
        }
    )
    by_epoch: dict[str, dict[str, object]] = {}
    for row in opt_rows:
        epoch = str(row.get("epoch"))
        bucket = by_epoch.setdefault(
            epoch,
            {
                "opt_rows": 0,
                "compile_true": 0,
                "compile_false": 0,
                "cache_hit_true": 0,
                "cache_hit_false": 0,
            },
        )
        bucket["opt_rows"] = int(bucket["opt_rows"]) + 1
        if row.get("edge_force_compile") is True:
            bucket["compile_true"] = int(bucket["compile_true"]) + 1
        if row.get("edge_force_compile") is False:
            bucket["compile_false"] = int(bucket["compile_false"]) + 1
        if row.get("edge_force_cache_hit") is True:
            bucket["cache_hit_true"] = int(bucket["cache_hit_true"]) + 1
        if row.get("edge_force_cache_hit") is False:
            bucket["cache_hit_false"] = int(bucket["cache_hit_false"]) + 1

    return {
        "path": str(path),
        "format": "jsonl",
        "opt_rows": len(opt_rows),
        "edge_force_compile_true": compile_true,
        "edge_force_compile_false": compile_false,
        "edge_force_cache_hit_true": cache_hit_true,
        "edge_force_cache_hit_false": cache_hit_false,
        "disabled_reasons": disabled_reasons,
        "compile_setup_seconds_avg": (sum(setup_values) / len(setup_values)) if setup_values else None,
        "compile_setup_seconds_max": max(setup_values) if setup_values else None,
        "step_seconds_avg": (sum(step_times) / len(step_times)) if step_times else None,
        "step_seconds_max": max(step_times) if step_times else None,
        "by_epoch": by_epoch,
    }


def summarize_log(path: Path) -> dict[str, object]:
    jsonl_summary = _summarize_jsonl_metrics(path)
    if jsonl_summary is not None:
        return jsonl_summary
    text = path.read_text(errors="replace")
    compile_true = _parse_bool_count(text, "edge_force_compile", "True") + _parse_bool_count(
        text, "edge_force_compile", "true"
    )
    compile_false = _parse_bool_count(text, "edge_force_compile", "False") + _parse_bool_count(
        text, "edge_force_compile", "false"
    )
    cache_hit_true = _parse_bool_count(text, "edge_force_cache_hit", "True") + _parse_bool_count(
        text, "edge_force_cache_hit", "true"
    )
    cache_hit_false = _parse_bool_count(text, "edge_force_cache_hit", "False") + _parse_bool_count(
        text, "edge_force_cache_hit", "false"
    )
    disabled_reasons = sorted(
        set(
            re.findall(
                r'edge_force_compile_disabled_reason[=: ]+[\"\']?([A-Za-z0-9_-]+)',
                text,
            )
        )
    )
    fallback_mentions = len(
        re.findall(r"fallback|failed; disabling compiled force loss", text, re.I)
    )
    setup_seconds = _parse_float_values(text, "edge_force_compile_setup_seconds")
    return {
        "path": str(path),
        "edge_force_compile_true": compile_true,
        "edge_force_compile_false": compile_false,
        "edge_force_cache_hit_true": cache_hit_true,
        "edge_force_cache_hit_false": cache_hit_false,
        "fallback_mentions": fallback_mentions,
        "disabled_reasons": disabled_reasons,
        "compile_setup_seconds_max": max(setup_seconds) if setup_seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize_log(path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
