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


def summarize_log(path: Path) -> dict[str, object]:
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
