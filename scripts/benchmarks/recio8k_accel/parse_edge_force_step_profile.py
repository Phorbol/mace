from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _speedup(base_ms: float | None, candidate_ms: float | None) -> float | None:
    if base_ms is None or candidate_ms is None or candidate_ms <= 0:
        return None
    return base_ms / candidate_ms


def _result_key(result: dict) -> str:
    return f"{result['optimizer']}/{result['mode']}"


def _gate_status(result: dict) -> tuple[bool | None, str | None]:
    gate = result.get("gate_result")
    if gate is None:
        return None, None
    return bool(gate.get("accepted", False)), gate.get("fallback_reason")


def summarize_payload(payload: dict) -> dict:
    results = payload.get("results", [])
    base_by_optimizer: dict[str, float] = {}
    for result in results:
        if result.get("mode") == "position_eager":
            base_by_optimizer[result["optimizer"]] = result["timings"]["total_ms"][
                "mean_ms"
            ]

    rows: dict[str, dict] = {}
    required_gate_statuses: list[bool] = []
    for result in results:
        key = _result_key(result)
        optimizer = result["optimizer"]
        mode = result["mode"]
        total_stats = result["timings"]["total_ms"]
        mean_total_ms = total_stats["mean_ms"]
        median_total_ms = total_stats.get("median_ms")
        gate_accepted, fallback_reason = _gate_status(result)
        if gate_accepted is not None:
            required_gate_statuses.append(gate_accepted)
        gate_result = result.get("gate_result") or {}
        graph_stats = gate_result.get("graph_stats") or {}
        rows[key] = {
            "optimizer": optimizer,
            "mode": mode,
            "setup_ms": result.get("setup_ms"),
            "mean_total_ms": mean_total_ms,
            "median_total_ms": median_total_ms,
            "speedup_vs_position_eager": _speedup(
                base_by_optimizer.get(optimizer), mean_total_ms
            ),
            "gate_accepted": gate_accepted,
            "fallback_reason": fallback_reason,
            "compiled_grad_count": gate_result.get("compiled_grad_count"),
            "compiled_grad_filter": gate_result.get("compiled_grad_filter"),
            "compiled_grad_filter_sequence": gate_result.get(
                "compiled_grad_filter_sequence"
            ),
            "graph_node_count": graph_stats.get("node_count"),
            "graph_output_tensor_count": graph_stats.get("output_tensor_count"),
            "loss_first": result.get("loss_first"),
            "loss_last": result.get("loss_last"),
        }

    compile_rows = [row for row in rows.values() if row["mode"] == "edge_compile"]
    compile_grads_rows = [
        row for row in rows.values() if row["mode"] == "edge_compile_grads"
    ]
    compile_grads_sequence_rows = [
        row for row in rows.values() if row["mode"] == "edge_compile_grads_sequence"
    ]
    position_rows = [row for row in rows.values() if row["mode"] == "position_eager"]
    return {
        "source": payload.get("source"),
        "torch_version": payload.get("torch_version"),
        "device": payload.get("device"),
        "device_name": payload.get("device_name"),
        "num_atoms": payload.get("num_atoms"),
        "model": payload.get("model"),
        "cueq": payload.get("cueq"),
        "warmup": payload.get("warmup"),
        "repeats": payload.get("repeats"),
        "rows": rows,
        "all_required_gates_accepted": all(required_gate_statuses)
        if required_gate_statuses
        else None,
        "mean_position_total_ms": mean(row["mean_total_ms"] for row in position_rows)
        if position_rows
        else None,
        "mean_edge_compile_total_ms": mean(row["mean_total_ms"] for row in compile_rows)
        if compile_rows
        else None,
        "mean_edge_compile_grads_total_ms": mean(
            row["mean_total_ms"] for row in compile_grads_rows
        )
        if compile_grads_rows
        else None,
        "mean_edge_compile_grads_sequence_total_ms": mean(
            row["mean_total_ms"] for row in compile_grads_sequence_rows
        )
        if compile_grads_sequence_rows
        else None,
    }


def summarize_file(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload["source"] = str(path)
    return summarize_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+")
    args = parser.parse_args()

    summaries = {
        str(Path(profile)): summarize_file(Path(profile))
        for profile in args.profiles
    }
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
