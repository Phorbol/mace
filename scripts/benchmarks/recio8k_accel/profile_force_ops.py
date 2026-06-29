from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from scripts.benchmarks.recio8k_accel.profile_force_energy_modes import (
    _prepare_profile_model,
    _run_mode_once_with_fallback,
)
from scripts.benchmarks.recio8k_accel.profile_training_step_phases import (
    _create_model,
    _load_batch,
    _sync,
    parse_indices,
)


def _event_value_us(event, field: str) -> float:
    value = getattr(event, field, None)
    if value is None and field == "cuda_time_total":
        value = getattr(event, "device_time_total", None)
    if value is None and field == "self_cuda_time_total":
        value = getattr(event, "self_device_time_total", None)
    return float(value or 0.0)


def _event_value_ms(event, field: str) -> float:
    return _event_value_us(event, field) / 1000.0


def _serialize_input_shapes(value):
    if value in (None, ""):
        return None
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_input_shapes(item) for item in value]
    return value


def summarize_events(
    events: Iterable[object], *, sort_by: str, top_k: int
) -> list[dict]:
    filtered_events = [
        event for event in events if not str(getattr(event, "key", "")).startswith("mace_")
    ]
    sorted_events = sorted(
        filtered_events, key=lambda event: _event_value_us(event, sort_by), reverse=True
    )
    rows: list[dict] = []
    for event in sorted_events[:top_k]:
        row = {
            "name": str(getattr(event, "key", "")),
            "count": int(getattr(event, "count", 0) or 0),
            "self_cuda_time_ms": _event_value_ms(event, "self_cuda_time_total"),
            "cuda_time_ms": _event_value_ms(event, "cuda_time_total"),
            "self_cpu_time_ms": _event_value_ms(event, "self_cpu_time_total"),
            "cpu_time_ms": _event_value_ms(event, "cpu_time_total"),
        }
        input_shapes = _serialize_input_shapes(getattr(event, "input_shapes", None))
        if input_shapes is not None:
            row["input_shapes"] = input_shapes
        rows.append(row)
    return rows


def _profile_mode(model, batch, mode: str, args: argparse.Namespace, device: torch.device) -> dict:
    model.zero_grad(set_to_none=True)
    for _ in range(args.warmup):
        output, loss = _run_mode_once_with_fallback(model, batch, mode, args)
        _sync(device)
        del output, loss
        model.zero_grad(set_to_none=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        with record_function(f"mace_{mode}"):
            output, loss = _run_mode_once_with_fallback(model, batch, mode, args)
            _sync(device)

    events = prof.key_averages(group_by_input_shape=args.group_by_input_shape)
    sort_by = "device_time_total" if device.type == "cuda" else "cpu_time_total"
    payload = {
        "mode": mode,
        "loss": float(loss.detach().cpu()),
        "compile_disabled": bool(getattr(model, "disabled", False)),
        "sort_by": sort_by,
        "top_ops": summarize_events(events, sort_by=sort_by, top_k=args.top_k),
    }
    del output, loss
    return payload


def run_profile(args: argparse.Namespace) -> dict:
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(args.xyz, indices, args.cutoff, device)
    torch.manual_seed(args.seed)
    model = _create_model(
        z_table=z_table,
        cutoff=args.cutoff,
        device=device,
        hidden_channels=args.hidden_channels,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        avg_num_neighbors=args.avg_num_neighbors,
        enable_cueq=args.enable_cueq,
    )
    model = _prepare_profile_model(model, args)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [_profile_mode(model, batch, mode, args, device) for mode in modes]
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "xyz": str(args.xyz),
        "indices": indices,
        "num_structures": len(indices),
        "num_atoms": int(batch.positions.shape[0]),
        "warmup": args.warmup,
        "enable_cueq": args.enable_cueq,
        "train_compile": args.train_compile,
        "train_compile_mode": args.train_compile_mode,
        "train_compile_fullgraph": args.train_compile_fullgraph,
        "train_compile_allow_fallback": args.train_compile_allow_fallback,
        "compile_disabled": bool(getattr(model, "disabled", False)),
        "max_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "results": results,
    }


class ShapeAwareArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if getattr(parsed, "group_by_input_shape", False):
            parsed.record_shapes = True
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = ShapeAwareArgumentParser()
    parser.add_argument("--xyz", type=Path, default=Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"))
    parser.add_argument("--indices", default="0:32")
    parser.add_argument("--output", type=Path, default=Path("force_op_profile.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--modes", default="energy_forward,force_loss_backward")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--max-ell", type=int, default=3)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--avg-num-neighbors", type=float, default=25.55)
    parser.add_argument("--enable-cueq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--energy-weight", type=float, default=40.0)
    parser.add_argument("--forces-weight", type=float, default=1000.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--train-compile", action="store_true", default=False)
    parser.add_argument(
        "--train-compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--train-compile-fullgraph", action="store_true", default=False)
    parser.add_argument(
        "--train-compile-allow-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--record-shapes", action="store_true", default=False)
    parser.add_argument("--group-by-input-shape", action="store_true", default=False)
    parser.add_argument("--profile-memory", action="store_true", default=False)
    parser.add_argument("--with-stack", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_profile(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
