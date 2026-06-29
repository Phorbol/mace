from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch


@contextmanager
def maybe_disable_donated_buffer(disable: bool):
    if not disable:
        yield
        return
    import torch._functorch.config as functorch_config

    previous = functorch_config.donated_buffer
    functorch_config.donated_buffer = False
    try:
        yield
    finally:
        functorch_config.donated_buffer = previous


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    op: str
    input_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]


CASES: tuple[SyntheticCase, ...] = (
    SyntheticCase(
        name="mm_4632x1280_1280x64",
        op="mm",
        input_shape=(4632, 1280),
        weight_shape=(1280, 64),
    ),
    SyntheticCase(
        name="bmm_1x286x128_128x128",
        op="bmm",
        input_shape=(1, 286, 128),
        weight_shape=(1, 128, 128),
    ),
    SyntheticCase(
        name="bmm_1x858x128_128x128",
        op="bmm",
        input_shape=(1, 858, 128),
        weight_shape=(1, 128, 128),
    ),
)


def select_cases(value: str) -> list[SyntheticCase]:
    by_name = {case.name: case for case in CASES}
    if value == "all":
        return list(CASES)
    selected: list[SyntheticCase] = []
    for name in [chunk.strip() for chunk in value.split(",") if chunk.strip()]:
        if name not in by_name:
            raise ValueError(
                f"unknown synthetic force-backward case {name!r}; choices: {sorted(by_name)}"
            )
        selected.append(by_name[name])
    if not selected:
        raise ValueError("at least one case is required")
    return selected


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype {name!r}")


def make_inputs(
    case: SyntheticCase, *, device: torch.device, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(case.input_shape, generator=generator, dtype=dtype).to(device)
    w = torch.randn(case.weight_shape, generator=generator, dtype=dtype).to(device)
    x.requires_grad_(True)
    w.requires_grad_(True)
    return x, w


def contraction_energy(case: SyntheticCase, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if case.op == "mm":
        y = x @ w
    elif case.op == "bmm":
        y = torch.bmm(x, w)
    else:
        raise ValueError(f"unsupported op {case.op!r}")
    return y.sin().square().mean() + 0.01 * y.square().mean()


def force_loss_value(case: SyntheticCase, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    energy = contraction_energy(case, x, w)
    force = -torch.autograd.grad(energy, x, create_graph=True)[0]
    return force.square().mean() + 0.001 * energy.square()


def _snapshot_with_fn(
    case: SyntheticCase,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict:
    x, w = make_inputs(case, device=device, dtype=dtype, seed=seed)
    loss = fn(x, w)
    loss.backward()
    if x.grad is None or w.grad is None:
        raise RuntimeError("force-loss snapshot did not produce both input and weight gradients")
    return {
        "loss": float(loss.detach().cpu()),
        "input_grad": x.grad.detach().clone(),
        "weight_grad": w.grad.detach().clone(),
        "input_grad_norm": float(x.grad.detach().norm().cpu()),
        "weight_grad_norm": float(w.grad.detach().norm().cpu()),
    }


def force_loss_snapshot(
    case: SyntheticCase, *, device: torch.device, dtype: torch.dtype, seed: int
) -> dict:
    return _snapshot_with_fn(
        case,
        lambda x, w: force_loss_value(case, x, w),
        device=device,
        dtype=dtype,
        seed=seed,
    )


def compare_snapshots(left: dict, right: dict, *, atol: float, rtol: float) -> dict:
    input_diff = (left["input_grad"] - right["input_grad"]).abs().max()
    weight_diff = (left["weight_grad"] - right["weight_grad"]).abs().max()
    loss_diff = abs(left["loss"] - right["loss"])
    input_ok = torch.allclose(left["input_grad"], right["input_grad"], atol=atol, rtol=rtol)
    weight_ok = torch.allclose(left["weight_grad"], right["weight_grad"], atol=atol, rtol=rtol)
    loss_ok = loss_diff <= atol + rtol * abs(left["loss"])
    return {
        "ok": bool(input_ok and weight_ok and loss_ok),
        "loss_abs_diff": float(loss_diff),
        "input_grad_max_abs_diff": float(input_diff.cpu()),
        "weight_grad_max_abs_diff": float(weight_diff.cpu()),
    }


def _summarize_snapshot(snapshot: dict) -> dict:
    return {
        "loss": snapshot["loss"],
        "input_grad_norm": snapshot["input_grad_norm"],
        "weight_grad_norm": snapshot["weight_grad_norm"],
    }


def _compiled_force_loss_fn(
    case: SyntheticCase, *, mode: str, fullgraph: bool
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return force_loss_value(case, x, w)

    return torch.compile(fn, mode=mode, fullgraph=fullgraph)


def _time_snapshot_fn(
    case: SyntheticCase,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict:
    times: list[float] = []
    last_loss = None
    for step in range(warmup + repeats):
        x, w = make_inputs(case, device=device, dtype=dtype, seed=seed + step)
        _sync(device)
        start = time.perf_counter()
        loss = fn(x, w)
        loss.backward()
        _sync(device)
        elapsed = time.perf_counter() - start
        if step >= warmup:
            times.append(elapsed)
        last_loss = float(loss.detach().cpu())
    return {
        "mean_seconds": mean(times) if times else None,
        "times_seconds": times,
        "last_loss": last_loss,
    }


def run_case(case: SyntheticCase, args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict:
    eager_fn = lambda x, w: force_loss_value(case, x, w)
    eager_snapshot = force_loss_snapshot(case, device=device, dtype=dtype, seed=args.seed)
    result = {
        "case": asdict(case),
        "eager_snapshot": _summarize_snapshot(eager_snapshot),
        "eager_timing": _time_snapshot_fn(
            case,
            eager_fn,
            device=device,
            dtype=dtype,
            seed=args.seed + 1000,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }
    if not args.compile:
        return result

    try:
        compiled_fn = _compiled_force_loss_fn(
            case, mode=args.compile_mode, fullgraph=args.compile_fullgraph
        )
        compiled_snapshot = _snapshot_with_fn(
            case,
            compiled_fn,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        result["compiled_status"] = "ok"
        result["compiled_snapshot"] = _summarize_snapshot(compiled_snapshot)
        result["compiled_equivalence"] = compare_snapshots(
            eager_snapshot, compiled_snapshot, atol=args.atol, rtol=args.rtol
        )
        result["compiled_timing"] = _time_snapshot_fn(
            case,
            compiled_fn,
            device=device,
            dtype=dtype,
            seed=args.seed + 1000,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    except Exception as exc:  # pylint: disable=broad-except
        result["compiled_status"] = "error"
        result["compiled_error"] = repr(exc)
    return result


def _run_probe_impl(args: argparse.Namespace) -> dict:
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    dtype = _dtype_from_name(args.dtype)
    cases = select_cases(args.cases)
    results = [run_case(case, args, device, dtype) for case in cases]
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": args.dtype,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "compile_fullgraph": args.compile_fullgraph,
        "disable_donated_buffer": args.disable_donated_buffer,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "max_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else None
        ),
        "results": results,
    }


def run_probe(args: argparse.Namespace) -> dict:
    with maybe_disable_donated_buffer(args.disable_donated_buffer):
        return _run_probe_impl(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all")
    parser.add_argument("--output", type=Path, default=Path("force_backward_compile_ops.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument("--compile-fullgraph", action="store_true", default=False)
    parser.add_argument("--disable-donated-buffer", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_probe(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
