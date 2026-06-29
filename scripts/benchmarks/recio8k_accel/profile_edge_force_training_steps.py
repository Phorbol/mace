from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch

from mace import modules
from mace.tools.force_compile import compile_fx_graph_module, trace_force_closure
from mace.tools.hybrid_muon import (
    HybridMuon,
    build_hybrid_muon_param_groups,
    summarize_hybrid_muon_routes,
)
from mace.tools.training_compile import edge_force_compile_result_from_trace

from scripts.benchmarks.recio8k_accel.probe_edge_vector_force_equivalence import (  # noqa: E402
    _batch_dict,
    _edge_vector_inputs,
    _mace_energy_from_vectors,
    _max_abs_diff,
    _named_parameter_grads,
    _within_tolerance,
    create_probe_model,
    edge_gradient_to_atomic_forces,
)
from scripts.benchmarks.recio8k_accel.probe_training_compile import (  # noqa: E402
    _load_batch,
)

OPTIMIZER_CHOICES = {"adam", "hybrid_muon"}
MODE_CHOICES = {"position_eager", "edge_eager", "edge_compile"}


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    optimizer: torch.optim.Optimizer
    route_summary: list[dict] | None = None


@dataclass(frozen=True)
class StepModeSpec:
    name: str
    step_fn: Callable[[torch.optim.Optimizer], torch.Tensor]
    gate_result: dict | None
    setup_ms: float


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [int(part) if part else None for part in chunk.split(":")]
            if len(parts) > 3:
                raise ValueError(f"invalid index range: {chunk}")
            start = 0 if parts[0] is None else parts[0]
            stop = parts[1]
            step = 1 if len(parts) < 3 or parts[2] is None else parts[2]
            if stop is None:
                raise ValueError(f"range stop is required: {chunk}")
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(chunk))
    if not indices:
        raise ValueError("at least one structure index is required")
    return indices


def parse_csv_choices(value: str, choices: set[str]) -> list[str]:
    parsed = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not parsed:
        raise ValueError("at least one value is required")
    unknown = [name for name in parsed if name not in choices]
    if unknown:
        raise ValueError(
            f"unknown value(s): {', '.join(unknown)}; choices are {sorted(choices)}"
        )
    return parsed


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _force_loss_fn(args: argparse.Namespace) -> torch.nn.Module:
    return modules.WeightedEnergyForcesLoss(
        energy_weight=args.energy_weight,
        forces_weight=args.forces_weight,
    )


def _build_optimizer(
    optimizer_name: str,
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    muon_lr_factor: float,
) -> OptimizerSpec:
    if optimizer_name == "adam":
        return OptimizerSpec(
            name=optimizer_name,
            optimizer=torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
            ),
        )
    if optimizer_name == "hybrid_muon":
        groups, summary = build_hybrid_muon_param_groups(
            model.named_parameters(),
            lr=lr,
            weight_decay=weight_decay,
            muon_weight_decay=0.0,
            muon_lr_factor=muon_lr_factor,
            beta=0.9,
            adam_betas=(0.9, 0.999),
            eps=1.0e-8,
            amsgrad=False,
        )
        return OptimizerSpec(
            name=optimizer_name,
            optimizer=HybridMuon(groups, lr=lr, weight_decay=weight_decay),
            route_summary=summary,
        )
    raise ValueError(f"unknown optimizer {optimizer_name!r}")


def _create_profile_model(args: argparse.Namespace, z_table, device: torch.device):
    cueq_conv_fusion = (
        args.cueq_conv_fusion
        if args.cueq_conv_fusion is not None
        else device.type == "cuda"
    )
    return create_probe_model(
        z_table=z_table,
        cutoff=args.cutoff,
        device=device,
        hidden_channels=args.hidden_channels,
        max_ell=args.max_ell,
        num_interactions=args.num_interactions,
        correlation=args.correlation,
        enable_cueq=args.enable_cueq,
        cueq_conv_fusion=cueq_conv_fusion,
        cueq_optimize_all=args.cueq_optimize_all,
        cueq_optimize_linear=args.cueq_optimize_linear,
        cueq_optimize_channelwise=args.cueq_optimize_channelwise,
        cueq_optimize_symmetric=args.cueq_optimize_symmetric,
        cueq_optimize_fctp=args.cueq_optimize_fctp,
    )


def _edge_force_loss_outputs(
    model: torch.nn.Module,
    batch,
    data: dict[str, torch.Tensor],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
    loss_fn: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    output = _mace_energy_from_vectors(model, data, vectors=vectors, lengths=lengths)
    edge_grad = torch.autograd.grad(
        outputs=[output["energy"]],
        inputs=[vectors],
        grad_outputs=[torch.ones_like(output["energy"])],
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    forces = edge_gradient_to_atomic_forces(
        edge_grad,
        edge_index=edge_index,
        num_atoms=positions.shape[0],
    )
    output = dict(output)
    output["forces"] = forces
    output["virials"] = None
    output["stress"] = None
    loss = loss_fn(pred=output, ref=batch)
    return output["energy"], forces, loss


def _position_snapshot(
    model: torch.nn.Module,
    batch,
    loss_fn: torch.nn.Module,
) -> dict:
    model.zero_grad(set_to_none=True)
    output = model(
        _batch_dict(batch),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    loss = loss_fn(pred=output, ref=batch)
    loss.backward()
    return {
        "energy": output["energy"].detach().clone(),
        "forces": output["forces"].detach().clone(),
        "loss": loss.detach().clone(),
        "grads": _named_parameter_grads(model),
    }


def _edge_snapshot(
    model: torch.nn.Module,
    batch,
    loss_fn: torch.nn.Module,
    executable: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> dict:
    model.zero_grad(set_to_none=True)
    data, positions, edge_index, vectors = _edge_vector_inputs(batch)
    vectors = vectors.detach().clone().requires_grad_(True)
    if executable is None:
        energy, forces, loss = _edge_force_loss_outputs(
            model, batch, data, positions, edge_index, vectors, loss_fn
        )
    else:
        energy, forces, loss = executable(vectors)
    loss.backward()
    return {
        "energy": energy.detach().clone(),
        "forces": forces.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": _named_parameter_grads(model),
    }


def compare_snapshots(left: dict, right: dict, *, atol: float, rtol: float) -> dict:
    failed_checks: list[str] = []
    for key in ("energy", "forces", "loss"):
        if not _within_tolerance(left[key], right[key], atol=atol, rtol=rtol):
            failed_checks.append(key)

    grad_diffs: dict[str, float] = {}
    left_names = set(left["grads"])
    right_names = set(right["grads"])
    for missing_name in sorted(left_names ^ right_names):
        grad_diffs[missing_name] = float("inf")
        failed_checks.append(f"grad:{missing_name}")
    for name in sorted(left_names & right_names):
        left_grad = left["grads"][name]
        right_grad = right["grads"][name]
        if left_grad is None and right_grad is None:
            grad_diffs[name] = 0.0
            continue
        if left_grad is None or right_grad is None:
            grad_diffs[name] = float("inf")
            failed_checks.append(f"grad:{name}")
            continue
        grad_diffs[name] = _max_abs_diff(left_grad, right_grad)
        if not _within_tolerance(left_grad, right_grad, atol=atol, rtol=rtol):
            failed_checks.append(f"grad:{name}")

    return {
        "ok": not failed_checks,
        "atol": atol,
        "rtol": rtol,
        "energy_max_abs_diff": _max_abs_diff(left["energy"], right["energy"]),
        "forces_max_abs_diff": _max_abs_diff(left["forces"], right["forces"]),
        "loss_abs_diff": _max_abs_diff(left["loss"], right["loss"]),
        "param_grad_max_abs_diff": grad_diffs,
        "failed_checks": failed_checks,
    }


def _make_step_mode(
    mode_name: str,
    model: torch.nn.Module,
    batch,
    loss_fn: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> StepModeSpec:
    setup_start = time.perf_counter()
    gate_result = None

    if mode_name == "position_eager":
        def step_fn(optimizer: torch.optim.Optimizer) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                _batch_dict(batch),
                training=True,
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            return loss_fn(pred=output, ref=batch)

        return StepModeSpec(mode_name, step_fn, gate_result, 0.0)

    if mode_name == "edge_eager":
        reference = _position_snapshot(model, batch, loss_fn)
        candidate = _edge_snapshot(model, batch, loss_fn)
        comparison = compare_snapshots(reference, candidate, atol=args.atol, rtol=args.rtol)
        gate_result = {
            "enabled": True,
            "accepted": comparison["ok"],
            "fallback_reason": None if comparison["ok"] else "equivalence_failed",
            "comparison": comparison,
        }

        def step_fn(optimizer: torch.optim.Optimizer) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            data, positions, edge_index, vectors = _edge_vector_inputs(batch)
            vectors = vectors.detach().clone().requires_grad_(True)
            _, _, loss = _edge_force_loss_outputs(
                model, batch, data, positions, edge_index, vectors, loss_fn
            )
            return loss

        return StepModeSpec(
            mode_name,
            step_fn,
            gate_result,
            (time.perf_counter() - setup_start) * 1.0e3,
        )

    if mode_name == "edge_compile":
        data, positions, edge_index, vectors = _edge_vector_inputs(batch)

        def closure(vectors_arg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _edge_force_loss_outputs(
                model, batch, data, positions, edge_index, vectors_arg, loss_fn
            )

        trace_result = trace_force_closure(
            closure,
            (vectors,),
            tracing_mode=args.edge_tracing_mode,
            strip_detach=args.edge_strip_detach,
        )
        executable, compile_kwargs = compile_fx_graph_module(
            trace_result.graph_module,
            compile_graph=args.edge_compile_graph,
            compile_mode=args.edge_compile_mode,
            compile_dynamic=args.edge_compile_dynamic,
        )
        reference = _position_snapshot(model, batch, loss_fn)
        candidate = _edge_snapshot(model, batch, loss_fn, executable=executable)
        comparison = compare_snapshots(reference, candidate, atol=args.atol, rtol=args.rtol)
        gate_result = edge_force_compile_result_from_trace(
            trace_result=trace_result,
            comparison=comparison,
            compile_kwargs=compile_kwargs,
        ).__dict__

        def step_fn(optimizer: torch.optim.Optimizer) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            vectors_arg = vectors.detach().clone().requires_grad_(True)
            _, _, loss = executable(vectors_arg)
            return loss

        return StepModeSpec(
            mode_name,
            step_fn,
            gate_result,
            (time.perf_counter() - setup_start) * 1.0e3,
        )

    raise ValueError(f"unknown mode {mode_name!r}")


def _empty_timing() -> dict[str, list[float]]:
    return {
        "forward_loss_ms": [],
        "backward_clip_ms": [],
        "optimizer_step_ms": [],
        "total_ms": [],
    }


def _record_phase(device: torch.device, timings: dict[str, list[float]], key: str, start: float) -> float:
    _sync(device)
    now = time.perf_counter()
    timings[key].append((now - start) * 1.0e3)
    return now


def profile_case(
    *,
    optimizer_name: str,
    mode_name: str,
    batch,
    z_table,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    torch.manual_seed(args.seed)
    model = _create_profile_model(args, z_table, device)
    loss_fn = _force_loss_fn(args)
    step_mode = _make_step_mode(mode_name, model, batch, loss_fn, args, device)
    model.zero_grad(set_to_none=True)
    optimizer_spec = _build_optimizer(
        optimizer_name,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        muon_lr_factor=args.hybrid_muon_lr_factor,
    )
    optimizer = optimizer_spec.optimizer
    timings = _empty_timing()
    losses: list[float] = []

    for step in range(args.warmup + args.repeats):
        collect = step >= args.warmup
        _sync(device)
        total_start = phase_start = time.perf_counter()

        loss = step_mode.step_fn(optimizer)
        if collect:
            phase_start = _record_phase(device, timings, "forward_loss_ms", phase_start)
        else:
            _sync(device)
            phase_start = time.perf_counter()

        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if collect:
            phase_start = _record_phase(device, timings, "backward_clip_ms", phase_start)
        else:
            _sync(device)
            phase_start = time.perf_counter()

        optimizer.step()
        if collect:
            _record_phase(device, timings, "optimizer_step_ms", phase_start)
            _sync(device)
            timings["total_ms"].append((time.perf_counter() - total_start) * 1.0e3)
            losses.append(float(loss.detach().cpu()))

    summary = {
        key: {
            "mean_ms": mean(values),
            "median_ms": median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for key, values in timings.items()
    }
    total_mean = summary["total_ms"]["mean_ms"]
    total_median = summary["total_ms"]["median_ms"]
    for key, stats in summary.items():
        stats["mean_fraction_pct"] = 100.0 * stats["mean_ms"] / total_mean
        stats["median_fraction_pct"] = 100.0 * stats["median_ms"] / total_median

    payload = {
        "optimizer": optimizer_name,
        "mode": mode_name,
        "setup_ms": step_mode.setup_ms,
        "gate_result": step_mode.gate_result,
        "timings": summary,
        "loss_mean": mean(losses),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "route_summary": optimizer_spec.route_summary,
    }
    if optimizer_spec.route_summary is not None:
        payload["route_text"] = summarize_hybrid_muon_routes(optimizer_spec.route_summary)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xyz",
        type=Path,
        default=Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"),
    )
    parser.add_argument("--indices", default="0:32")
    parser.add_argument("--output", type=Path, default=Path("edge_force_training_steps.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimizers", default="adam,hybrid_muon")
    parser.add_argument("--modes", default="position_eager,edge_eager,edge_compile")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--max-ell", type=int, default=2)
    parser.add_argument("--num-interactions", type=int, default=2)
    parser.add_argument("--correlation", type=int, default=3)
    parser.add_argument("--enable-cueq", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cueq-conv-fusion", dest="cueq_conv_fusion", action="store_true", default=None)
    parser.add_argument("--no-cueq-conv-fusion", dest="cueq_conv_fusion", action="store_false")
    parser.add_argument("--cueq-optimize-all", dest="cueq_optimize_all", action="store_true", default=True)
    parser.add_argument("--no-cueq-optimize-all", dest="cueq_optimize_all", action="store_false")
    parser.add_argument("--cueq-optimize-linear", dest="cueq_optimize_linear", action="store_true", default=False)
    parser.add_argument("--no-cueq-optimize-linear", dest="cueq_optimize_linear", action="store_false")
    parser.add_argument("--cueq-optimize-channelwise", dest="cueq_optimize_channelwise", action="store_true", default=False)
    parser.add_argument("--no-cueq-optimize-channelwise", dest="cueq_optimize_channelwise", action="store_false")
    parser.add_argument("--cueq-optimize-symmetric", dest="cueq_optimize_symmetric", action="store_true", default=False)
    parser.add_argument("--no-cueq-optimize-symmetric", dest="cueq_optimize_symmetric", action="store_false")
    parser.add_argument("--cueq-optimize-fctp", dest="cueq_optimize_fctp", action="store_true", default=False)
    parser.add_argument("--no-cueq-optimize-fctp", dest="cueq_optimize_fctp", action="store_false")
    parser.add_argument("--edge-tracing-mode", choices=["real", "fake", "symbolic"], default="real")
    parser.add_argument("--edge-strip-detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-compile-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--edge-compile-dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=5.0e-7)
    parser.add_argument("--hybrid-muon-lr-factor", type=float, default=0.1)
    parser.add_argument("--energy-weight", type=float, default=40.0)
    parser.add_argument("--forces-weight", type=float, default=1000.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    optimizers = parse_csv_choices(args.optimizers, OPTIMIZER_CHOICES)
    modes = parse_csv_choices(args.modes, MODE_CHOICES)
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(args.xyz, indices, cutoff=args.cutoff, device=device)

    results = [
        profile_case(
            optimizer_name=optimizer_name,
            mode_name=mode_name,
            batch=batch,
            z_table=z_table,
            args=args,
            device=device,
        )
        for optimizer_name in optimizers
        for mode_name in modes
    ]
    payload = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "xyz": str(args.xyz),
        "indices": indices,
        "num_structures": len(indices),
        "num_atoms": int(batch.num_nodes),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "model": {
            "hidden_channels": args.hidden_channels,
            "max_ell": args.max_ell,
            "num_interactions": args.num_interactions,
            "correlation": args.correlation,
        },
        "cueq": {
            "enabled": args.enable_cueq,
            "conv_fusion": args.cueq_conv_fusion,
            "optimize_all": args.cueq_optimize_all,
            "optimize_linear": args.cueq_optimize_linear,
            "optimize_channelwise": args.cueq_optimize_channelwise,
            "optimize_symmetric": args.cueq_optimize_symmetric,
            "optimize_fctp": args.cueq_optimize_fctp,
        },
        "max_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
