from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
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
MODE_CHOICES = {
    "position_eager",
    "edge_eager",
    "edge_compile",
    "edge_compile_compiled_autograd",
    "edge_compile_grads",
    "edge_compile_grads_sequence",
}


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
    grads_materialized: bool = False
    compiled_autograd: bool = False


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


def parse_edge_compile_grad_filter_sequence(value: str) -> list[str]:
    groups = [chunk.strip() for chunk in value.split(":")]
    if not groups or any(not group for group in groups):
        raise ValueError("edge compile grad filter sequence contains an empty group")
    return groups


def _load_compiled_autograd_module():
    try:
        import torch._dynamo.compiled_autograd as compiled_autograd
    except Exception as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError("torch compiled_autograd is unavailable") from exc
    if not hasattr(compiled_autograd, "_enable"):
        raise RuntimeError("torch compiled_autograd._enable is unavailable")
    return compiled_autograd


@contextlib.contextmanager
def _compiled_autograd_context(
    *,
    enabled: bool,
    compile_mode: str,
    compile_dynamic: bool,
):
    if not enabled:
        yield
        return
    compiled_autograd = _load_compiled_autograd_module()

    def compiler_fn(graph_module: torch.fx.GraphModule):
        kwargs = {"mode": compile_mode, "dynamic": compile_dynamic}
        return torch.compile(graph_module, **kwargs)

    with compiled_autograd._enable(compiler_fn, dynamic=compile_dynamic):
        yield


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup_case(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _reset_compile_state() -> None:
    compiler = getattr(torch, "compiler", None)
    reset = getattr(compiler, "reset", None) if compiler is not None else None
    if reset is not None:
        reset()
    try:
        compiled_autograd = _load_compiled_autograd_module()
    except RuntimeError:
        return
    reset_compiled_autograd = getattr(compiled_autograd, "reset", None)
    if reset_compiled_autograd is not None:
        reset_compiled_autograd()


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
    muon_mode: str,
    muon_routing: str,
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
            muon_mode=muon_mode,
            routing=muon_routing,
            module_map=dict(model.named_modules()),
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


def _fx_target_name(target) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if name is not None:
        return str(name)
    return str(target)


def _nested_leaf_count(value) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_nested_leaf_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_nested_leaf_count(item) for item in value.values())
    return 1


def summarize_fx_graph(graph_module: torch.fx.GraphModule) -> dict:
    op_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    output_tensor_count = 0
    nodes = list(graph_module.graph.nodes)
    for node in nodes:
        op_counts[node.op] = op_counts.get(node.op, 0) + 1
        if node.op not in {"placeholder", "output"}:
            key = f"{node.op}:{_fx_target_name(node.target)}"
            target_counts[key] = target_counts.get(key, 0) + 1
        if node.op == "output" and node.args:
            output_tensor_count = _nested_leaf_count(node.args[0])
    return {
        "node_count": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "output_tensor_count": output_tensor_count,
    }


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

def _trainable_named_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _select_named_parameters_by_filter(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...], filter_value: str
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    patterns = tuple(
        chunk.strip() for chunk in filter_value.split(",") if chunk.strip()
    )
    if not patterns:
        return named_parameters
    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if any(pattern in name for pattern in patterns)
    )
    if not selected:
        raise ValueError(
            "edge compile grad filter matched no trainable parameters: "
            f"{filter_value!r}"
        )
    return selected


def _snapshot_with_only_grads(snapshot: dict, names: tuple[str, ...]) -> dict:
    filtered = dict(snapshot)
    filtered["grads"] = {name: snapshot["grads"].get(name) for name in names}
    return filtered


def _edge_force_loss_and_param_grads_outputs(
    model: torch.nn.Module,
    batch,
    data: dict[str, torch.Tensor],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
    loss_fn: torch.nn.Module,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    energy, forces, loss = _edge_force_loss_outputs(
        model, batch, data, positions, edge_index, vectors, loss_fn
    )
    grads = torch.autograd.grad(
        outputs=[loss],
        inputs=parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
        materialize_grads=True,
    )
    return energy, forces, loss, *grads


def _param_grad_snapshot_from_outputs(
    names: tuple[str, ...], grads: tuple[torch.Tensor, ...]
) -> dict[str, torch.Tensor]:
    return {name: grad.detach().clone() for name, grad in zip(names, grads, strict=True)}


def _assign_param_grads(
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    grads: tuple[torch.Tensor, ...],
) -> None:
    for (_name, parameter), grad in zip(named_parameters, grads, strict=True):
        parameter.grad = grad.detach()


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
    executable: Callable[[torch.Tensor], tuple[torch.Tensor, ...]] | None = None,
    *,
    executable_returns_grads: bool = False,
    grad_names: tuple[str, ...] | None = None,
) -> dict:
    model.zero_grad(set_to_none=True)
    data, positions, edge_index, vectors = _edge_vector_inputs(batch)
    vectors = vectors.detach().clone().requires_grad_(True)
    if executable is None:
        energy, forces, loss = _edge_force_loss_outputs(
            model, batch, data, positions, edge_index, vectors, loss_fn
        )
        loss.backward()
        grads = _named_parameter_grads(model)
    else:
        outputs = executable(vectors)
        energy, forces, loss = outputs[:3]
        if executable_returns_grads:
            names = (
                grad_names
                if grad_names is not None
                else tuple(name for name, _param in _trainable_named_parameters(model))
            )
            grads = _param_grad_snapshot_from_outputs(names, tuple(outputs[3:]))
        else:
            loss.backward()
            grads = _named_parameter_grads(model)
    return {
        "energy": energy.detach().clone(),
        "forces": forces.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": grads,
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
            compiled_autograd=mode_name == "edge_compile_compiled_autograd",
        )

    if mode_name in {"edge_compile", "edge_compile_compiled_autograd"}:
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
        gate_result["graph_stats"] = summarize_fx_graph(trace_result.graph_module)
        gate_result["compiled_autograd"] = mode_name == "edge_compile_compiled_autograd"

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

    if mode_name == "edge_compile_grads":
        data, positions, edge_index, vectors = _edge_vector_inputs(batch)
        named_parameters = _select_named_parameters_by_filter(
            _trainable_named_parameters(model), args.edge_compile_grad_filter
        )
        grad_names = tuple(name for name, _parameter in named_parameters)
        parameters = tuple(parameter for _name, parameter in named_parameters)

        def closure(vectors_arg: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return _edge_force_loss_and_param_grads_outputs(
                model, batch, data, positions, edge_index, vectors_arg, loss_fn, parameters
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
        reference = _snapshot_with_only_grads(
            _position_snapshot(model, batch, loss_fn), grad_names
        )
        candidate = _edge_snapshot(
            model,
            batch,
            loss_fn,
            executable=executable,
            executable_returns_grads=True,
            grad_names=grad_names,
        )
        comparison = compare_snapshots(reference, candidate, atol=args.atol, rtol=args.rtol)
        gate_result = edge_force_compile_result_from_trace(
            trace_result=trace_result,
            comparison=comparison,
            compile_kwargs=compile_kwargs,
        ).__dict__
        gate_result["graph_stats"] = summarize_fx_graph(trace_result.graph_module)
        gate_result["compiled_grad_names"] = list(grad_names)
        gate_result["compiled_grad_count"] = len(grad_names)
        gate_result["compiled_grad_filter"] = args.edge_compile_grad_filter

        def step_fn(optimizer: torch.optim.Optimizer) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            vectors_arg = vectors.detach().clone().requires_grad_(True)
            outputs = executable(vectors_arg)
            loss = outputs[2]
            _assign_param_grads(named_parameters, tuple(outputs[3:]))
            return loss

        return StepModeSpec(
            mode_name,
            step_fn,
            gate_result,
            (time.perf_counter() - setup_start) * 1.0e3,
            grads_materialized=True,
        )

    if mode_name == "edge_compile_grads_sequence":
        data, positions, edge_index, vectors = _edge_vector_inputs(batch)
        filter_sequence = parse_edge_compile_grad_filter_sequence(
            args.edge_compile_grad_filter_sequence
        )
        all_named_parameters = _trainable_named_parameters(model)
        reference_full = _position_snapshot(model, batch, loss_fn)
        compiled_steps = []
        substeps = []
        all_grad_names: list[str] = []
        seen_grad_names: set[str] = set()

        for filter_value in filter_sequence:
            named_parameters = _select_named_parameters_by_filter(
                all_named_parameters, filter_value
            )
            grad_names = tuple(name for name, _parameter in named_parameters)
            duplicate_names = sorted(seen_grad_names.intersection(grad_names))
            if duplicate_names:
                raise ValueError(
                    "edge compile grad filter sequence selects duplicate parameters: "
                    f"{duplicate_names}"
                )
            seen_grad_names.update(grad_names)
            all_grad_names.extend(grad_names)
            parameters = tuple(parameter for _name, parameter in named_parameters)

            def make_closure(
                selected_parameters: tuple[torch.nn.Parameter, ...]
            ) -> Callable[[torch.Tensor], tuple[torch.Tensor, ...]]:
                def closure(vectors_arg: torch.Tensor) -> tuple[torch.Tensor, ...]:
                    return _edge_force_loss_and_param_grads_outputs(
                        model,
                        batch,
                        data,
                        positions,
                        edge_index,
                        vectors_arg,
                        loss_fn,
                        selected_parameters,
                    )

                return closure

            trace_result = trace_force_closure(
                make_closure(parameters),
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
            reference = _snapshot_with_only_grads(reference_full, grad_names)
            candidate = _edge_snapshot(
                model,
                batch,
                loss_fn,
                executable=executable,
                executable_returns_grads=True,
                grad_names=grad_names,
            )
            comparison = compare_snapshots(
                reference, candidate, atol=args.atol, rtol=args.rtol
            )
            sub_gate = edge_force_compile_result_from_trace(
                trace_result=trace_result,
                comparison=comparison,
                compile_kwargs=compile_kwargs,
            ).__dict__
            sub_gate["graph_stats"] = summarize_fx_graph(trace_result.graph_module)
            sub_gate["compiled_grad_names"] = list(grad_names)
            sub_gate["compiled_grad_count"] = len(grad_names)
            sub_gate["compiled_grad_filter"] = filter_value
            substeps.append(sub_gate)
            compiled_steps.append((named_parameters, executable))

        accepted = all(bool(substep.get("accepted", False)) for substep in substeps)
        gate_result = {
            "enabled": True,
            "accepted": accepted,
            "fallback_reason": None if accepted else "sequence_equivalence_failed",
            "compiled_grad_filter_sequence": filter_sequence,
            "compiled_grad_names": all_grad_names,
            "compiled_grad_count": len(all_grad_names),
            "substeps": substeps,
            "graph_stats": {
                "node_count": sum(
                    substep["graph_stats"]["node_count"] for substep in substeps
                ),
                "output_tensor_count": sum(
                    substep["graph_stats"]["output_tensor_count"]
                    for substep in substeps
                ),
                "subgraph_count": len(substeps),
            },
        }

        def step_fn(optimizer: torch.optim.Optimizer) -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = None
            for named_parameters, executable in compiled_steps:
                vectors_arg = vectors.detach().clone().requires_grad_(True)
                outputs = executable(vectors_arg)
                if loss is None:
                    loss = outputs[2]
                _assign_param_grads(named_parameters, tuple(outputs[3:]))
            if loss is None:
                raise RuntimeError("empty edge compile grad sequence")
            return loss

        return StepModeSpec(
            mode_name,
            step_fn,
            gate_result,
            (time.perf_counter() - setup_start) * 1.0e3,
            grads_materialized=True,
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
        muon_mode=args.hybrid_muon_mode,
        muon_routing=args.hybrid_muon_routing,
    )
    optimizer = optimizer_spec.optimizer
    timings = _empty_timing()
    losses: list[float] = []

    for step in range(args.warmup + args.repeats):
        collect = step >= args.warmup
        _sync(device)
        total_start = phase_start = time.perf_counter()

        with _compiled_autograd_context(
            enabled=step_mode.compiled_autograd,
            compile_mode=args.edge_compile_mode,
            compile_dynamic=args.edge_compile_dynamic,
        ):
            loss = step_mode.step_fn(optimizer)
            if collect:
                phase_start = _record_phase(device, timings, "forward_loss_ms", phase_start)
            else:
                _sync(device)
                phase_start = time.perf_counter()

            if not step_mode.grads_materialized:
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



def _append_bool_flag(command: list[str], name: str, value: bool) -> None:
    command.append(f"--{name}" if value else f"--no-{name}")


def _append_optional_bool_flag(
    command: list[str], name: str, value: bool | None
) -> None:
    if value is not None:
        _append_bool_flag(command, name, value)


def _case_worker_command(
    args: argparse.Namespace,
    *,
    optimizer_name: str,
    mode_name: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--case-worker",
        "--xyz",
        str(args.xyz),
        "--indices",
        args.indices,
        "--output",
        str(output),
        "--device",
        args.device,
        "--optimizers",
        optimizer_name,
        "--modes",
        mode_name,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--cutoff",
        str(args.cutoff),
        "--hidden-channels",
        str(args.hidden_channels),
        "--max-ell",
        str(args.max_ell),
        "--num-interactions",
        str(args.num_interactions),
        "--correlation",
        str(args.correlation),
        "--edge-tracing-mode",
        args.edge_tracing_mode,
        "--edge-compile-mode",
        args.edge_compile_mode,
        "--edge-compile-grad-filter",
        args.edge_compile_grad_filter,
        "--edge-compile-grad-filter-sequence",
        args.edge_compile_grad_filter_sequence,
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--hybrid-muon-lr-factor",
        str(args.hybrid_muon_lr_factor),
        "--energy-weight",
        str(args.energy_weight),
        "--forces-weight",
        str(args.forces_weight),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--seed",
        str(args.seed),
    ]
    _append_bool_flag(command, "enable-cueq", args.enable_cueq)
    _append_optional_bool_flag(command, "cueq-conv-fusion", args.cueq_conv_fusion)
    _append_bool_flag(command, "cueq-optimize-all", args.cueq_optimize_all)
    _append_bool_flag(command, "cueq-optimize-linear", args.cueq_optimize_linear)
    _append_bool_flag(
        command, "cueq-optimize-channelwise", args.cueq_optimize_channelwise
    )
    _append_bool_flag(command, "cueq-optimize-symmetric", args.cueq_optimize_symmetric)
    _append_bool_flag(command, "cueq-optimize-fctp", args.cueq_optimize_fctp)
    _append_bool_flag(command, "edge-strip-detach", args.edge_strip_detach)
    _append_bool_flag(command, "edge-compile-graph", args.edge_compile_graph)
    _append_bool_flag(command, "edge-compile-dynamic", args.edge_compile_dynamic)
    return command


def _run_isolated_cases(
    args: argparse.Namespace,
    *,
    optimizers: list[str],
    modes: list[str],
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_outputs: list[str] = []
    results: list[dict] = []
    merged_payload: dict | None = None
    for optimizer_name in optimizers:
        for mode_name in modes:
            case_output = args.output.with_name(
                f"{args.output.stem}_{optimizer_name}_{mode_name}{args.output.suffix}"
            )
            command = _case_worker_command(
                args,
                optimizer_name=optimizer_name,
                mode_name=mode_name,
                output=case_output,
            )
            print(
                f"[edge-force-step] launching isolated optimizer={optimizer_name} "
                f"mode={mode_name}",
                file=sys.stderr,
                flush=True,
            )
            subprocess.run(command, check=True)
            case_payload = json.loads(case_output.read_text())
            case_outputs.append(str(case_output))
            results.extend(case_payload["results"])
            if merged_payload is None:
                merged_payload = dict(case_payload)

    if merged_payload is None:
        raise RuntimeError("no isolated cases were executed")
    merged_payload["results"] = results
    merged_payload["case_outputs"] = case_outputs
    args.output.write_text(json.dumps(merged_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(merged_payload, indent=2, sort_keys=True))

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
    parser.add_argument(
        "--edge-compile-grad-filter",
        default="",
        help="Comma-separated substrings selecting trainable parameters for edge_compile_grads diagnostics.",
    )
    parser.add_argument(
        "--edge-compile-grad-filter-sequence",
        default="readouts:conv_tp_weights",
        help=(
            "Colon-separated edge_compile_grads filters to compile and execute "
            "sequentially as a diagnostic for repeated force-graph replay."
        ),
    )
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--weight-decay", type=float, default=5.0e-7)
    parser.add_argument("--hybrid-muon-lr-factor", type=float, default=0.1)
    parser.add_argument("--hybrid-muon-mode", choices=("2d", "slice"), default="2d")
    parser.add_argument("--hybrid-muon-routing", choices=("mace", "tace"), default="mace")
    parser.add_argument("--energy-weight", type=float, default=40.0)
    parser.add_argument("--forces-weight", type=float, default=1000.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--case-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    optimizers = parse_csv_choices(args.optimizers, OPTIMIZER_CHOICES)
    modes = parse_csv_choices(args.modes, MODE_CHOICES)
    if not args.case_worker and len(optimizers) * len(modes) > 1:
        _run_isolated_cases(args, optimizers=optimizers, modes=modes)
        return

    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    indices = parse_indices(args.indices)
    batch, z_table = _load_batch(args.xyz, indices, cutoff=args.cutoff, device=device)

    results = []
    for optimizer_name in optimizers:
        for mode_name in modes:
            print(
                f"[edge-force-step] starting optimizer={optimizer_name} mode={mode_name}",
                file=sys.stderr,
                flush=True,
            )
            _reset_compile_state()
            results.append(
                profile_case(
                    optimizer_name=optimizer_name,
                    mode_name=mode_name,
                    batch=batch,
                    z_table=z_table,
                    args=args,
                    device=device,
                )
            )
            _cleanup_case(device)
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
