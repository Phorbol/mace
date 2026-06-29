from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

import torch

from mace import modules
from mace.modules.utils import get_edge_vectors_and_lengths, get_outputs, prepare_graph
from mace.tools import compile as mace_compile
from mace.tools.force_compile import (
    compile_fx_graph_module,
    disable_functorch_donated_buffer,
    edge_gradient_to_atomic_forces,
    trace_force_closure,
)
from mace.tools.scatter import scatter_sum


@dataclasses.dataclass(frozen=True)
class EdgeForceCompileConfig:
    enabled: bool = False
    tracing_mode: str = "real"
    strip_detach: bool = True
    compile_graph: bool = True
    compile_mode: str = "default"
    compile_dynamic: bool = True
    allow_fallback: bool = True
    atol: float = 1.0e-5
    rtol: float = 1.0e-4
    cache_hit_gate: bool = True
    cache_policy: str = "repeat_only"
    min_repeats: int = 2
    disable_negative_speedup: bool = True
    negative_speedup_min_steps: int = 4


@dataclasses.dataclass(frozen=True)
class EdgeForceCompileGateResult:
    enabled: bool
    accepted: bool
    fallback_reason: str | None
    detach_nodes_before: int | None = None
    detach_nodes_after: int | None = None
    node_count: int | None = None
    comparison: dict[str, Any] | None = None
    compile_kwargs: dict[str, Any] | None = None
    cache_hit: bool | None = None
    cache_key: list[Any] | None = None


@dataclasses.dataclass(frozen=True)
class EdgeForceCachePolicyDecision:
    cache_policy: str
    compile_allowed: bool
    reason: str | None
    seen_count: int
    compile_count: int
    cache_hit_count: int
    disabled: bool = False


@dataclasses.dataclass
class EdgeForceCacheEntryStats:
    seen_count: int = 0
    compile_count: int = 0
    cache_hit_count: int = 0
    disabled_reason: str | None = None
    compile_setup_seconds: float = 0.0
    compiled_step_seconds_ema: float | None = None
    eager_step_seconds_ema: float | None = None


@dataclasses.dataclass
class EdgeForceCachePolicyState:
    entries: dict[tuple, EdgeForceCacheEntryStats] = dataclasses.field(
        default_factory=dict
    )

    def stats_for(self, cache_key: tuple) -> EdgeForceCacheEntryStats:
        return self.entries.setdefault(cache_key, EdgeForceCacheEntryStats())

    def record_and_decide(
        self,
        cache_key: tuple,
        *,
        policy: str,
        min_repeats: int,
        cache_hit: bool = False,
    ) -> EdgeForceCachePolicyDecision:
        stats = self.stats_for(cache_key)
        stats.seen_count += 1
        if cache_hit:
            stats.cache_hit_count += 1
        if stats.disabled_reason is not None:
            return EdgeForceCachePolicyDecision(
                cache_policy=policy,
                compile_allowed=False,
                reason=stats.disabled_reason,
                seen_count=stats.seen_count,
                compile_count=stats.compile_count,
                cache_hit_count=stats.cache_hit_count,
                disabled=True,
            )
        if policy == "shape":
            allowed = True
            reason = None
        elif policy == "repeat_only":
            allowed = stats.seen_count >= max(1, int(min_repeats))
            reason = None if allowed else "min_repeats"
        elif policy == "bucket":
            allowed = True
            reason = None
        else:
            raise ValueError(f"unknown edge-force cache policy: {policy}")
        return EdgeForceCachePolicyDecision(
            cache_policy=policy,
            compile_allowed=allowed,
            reason=reason,
            seen_count=stats.seen_count,
            compile_count=stats.compile_count,
            cache_hit_count=stats.cache_hit_count,
            disabled=False,
        )

    def record_compile(self, cache_key: tuple, *, setup_seconds: float) -> None:
        stats = self.stats_for(cache_key)
        stats.compile_count += 1
        stats.compile_setup_seconds += float(setup_seconds)

    def record_step_time(
        self,
        cache_key: tuple,
        *,
        compiled: bool,
        seconds: float,
        ema_decay: float = 0.9,
    ) -> None:
        stats = self.stats_for(cache_key)
        value = float(seconds)
        if compiled:
            old = stats.compiled_step_seconds_ema
            stats.compiled_step_seconds_ema = (
                value if old is None else ema_decay * old + (1.0 - ema_decay) * value
            )
        else:
            old = stats.eager_step_seconds_ema
            stats.eager_step_seconds_ema = (
                value if old is None else ema_decay * old + (1.0 - ema_decay) * value
            )

    def disable(self, cache_key: tuple, reason: str) -> None:
        self.stats_for(cache_key).disabled_reason = reason


_EDGE_FORCE_INPUT_KEYS = ("positions", "edge_index", "node_attrs", "batch", "ptr", "head")


def edge_force_compile_input_names(data_keys) -> tuple[str, ...]:
    keys = set(data_keys)
    return tuple(name for name in _EDGE_FORCE_INPUT_KEYS if name in keys)


def edge_force_compile_shape_cache_key(
    *, num_atoms: int, num_edges: int, input_shapes: dict[str, tuple[int, ...]]
) -> tuple:
    return (
        "shape",
        int(num_atoms),
        int(num_edges),
        tuple((name, tuple(shape)) for name, shape in sorted(input_shapes.items())),
    )


def edge_force_cache_hit_gate_result(
    *, comparison: dict[str, Any], cache_key: tuple
) -> EdgeForceCompileGateResult:
    accepted = bool(comparison.get("ok", False))
    return EdgeForceCompileGateResult(
        enabled=True,
        accepted=accepted,
        fallback_reason=None if accepted else "cache_hit_equivalence_failed",
        comparison=comparison,
        cache_hit=True,
        cache_key=list(cache_key),
    )


def edge_force_compile_gate(
    *,
    model,
    batch,
    config: EdgeForceCompileConfig,
    compute_virials: bool = False,
    compute_stress: bool = False,
    compute_displacement: bool = False,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
    compute_atomic_stresses: bool = False,
):
    if not config.enabled:
        return EdgeForceCompileGateResult(
            enabled=False,
            accepted=False,
            fallback_reason="disabled",
        )
    if any(
        (
            compute_virials,
            compute_stress,
            compute_displacement,
            compute_hessian,
            compute_edge_forces,
            compute_atomic_stresses,
        )
    ):
        return EdgeForceCompileGateResult(
            enabled=True,
            accepted=False,
            fallback_reason="unsupported_outputs",
        )
    return EdgeForceCompileGateResult(
        enabled=True,
        accepted=False,
        fallback_reason="gate_not_run",
    )


def edge_force_compile_result_from_trace(
    *,
    trace_result,
    comparison: dict[str, Any],
    compile_kwargs: dict[str, Any] | None,
) -> EdgeForceCompileGateResult:
    accepted = bool(comparison.get("ok", False))
    return EdgeForceCompileGateResult(
        enabled=True,
        accepted=accepted,
        fallback_reason=None if accepted else "equivalence_failed",
        detach_nodes_before=trace_result.detach_nodes_before,
        detach_nodes_after=trace_result.detach_nodes_after,
        node_count=len(list(trace_result.graph_module.graph.nodes)),
        comparison=comparison,
        compile_kwargs=compile_kwargs,
    )



def _canonical_parameter_name(name: str) -> str:
    return name.replace("._orig_mod", "").replace("_orig_mod.", "")


def _named_parameter_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        _canonical_parameter_name(name): (
            None if param.grad is None else param.grad.detach().clone()
        )
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left.detach() - right.detach()).abs().max().cpu())


def _within_tolerance(
    left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float
) -> bool:
    if left.shape != right.shape:
        return False
    return bool(torch.allclose(left.detach(), right.detach(), atol=atol, rtol=rtol))


def _compare_edge_force_snapshots(
    left: dict[str, Any], right: dict[str, Any], *, atol: float, rtol: float
) -> dict[str, Any]:
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


def _mace_energy_from_edge_vectors(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    *,
    vectors: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not isinstance(model, modules.ScaleShiftMACE):
        raise TypeError("edge-force compiled loss currently supports ScaleShiftMACE only")

    num_graphs = int(data["ptr"].numel() - 1)
    num_atoms_arange = torch.arange(data["positions"].shape[0], device=vectors.device)
    node_heads = (
        data["head"][data["batch"]]
        if "head" in data
        else torch.zeros_like(data["batch"])
    ).to(torch.int64)

    node_e0 = model.atomic_energies_fn(data["node_attrs"])[
        num_atoms_arange.to(torch.int64), node_heads
    ]
    e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs).to(
        vectors.dtype
    )

    node_feats = model.node_embedding(data["node_attrs"])
    edge_attrs = model.spherical_harmonics(vectors)
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )

    if hasattr(model, "pair_repulsion"):
        pair_node_energy = model.pair_repulsion_fn(
            lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
        )
    else:
        pair_node_energy = torch.zeros_like(node_e0)

    if hasattr(model, "joint_embedding"):
        embedding_features: dict[str, torch.Tensor] = {}
        for name, _ in model.embedding_specs.items():
            embedding_features[name] = data[name]
        node_feats = node_feats + model.joint_embedding(data["batch"], embedding_features)
        if hasattr(model, "embedding_readout"):
            embedding_node_energy = torch.atleast_1d(
                model.embedding_readout(node_feats, node_heads).squeeze(-1)
            )
            embedding_energy = scatter_sum(
                src=embedding_node_energy,
                index=data["batch"],
                dim=0,
                dim_size=num_graphs,
            )
            e0 = e0 + embedding_energy

    node_es_list = [pair_node_energy]
    node_feats_list: list[torch.Tensor] = []
    for i, (interaction, product) in enumerate(zip(model.interactions, model.products)):
        node_feats, sc = interaction(
            node_attrs=data["node_attrs"],
            node_feats=node_feats,
            edge_attrs=edge_attrs,
            edge_feats=edge_feats,
            edge_index=data["edge_index"],
            cutoff=cutoff,
            first_layer=(i == 0),
        )
        node_feats = product(
            node_feats=node_feats,
            sc=sc,
            node_attrs=data["node_attrs"],
        )
        node_feats_list.append(node_feats)

    for i, readout in enumerate(model.readouts):
        feat_idx = -1 if len(model.readouts) == 1 else i
        node_es_list.append(
            readout(node_feats_list[feat_idx], node_heads)[
                num_atoms_arange.to(torch.int64), node_heads
            ]
        )

    node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
    node_inter_es = model.scale_shift(node_inter_es, node_heads)
    inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)
    total_energy = e0 + inter_e
    node_energy = node_e0.clone().double() + node_inter_es.clone().double()
    return {
        "energy": total_energy,
        "node_energy": node_energy,
        "interaction_energy": inter_e,
    }


def _edge_force_outputs(
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    output = _mace_energy_from_edge_vectors(
        model, data_dict, vectors=vectors, lengths=lengths
    )
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
    return output["energy"], forces


def _loss_from_energy_forces(
    *, batch, loss_fn, energy: torch.Tensor, forces: torch.Tensor
) -> torch.Tensor:
    output = {
        "energy": energy,
        "forces": forces,
        "virials": None,
        "stress": None,
    }
    return loss_fn(pred=output, ref=batch)


def _edge_vector_inputs(batch) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    data = batch.to_dict()
    positions = data["positions"].detach()
    edge_index = data["edge_index"]
    vectors, _ = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=data["shifts"].detach(),
    )
    vectors = vectors.detach().clone().requires_grad_(True)
    return data, positions, edge_index, vectors


def _edge_force_snapshot_from_executable(
    *,
    model: torch.nn.Module,
    batch,
    loss_fn,
    executable,
    input_names: tuple[str, ...],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    vectors = vectors.detach().clone().requires_grad_(True)
    inputs = [data_dict[name] for name in input_names]
    energy, forces = executable(vectors, *inputs)
    loss = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    loss.backward()
    return {
        "energy": energy.detach().clone(),
        "forces": forces.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": _named_parameter_grads(model),
    }


def _edge_force_value_snapshot_from_executable(
    *,
    model: torch.nn.Module,
    batch,
    loss_fn,
    executable,
    input_names: tuple[str, ...],
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    vectors = vectors.detach().clone().requires_grad_(True)
    inputs = [data_dict[name] for name in input_names]
    energy, forces = executable(vectors, *inputs)
    loss = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    return {
        "energy": energy.detach().clone(),
        "forces": forces.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": {},
    }


def _position_force_snapshot(
    *, model: torch.nn.Module, batch, loss_fn
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    output = model(
        batch.to_dict(),
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


def _position_force_value_snapshot(
    *, model: torch.nn.Module, batch, loss_fn
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    output = model(
        batch.to_dict(),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    loss = loss_fn(pred=output, ref=batch)
    return {
        "energy": output["energy"].detach().clone(),
        "forces": output["forces"].detach().clone(),
        "loss": loss.detach().clone(),
        "grads": {},
    }


@dataclasses.dataclass
class _CompiledEdgeForceStep:
    executable: Any
    gate_result: EdgeForceCompileGateResult
    cache_key: tuple
    input_names: tuple[str, ...]


class EdgeForceCompiledLossModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, config: EdgeForceCompileConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.cache: dict[tuple, _CompiledEdgeForceStep] = {}
        self.cache_policy_state = EdgeForceCachePolicyState()
        self.disabled = False
        self.functorch_donated_buffer_disabled = (
            disable_functorch_donated_buffer() if config.compile_graph else False
        )

    def disable_compile_fallback(self, exc: Exception) -> bool:
        if not self.config.allow_fallback:
            return False
        logging.warning(
            "edge-force compiled loss failed; disabling compiled force loss and "
            "continuing eager: %s",
            exc,
        )
        self.disabled = True
        return True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _eager_force_loss(
        self,
        *,
        batch,
        loss_fn,
        disabled_reason: str = "disabled",
        policy_decision: EdgeForceCachePolicyDecision | None = None,
    ):
        output = self.model(
            batch.to_dict(),
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        metrics = {
            "edge_force_compile": False,
            "edge_force_compile_disabled": True,
            "edge_force_compile_disabled_reason": disabled_reason,
        }
        if policy_decision is not None:
            metrics.update(
                {
                    "edge_force_compile_cache_policy": policy_decision.cache_policy,
                    "edge_force_cache_seen_count": policy_decision.seen_count,
                    "edge_force_cache_compile_count": policy_decision.compile_count,
                    "edge_force_cache_hit_count": policy_decision.cache_hit_count,
                }
            )
        return loss_fn(pred=output, ref=batch), metrics

    def _cache_key(
        self,
        *,
        batch,
        input_names: tuple[str, ...],
        data_dict: dict[str, torch.Tensor],
    ) -> tuple:
        return edge_force_compile_shape_cache_key(
            num_atoms=batch.positions.shape[0],
            num_edges=batch.edge_index.shape[1],
            input_shapes={name: tuple(data_dict[name].shape) for name in input_names},
        )

    def _compile_step(self, *, batch, loss_fn, cache_key: tuple) -> _CompiledEdgeForceStep:
        data_dict, _, _, vectors = _edge_vector_inputs(batch)
        input_names = edge_force_compile_input_names(data_dict.keys())
        example_inputs = tuple(data_dict[name] for name in input_names)

        def closure(vectors_arg: torch.Tensor, *input_tensors: torch.Tensor):
            current_data = dict(data_dict)
            current_data.update(zip(input_names, input_tensors, strict=True))
            return _edge_force_outputs(
                self.model,
                current_data,
                current_data["positions"],
                current_data["edge_index"],
                vectors_arg,
            )

        trace_result = trace_force_closure(
            closure,
            (vectors, *example_inputs),
            tracing_mode=self.config.tracing_mode,
            strip_detach=self.config.strip_detach,
        )
        executable, compile_kwargs = compile_fx_graph_module(
            trace_result.graph_module,
            compile_graph=self.config.compile_graph,
            compile_mode=self.config.compile_mode,
            compile_dynamic=self.config.compile_dynamic,
        )
        if self.config.compile_graph:
            reference = _position_force_value_snapshot(
                model=self.model,
                batch=batch,
                loss_fn=loss_fn,
            )
            candidate = _edge_force_value_snapshot_from_executable(
                model=self.model,
                batch=batch,
                loss_fn=loss_fn,
                executable=executable,
                input_names=input_names,
            )
        else:
            reference = _position_force_snapshot(
                model=self.model,
                batch=batch,
                loss_fn=loss_fn,
            )
            candidate = _edge_force_snapshot_from_executable(
                model=self.model,
                batch=batch,
                loss_fn=loss_fn,
                executable=executable,
                input_names=input_names,
            )
        comparison = _compare_edge_force_snapshots(
            reference,
            candidate,
            atol=self.config.atol,
            rtol=self.config.rtol,
        )
        gate_result = edge_force_compile_result_from_trace(
            trace_result=trace_result,
            comparison=comparison,
            compile_kwargs=compile_kwargs,
        )
        if not gate_result.accepted:
            raise RuntimeError(f"edge-force compile gate failed: {comparison}")

        training_executable = executable
        if self.config.compile_graph:
            # The gate runs a backward pass through the compiled callable.  Some
            # Inductor/AOTAutograd graphs keep saved-tensor state on the callable,
            # so cache a fresh executable for the real optimizer step.
            training_executable, _ = compile_fx_graph_module(
                trace_result.graph_module,
                compile_graph=self.config.compile_graph,
                compile_mode=self.config.compile_mode,
                compile_dynamic=self.config.compile_dynamic,
            )

        self.model.zero_grad(set_to_none=True)
        return _CompiledEdgeForceStep(
            executable=training_executable,
            gate_result=gate_result,
            cache_key=cache_key,
            input_names=input_names,
        )

    def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
        del output_args
        if self.disabled:
            return self._eager_force_loss(batch=batch, loss_fn=loss_fn)
        try:
            data_dict, _, _, vectors = _edge_vector_inputs(batch)
            input_names = edge_force_compile_input_names(data_dict.keys())
            cache_key = self._cache_key(
                batch=batch,
                input_names=input_names,
                data_dict=data_dict,
            )
            compiled = self.cache.get(cache_key)
            cache_hit = compiled is not None
            policy_decision = self.cache_policy_state.record_and_decide(
                cache_key,
                policy=self.config.cache_policy,
                min_repeats=self.config.min_repeats,
                cache_hit=cache_hit,
            )
            if not policy_decision.compile_allowed:
                eager_start = time.perf_counter()
                loss, metrics = self._eager_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    disabled_reason=policy_decision.reason or "policy_disabled",
                    policy_decision=policy_decision,
                )
                self.cache_policy_state.record_step_time(
                    cache_key,
                    compiled=False,
                    seconds=time.perf_counter() - eager_start,
                )
                stats = self.cache_policy_state.stats_for(cache_key)
                metrics.update(
                    {
                        "edge_force_compile_setup_seconds": stats.compile_setup_seconds,
                        "edge_force_compiled_step_seconds_ema": stats.compiled_step_seconds_ema,
                        "edge_force_eager_step_seconds_ema": stats.eager_step_seconds_ema,
                    }
                )
                return loss, metrics
            cache_hit_gate_accepted = None
            if compiled is None:
                setup_start = time.perf_counter()
                compiled = self._compile_step(
                    batch=batch,
                    loss_fn=loss_fn,
                    cache_key=cache_key,
                )
                setup_seconds = time.perf_counter() - setup_start
                self.cache_policy_state.record_compile(
                    cache_key,
                    setup_seconds=setup_seconds,
                )
                self.cache[cache_key] = compiled
            elif compiled.input_names != input_names:
                raise RuntimeError(
                    "edge-force compile cache input mismatch: "
                    f"{compiled.input_names} != {input_names}"
                )
            elif self.config.cache_hit_gate:
                if self.config.compile_graph:
                    reference = _position_force_value_snapshot(
                        model=self.model,
                        batch=batch,
                        loss_fn=loss_fn,
                    )
                    candidate = _edge_force_value_snapshot_from_executable(
                        model=self.model,
                        batch=batch,
                        loss_fn=loss_fn,
                        executable=compiled.executable,
                        input_names=compiled.input_names,
                    )
                else:
                    reference = _position_force_snapshot(
                        model=self.model,
                        batch=batch,
                        loss_fn=loss_fn,
                    )
                    candidate = _edge_force_snapshot_from_executable(
                        model=self.model,
                        batch=batch,
                        loss_fn=loss_fn,
                        executable=compiled.executable,
                        input_names=compiled.input_names,
                    )
                comparison = _compare_edge_force_snapshots(
                    reference,
                    candidate,
                    atol=self.config.atol,
                    rtol=self.config.rtol,
                )
                gate_result = edge_force_cache_hit_gate_result(
                    comparison=comparison,
                    cache_key=compiled.cache_key,
                )
                cache_hit_gate_accepted = gate_result.accepted
                if not gate_result.accepted:
                    raise RuntimeError(
                        f"edge-force compile cache-hit gate failed: {comparison}"
                    )
                self.model.zero_grad(set_to_none=True)
            compiled_start = time.perf_counter()
            vectors = vectors.detach().clone().requires_grad_(True)
            inputs = [data_dict[name] for name in compiled.input_names]
            energy, forces = compiled.executable(vectors, *inputs)
            loss = _loss_from_energy_forces(
                batch=batch,
                loss_fn=loss_fn,
                energy=energy,
                forces=forces,
            )
            self.cache_policy_state.record_step_time(
                cache_key,
                compiled=True,
                seconds=time.perf_counter() - compiled_start,
            )
            stats = self.cache_policy_state.stats_for(cache_key)
            if (
                self.config.disable_negative_speedup
                and stats.cache_hit_count >= self.config.negative_speedup_min_steps
                and stats.compiled_step_seconds_ema is not None
                and stats.eager_step_seconds_ema is not None
                and stats.compiled_step_seconds_ema >= stats.eager_step_seconds_ema
            ):
                self.cache_policy_state.disable(cache_key, "negative_speedup")
            metrics = {
                "edge_force_compile": True,
                "edge_force_cache_hit": cache_hit,
                "edge_force_gate_accepted": compiled.gate_result.accepted,
                "edge_force_compile_cache_policy": policy_decision.cache_policy,
                "edge_force_cache_seen_count": policy_decision.seen_count,
                "edge_force_cache_compile_count": stats.compile_count,
                "edge_force_cache_hit_count": stats.cache_hit_count,
                "edge_force_compile_setup_seconds": stats.compile_setup_seconds,
                "edge_force_compiled_step_seconds_ema": stats.compiled_step_seconds_ema,
                "edge_force_eager_step_seconds_ema": stats.eager_step_seconds_ema,
            }
            if self.config.compile_graph:
                metrics["_retain_graph_for_backward"] = True
            if cache_hit_gate_accepted is not None:
                metrics["edge_force_cache_hit_gate_accepted"] = cache_hit_gate_accepted
            return loss, metrics
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self._eager_force_loss(batch=batch, loss_fn=loss_fn)


class RuntimeFallbackCompiledModule(torch.nn.Module):
    def __init__(
        self,
        *,
        eager_model: torch.nn.Module,
        compiled_model: torch.nn.Module,
        allow_fallback: bool,
    ) -> None:
        super().__init__()
        self.eager_model = eager_model
        self.__dict__["compiled_model"] = compiled_model
        self.allow_fallback = allow_fallback
        self.disabled = False

    def disable_compile_fallback(self, exc: Exception) -> bool:
        if not self.allow_fallback:
            return False
        logging.warning(
            "training torch.compile failed during backward; disabling compiled "
            "training model and retrying eager: %s",
            exc,
        )
        self.disabled = True
        return True

    def forward(self, *args, **kwargs):
        if self.disabled:
            return self.eager_model(*args, **kwargs)
        try:
            return self.compiled_model(*args, **kwargs)
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self.eager_model(*args, **kwargs)


class EnergyOnlyForceCompiledModule(RuntimeFallbackCompiledModule):
    def forward(self, data, *args, **kwargs):
        compute_force = kwargs.get("compute_force", True)
        unsupported_force_outputs = any(
            kwargs.get(name, False)
            for name in (
                "compute_virials",
                "compute_stress",
                "compute_displacement",
                "compute_hessian",
                "compute_edge_forces",
                "compute_atomic_stresses",
                "lammps_mliap",
            )
        )
        if self.disabled or unsupported_force_outputs:
            return self.eager_model(data, *args, **kwargs)

        try:
            if not compute_force:
                return self.compiled_model(data, *args, **kwargs)
            if "positions" in data:
                data["positions"].requires_grad_(True)
            energy_kwargs = dict(kwargs)
            energy_kwargs.update(
                {
                    "compute_force": False,
                    "compute_virials": False,
                    "compute_stress": False,
                    "compute_displacement": False,
                    "compute_hessian": False,
                    "compute_edge_forces": False,
                    "compute_atomic_stresses": False,
                }
            )
            output = self.compiled_model(data, *args, **energy_kwargs)
            ctx = prepare_graph(data)
            forces, _, _, _, _ = get_outputs(
                energy=output["energy"],
                positions=ctx.positions,
                displacement=ctx.displacement,
                vectors=ctx.vectors,
                cell=ctx.cell,
                training=kwargs.get("training", False),
                compute_force=True,
                compute_virials=False,
                compute_stress=False,
            )
            output = dict(output)
            output.update(
                {
                    "forces": forces,
                    "virials": None,
                    "stress": None,
                    "hessian": None,
                    "edge_forces": None,
                }
            )
            return output
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self.eager_model(data, *args, **kwargs)


def prepare_edge_force_compiled_loss(
    model: torch.nn.Module,
    *,
    config: EdgeForceCompileConfig,
) -> torch.nn.Module:
    if not config.enabled:
        return model
    if not isinstance(model, modules.ScaleShiftMACE):
        message = (
            "edge-force compiled loss currently supports ScaleShiftMACE only; "
            f"got {type(model).__name__}"
        )
        if config.allow_fallback:
            logging.warning("%s; continuing with eager training", message)
            return model
        raise TypeError(message)
    return EdgeForceCompiledLossModule(model, config=config)


def prepare_model_for_training_compile(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str,
    fullgraph: bool,
    allow_fallback: bool,
) -> torch.nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        message = "torch.compile is unavailable in this PyTorch build"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message)

    try:
        mace_compile.configure_autograd_for_compile(allow_autograd=True)
        import torch._dynamo.config as dynamo_config

        dynamo_config.optimize_ddp = False
        if allow_fallback:
            dynamo_config.suppress_errors = True
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logging.info(
            "Enabled training torch.compile: mode=%s fullgraph=%s",
            mode,
            fullgraph,
        )
        return EnergyOnlyForceCompiledModule(
            eager_model=model,
            compiled_model=compiled,
            allow_fallback=allow_fallback,
        )
    except Exception as exc:
        message = f"training torch.compile setup failed: {exc}"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message) from exc
