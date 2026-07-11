from __future__ import annotations

import dataclasses
import importlib.metadata as importlib_metadata
import logging
import math
import time
from typing import Any

import torch
from mace.cache_utils import BoundedLRUCache
from mace.tools import compile as mace_compile
from mace.tools.force_compile import (
    compile_fx_graph_module,
    edge_gradient_to_atomic_forces,
    rebuild_fx_graph_module,
    trace_force_closure,
)
from mace.tools.scatter import scatter_sum


_COMPILE_FALLBACK_DENYLIST_PATTERNS = (
    "out of memory",
    "cuda error",
    "device-side assert",
    "cublas",
    "cudnn",
    "non-finite",
    "nonfinite",
    "nan",
)


def _is_safe_compile_fallback_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return not any(
        pattern in message for pattern in _COMPILE_FALLBACK_DENYLIST_PATTERNS
    )


@dataclasses.dataclass(frozen=True)
class EdgeForceCompileConfig:
    enabled: bool = False
    tracing_mode: str = "real"
    strip_detach: bool = False
    compile_graph: bool = True
    compile_mode: str = "default"
    compile_dynamic: bool = True
    compile_shape_padding: bool = True
    compile_max_fusion_size: int = 8
    use_e3nn_spherical_harmonics: bool = False
    force_gradient_mode: str = "edge"
    allow_fallback: bool = True
    atol: float = 1.0e-5
    rtol: float = 1.0e-4
    cache_hit_gate: bool = False
    setup_gate: str = "strict"
    cache_policy: str = "repeat_only"
    min_repeats: int = 2
    break_even_expected_remaining_hits: int = 0
    max_cache_entries: int = 32
    disable_negative_speedup: bool = True
    negative_speedup_min_steps: int = 4
    bucket_atoms: tuple[int, ...] = ()
    bucket_edges: tuple[int, ...] = ()
    bucket_margin: float = 1.0
    refresh_executable_each_step: bool | None = None
    parity_check_interval: int = 0
    parity_check_gradients: bool = True
    parity_check_strict: bool = True
    fixed_probe_interval: int = 0
    fixed_probe_gradients: bool = False
    fixed_probe_strict: bool = False

    def __post_init__(self) -> None:
        if self.setup_gate not in {"strict", "none"}:
            raise ValueError("edge-force compile setup_gate must be 'strict' or 'none'")
        if self.force_gradient_mode not in {"edge", "positions"}:
            raise ValueError(
                "edge-force compile force_gradient_mode must be 'edge' or 'positions'"
            )
        if self.cache_policy not in {
            "shape",
            "repeat_only",
            "bucket",
            "dynamic",
            "break_even",
        }:
            raise ValueError(
                "edge-force compile cache_policy must be one of 'shape', "
                "'repeat_only', 'bucket', 'dynamic', or 'break_even'"
            )
        if self.min_repeats < 0:
            raise ValueError(
                "edge-force compile min_repeats must be non-negative"
            )
        if self.break_even_expected_remaining_hits < 0:
            raise ValueError(
                "edge-force compile break_even_expected_remaining_hits must be "
                "non-negative"
            )
        if self.max_cache_entries < 0:
            raise ValueError(
                "edge-force compile max_cache_entries must be non-negative"
            )
        if self.refresh_executable_each_step is None:
            object.__setattr__(self, "refresh_executable_each_step", False)


@dataclasses.dataclass(frozen=True)
class EdgeForceCompileGateResult:
    enabled: bool
    accepted: bool | None
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
    break_even_hits: float | None = None
    break_even_speedup: float | None = None
    expected_remaining_hits: int | None = None


@dataclasses.dataclass
class EdgeForceCacheEntryStats:
    seen_count: int = 0
    compile_count: int = 0
    cache_hit_count: int = 0
    disabled_reason: str | None = None
    compile_setup_seconds: float = 0.0
    compiled_step_seconds_ema: float | None = None
    eager_step_seconds_ema: float | None = None


def _break_even_policy_metrics(
    policy_decision: EdgeForceCachePolicyDecision,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if policy_decision.break_even_speedup is not None:
        metrics["edge_force_break_even_speedup_seconds"] = float(
            policy_decision.break_even_speedup
        )
    if policy_decision.break_even_hits is not None:
        metrics["edge_force_break_even_hits"] = float(policy_decision.break_even_hits)
    if policy_decision.expected_remaining_hits is not None:
        metrics["edge_force_break_even_expected_remaining_hits"] = int(
            policy_decision.expected_remaining_hits
        )
    return metrics


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
        break_even_expected_remaining_hits: int = 0,
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
        break_even_hits = None
        break_even_speedup = None
        expected_remaining_hits = None
        if policy == "shape":
            allowed = True
            reason = None
        elif policy == "repeat_only":
            allowed = stats.seen_count >= max(1, int(min_repeats))
            reason = None if allowed else "min_repeats"
        elif policy in ("bucket", "dynamic"):
            allowed = True
            reason = None
        elif policy == "break_even":
            allowed = stats.seen_count >= max(1, int(min_repeats))
            reason = None if allowed else "min_repeats"
            if allowed and (
                stats.compile_setup_seconds > 0.0
                and stats.eager_step_seconds_ema is not None
                and stats.compiled_step_seconds_ema is not None
            ):
                speedup = stats.eager_step_seconds_ema - stats.compiled_step_seconds_ema
                break_even_speedup = float(speedup)
                expected_remaining_hits = int(break_even_expected_remaining_hits)
                if speedup <= 0.0:
                    allowed = False
                    reason = "break_even_no_speedup"
                else:
                    break_even_hits = stats.compile_setup_seconds / speedup
                    allowed = expected_remaining_hits >= math.ceil(break_even_hits)
                    reason = None if allowed else "break_even"
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
            break_even_hits=break_even_hits,
            break_even_speedup=break_even_speedup,
            expected_remaining_hits=expected_remaining_hits,
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


class _EdgeForceFrozenBatch:
    def __init__(self, data_dict: dict[str, Any]) -> None:
        self.data_dict = data_dict

    def to(self, device, **kwargs):
        return _EdgeForceFrozenBatch(
            {
                key: value.to(device, **kwargs) if hasattr(value, "to") else value
                for key, value in self.data_dict.items()
            }
        )

    def to_dict(self):
        return dict(self.data_dict)

    def __getattr__(self, name):
        try:
            return self.data_dict[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name):
        return self.data_dict[name]


def _freeze_edge_force_batch(batch) -> _EdgeForceFrozenBatch:
    frozen: dict[str, Any] = {}
    for key, value in batch.to_dict().items():
        if torch.is_tensor(value):
            frozen[key] = value.detach().clone().cpu()
        else:
            frozen[key] = value
    return _EdgeForceFrozenBatch(frozen)


_EDGE_FORCE_INPUT_KEYS = (
    "positions",
    "edge_index",
    "node_attrs",
    "batch",
    "ptr",
    "head",
    "_node_mask",
    "_edge_mask",
)
_POSITION_FORCE_INPUT_KEYS = (
    "positions",
    "edge_index",
    "shifts",
    "cell",
    "node_attrs",
    "batch",
    "ptr",
    "head",
    "_node_mask",
    "_edge_mask",
)


def edge_force_compile_input_names(
    data_keys, *, force_gradient_mode: str = "edge"
) -> tuple[str, ...]:
    keys = set(data_keys)
    if force_gradient_mode == "positions":
        input_keys = _POSITION_FORCE_INPUT_KEYS
    elif force_gradient_mode == "edge":
        input_keys = _EDGE_FORCE_INPUT_KEYS
    else:
        raise ValueError("force_gradient_mode must be 'edge' or 'positions'")
    return tuple(name for name in input_keys if name in keys)


_EDGE_FORCE_LOSS_INPUT_KEYS = (
    "energy",
    "forces",
    "weight",
    "energy_weight",
    "forces_weight",
)


def _compiled_tensor_loss_input_names_for_loss(
    loss_fn: torch.nn.Module | None,
) -> tuple[str, ...]:
    if loss_fn is None or _is_weighted_energy_forces_loss(loss_fn):
        return _EDGE_FORCE_LOSS_INPUT_KEYS
    if _is_weighted_forces_loss(loss_fn):
        return ("forces", "weight", "forces_weight")
    if _is_weighted_energy_forces_l1l2_loss(loss_fn):
        return ("energy", "forces", "weight", "energy_weight")
    if _is_weighted_energy_forces_stress_loss(loss_fn) or (
        _is_weighted_huber_energy_forces_stress_loss(loss_fn)
    ):
        return (
            "energy",
            "forces",
            "stress",
            "weight",
            "energy_weight",
            "forces_weight",
            "stress_weight",
        )
    if _is_universal_loss(loss_fn):
        return (
            "energy",
            "forces",
            "stress",
            "energy_weight",
            "forces_weight",
            "stress_weight",
        )
    if _is_weighted_energy_forces_virials_loss(loss_fn):
        return (
            "energy",
            "forces",
            "virials",
            "weight",
            "energy_weight",
            "forces_weight",
            "virials_weight",
        )
    return ()


def edge_force_compile_loss_input_names(
    data_keys, *, loss_fn: torch.nn.Module | None = None
) -> tuple[str, ...]:
    keys = set(data_keys)
    input_keys = _compiled_tensor_loss_input_names_for_loss(loss_fn)
    if not input_keys or not all(name in keys for name in input_keys):
        return ()
    return input_keys


@dataclasses.dataclass(frozen=True)
class LossOutputCapability:
    required_outputs: tuple[str, ...]
    edge_force_supported: bool
    compiled_tensor_loss_supported: bool = False
    unsupported_reason: str | None = None


_LOSS_OUTPUT_CAPABILITY_BY_TYPE_NAME = {
    "WeightedEnergyForcesLoss": LossOutputCapability(
        required_outputs=("energy", "forces"),
        edge_force_supported=True,
        compiled_tensor_loss_supported=True,
    ),
    "WeightedForcesLoss": LossOutputCapability(
        required_outputs=("forces",),
        edge_force_supported=True,
        compiled_tensor_loss_supported=True,
    ),
    "WeightedEnergyForcesL1L2Loss": LossOutputCapability(
        required_outputs=("energy", "forces"),
        edge_force_supported=True,
        compiled_tensor_loss_supported=True,
    ),
    "WeightedEnergyForcesStressLoss": LossOutputCapability(
        required_outputs=("energy", "forces", "stress"),
        edge_force_supported=False,
        compiled_tensor_loss_supported=True,
        unsupported_reason="unsupported_loss",
    ),
    "WeightedHuberEnergyForcesStressLoss": LossOutputCapability(
        required_outputs=("energy", "forces", "stress"),
        edge_force_supported=False,
        compiled_tensor_loss_supported=True,
        unsupported_reason="unsupported_loss",
    ),
    "UniversalLoss": LossOutputCapability(
        required_outputs=("energy", "forces", "stress"),
        edge_force_supported=False,
        compiled_tensor_loss_supported=True,
        unsupported_reason="unsupported_loss",
    ),
    "WeightedEnergyForcesVirialsLoss": LossOutputCapability(
        required_outputs=("energy", "forces", "virials"),
        edge_force_supported=False,
        compiled_tensor_loss_supported=True,
        unsupported_reason="unsupported_loss",
    ),
    "DipoleSingleLoss": LossOutputCapability(
        required_outputs=("dipole",),
        edge_force_supported=False,
        unsupported_reason="unsupported_loss",
    ),
    "DipolePolarLoss": LossOutputCapability(
        required_outputs=("dipole", "polarizability"),
        edge_force_supported=False,
        unsupported_reason="unsupported_loss",
    ),
    "WeightedEnergyForcesDipoleLoss": LossOutputCapability(
        required_outputs=("energy", "forces", "dipole"),
        edge_force_supported=False,
        unsupported_reason="unsupported_loss",
    ),
}


def _is_instance_of_named_type(obj: object, type_names: tuple[str, ...]) -> bool:
    return any(cls.__name__ in type_names for cls in type(obj).__mro__)


def _loss_type_names(loss_fn: torch.nn.Module) -> tuple[str, ...]:
    return tuple(cls.__name__ for cls in type(loss_fn).__mro__)


def edge_force_loss_output_capability(
    loss_fn: torch.nn.Module,
) -> LossOutputCapability:
    for type_name in _loss_type_names(loss_fn):
        capability = _LOSS_OUTPUT_CAPABILITY_BY_TYPE_NAME.get(type_name)
        if capability is not None:
            return capability
    return LossOutputCapability(
        required_outputs=(),
        edge_force_supported=False,
        unsupported_reason="unknown_loss",
    )


def _is_weighted_energy_forces_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("WeightedEnergyForcesLoss",))


def _is_weighted_forces_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("WeightedForcesLoss",))


def _is_weighted_energy_forces_l1l2_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("WeightedEnergyForcesL1L2Loss",))


def _is_weighted_energy_forces_stress_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("WeightedEnergyForcesStressLoss",))


def _is_weighted_huber_energy_forces_stress_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(
        loss_fn, ("WeightedHuberEnergyForcesStressLoss",)
    )


def _is_weighted_energy_forces_virials_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("WeightedEnergyForcesVirialsLoss",))


def _is_universal_loss(loss_fn: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(loss_fn, ("UniversalLoss",))


def _compiled_tensor_loss_weights(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    if _is_weighted_energy_forces_loss(loss_fn):
        return (loss_fn.energy_weight, loss_fn.forces_weight)
    if _is_weighted_forces_loss(loss_fn):
        return (loss_fn.forces_weight,)
    if _is_weighted_energy_forces_l1l2_loss(loss_fn):
        return (loss_fn.energy_weight, loss_fn.forces_weight)
    if _is_weighted_energy_forces_stress_loss(loss_fn):
        return (loss_fn.energy_weight, loss_fn.forces_weight, loss_fn.stress_weight)
    if _is_weighted_huber_energy_forces_stress_loss(loss_fn):
        huber_delta = torch.as_tensor(
            loss_fn.huber_delta,
            dtype=loss_fn.energy_weight.dtype,
            device=loss_fn.energy_weight.device,
        )
        return (
            loss_fn.energy_weight,
            loss_fn.forces_weight,
            loss_fn.stress_weight,
            huber_delta,
        )
    if _is_weighted_energy_forces_virials_loss(loss_fn):
        return (loss_fn.energy_weight, loss_fn.forces_weight, loss_fn.virials_weight)
    if _is_universal_loss(loss_fn):
        huber_delta = torch.as_tensor(
            loss_fn.huber_delta,
            dtype=loss_fn.energy_weight.dtype,
            device=loss_fn.energy_weight.device,
        )
        return (
            loss_fn.energy_weight,
            loss_fn.forces_weight,
            loss_fn.stress_weight,
            huber_delta,
        )
    raise TypeError(
        f"compiled tensor loss does not support {type(loss_fn).__name__}"
    )


def _compiled_tensor_loss_kind(loss_fn: torch.nn.Module) -> str:
    if _is_weighted_energy_forces_loss(loss_fn):
        return "weighted_energy_forces"
    if _is_weighted_forces_loss(loss_fn):
        return "weighted_forces"
    if _is_weighted_energy_forces_l1l2_loss(loss_fn):
        return "weighted_energy_forces_l1l2"
    if _is_weighted_energy_forces_stress_loss(loss_fn):
        return "weighted_energy_forces_stress"
    if _is_weighted_huber_energy_forces_stress_loss(loss_fn):
        return "weighted_huber_energy_forces_stress"
    if _is_weighted_energy_forces_virials_loss(loss_fn):
        return "weighted_energy_forces_virials"
    if _is_universal_loss(loss_fn):
        return "universal"
    raise TypeError(
        f"compiled tensor loss does not support {type(loss_fn).__name__}"
    )


def _is_scale_shift_mace(model: torch.nn.Module) -> bool:
    return _is_instance_of_named_type(model, ("ScaleShiftMACE",))


def _edge_force_can_use_energy_force_outputs(loss_fn: torch.nn.Module) -> bool:
    capability = edge_force_loss_output_capability(loss_fn)
    return capability.edge_force_supported and all(
        output in {"energy", "forces"} for output in capability.required_outputs
    )


def _edge_force_requires_non_energy_force_outputs(loss_fn: torch.nn.Module) -> bool:
    capability = edge_force_loss_output_capability(loss_fn)
    return (
        capability.unsupported_reason == "unsupported_loss"
        and not capability.edge_force_supported
    )


def _edge_force_can_compile_loss(loss_fn: torch.nn.Module, data_keys) -> bool:
    capability = edge_force_loss_output_capability(loss_fn)
    return capability.compiled_tensor_loss_supported and bool(
        edge_force_compile_loss_input_names(data_keys, loss_fn=loss_fn)
    )


def parse_edge_force_bucket_sizes(value: str | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(size <= 0 for size in sizes):
        raise ValueError("edge-force bucket sizes must be positive integers")
    return tuple(sorted(set(sizes)))


_EDGE_FORCE_COMPILE_DEPENDENCY_PACKAGES = (
    "triton",
    "e3nn",
    "cuequivariance",
    "cuequivariance-torch",
)


def _edge_force_dependency_abi_signature() -> tuple:
    versions: list[tuple[str, str | None]] = []
    for package_name in _EDGE_FORCE_COMPILE_DEPENDENCY_PACKAGES:
        try:
            package_version = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            package_version = None
        versions.append((package_name, package_version))
    return ("deps", *versions)


def _edge_force_distributed_abi_signature() -> tuple:
    dist = getattr(torch, "distributed", None)
    if dist is None:
        return (
            "distributed",
            ("available", False),
            ("initialized", False),
            ("backend", None),
            ("world_size", None),
            ("rank", None),
        )
    try:
        available = bool(dist.is_available())
    except Exception:  # pragma: no cover - defensive for unusual torch builds
        available = False
    try:
        initialized = bool(dist.is_initialized()) if available else False
    except Exception:  # pragma: no cover - defensive for unusual torch builds
        initialized = False
    backend = None
    world_size = None
    rank = None
    if initialized:
        try:
            backend = str(dist.get_backend())
        except Exception:  # pragma: no cover - defensive for distributed teardown
            backend = None
        try:
            world_size = int(dist.get_world_size())
        except Exception:  # pragma: no cover - defensive for distributed teardown
            world_size = None
        try:
            rank = int(dist.get_rank())
        except Exception:  # pragma: no cover - defensive for distributed teardown
            rank = None
    return (
        "distributed",
        ("available", available),
        ("initialized", initialized),
        ("backend", backend),
        ("world_size", world_size),
        ("rank", rank),
    )


def _edge_force_autocast_abi_signature() -> tuple:
    entries: list[tuple[str, bool, str | None]] = []
    for device_type in ("cpu", "cuda"):
        try:
            enabled = bool(torch.is_autocast_enabled(device_type))
        except TypeError:  # pragma: no cover - compatibility with older torch
            enabled = bool(torch.is_autocast_enabled())
        except Exception:  # pragma: no cover - defensive for unusual runtimes
            enabled = False
        try:
            dtype = str(torch.get_autocast_dtype(device_type))
        except Exception:  # pragma: no cover - defensive for unsupported devices
            dtype = None
        entries.append((device_type, enabled, dtype))
    return ("autocast", *entries)


def _edge_force_device_abi_signature(device: torch.device | str) -> tuple:
    resolved_device = torch.device(device)
    device_index = resolved_device.index
    if resolved_device.type == "cuda" and device_index is None:
        try:
            device_index = int(torch.cuda.current_device())
        except Exception:  # pragma: no cover - defensive for broken CUDA runtimes
            device_index = None

    signature: list[Any] = ["device", resolved_device.type, device_index]
    if resolved_device.type == "cuda" and device_index is not None:
        try:
            capability = torch.cuda.get_device_capability(device_index)
            signature.append(("capability", int(capability[0]), int(capability[1])))
        except Exception:  # pragma: no cover - depends on CUDA runtime availability
            signature.append(("capability", None))
        try:
            signature.append(("name", str(torch.cuda.get_device_name(device_index))))
        except Exception:  # pragma: no cover - depends on CUDA runtime availability
            signature.append(("name", None))
    return tuple(signature)


def _with_cache_abi_signature(
    cache_key: tuple, abi_signature: tuple | None
) -> tuple:
    if abi_signature is None:
        return cache_key
    return (*cache_key[:-1], abi_signature, cache_key[-1])


def edge_force_compile_shape_cache_key(
    *,
    num_atoms: int,
    num_edges: int,
    input_shapes: dict[str, tuple[int, ...]],
    abi_signature: tuple | None = None,
) -> tuple:
    cache_key = (
        "shape",
        int(num_atoms),
        int(num_edges),
        tuple((name, tuple(shape)) for name, shape in sorted(input_shapes.items())),
    )
    return _with_cache_abi_signature(cache_key, abi_signature)


def _select_edge_force_bucket(
    size: int, buckets: tuple[int, ...], *, bucket_margin: float
) -> int | None:
    for bucket in buckets:
        if bucket >= size:
            if bucket_margin > 0.0 and bucket > size * bucket_margin:
                return None
            return bucket
    return None


def _pad_first_dim(tensor: torch.Tensor, target_size: int, *, fill: float = 0.0) -> torch.Tensor:
    current_size = int(tensor.shape[0])
    if current_size == target_size:
        return tensor
    if current_size > target_size:
        raise ValueError(f"cannot pad tensor with first dim {current_size} to {target_size}")
    shape = list(tensor.shape)
    shape[0] = target_size - current_size
    padding = torch.full(shape, fill, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, padding), dim=0)


def _pad_edge_force_data_to_bucket(
    data: dict[str, torch.Tensor],
    *,
    atom_bucket: int,
    edge_bucket: int,
    r_max: float,
) -> dict[str, torch.Tensor]:
    real_num_atoms = int(data["positions"].shape[0])
    real_num_edges = int(data["edge_index"].shape[1])
    if atom_bucket < real_num_atoms or edge_bucket < real_num_edges:
        raise ValueError(
            "edge-force bucket must be at least as large as current atom/edge counts"
        )

    padded = dict(data)
    device = data["positions"].device
    dtype = data["positions"].dtype

    padded["positions"] = _pad_first_dim(data["positions"], atom_bucket)
    if "node_attrs" in data:
        node_attrs = _pad_first_dim(data["node_attrs"], atom_bucket)
        if atom_bucket > real_num_atoms and node_attrs.shape[1] > 0:
            node_attrs[real_num_atoms:, :] = 0
            node_attrs[real_num_atoms:, 0] = 1
        padded["node_attrs"] = node_attrs
    if "batch" in data:
        padded["batch"] = _pad_first_dim(data["batch"], atom_bucket)
    if "forces" in data:
        padded["forces"] = _pad_first_dim(data["forces"], atom_bucket)

    edge_index = torch.zeros(
        (2, edge_bucket), dtype=data["edge_index"].dtype, device=data["edge_index"].device
    )
    edge_index[:, :real_num_edges] = data["edge_index"]
    if edge_bucket > real_num_edges:
        last_atom = max(atom_bucket - 1, 0)
        edge_index[:, real_num_edges:] = last_atom
    padded["edge_index"] = edge_index

    if "shifts" in data:
        shifts = _pad_first_dim(data["shifts"], edge_bucket)
        if edge_bucket > real_num_edges:
            shifts[real_num_edges:, :] = 0
            shifts[real_num_edges:, 0] = float(r_max) * 2.0
        padded["shifts"] = shifts
    if "unit_shifts" in data:
        unit_shifts = _pad_first_dim(data["unit_shifts"], edge_bucket)
        if edge_bucket > real_num_edges:
            unit_shifts[real_num_edges:, :] = 0
            unit_shifts[real_num_edges:, 0] = 1
        padded["unit_shifts"] = unit_shifts

    node_mask = torch.zeros(atom_bucket, dtype=dtype, device=device)
    node_mask[:real_num_atoms] = 1
    edge_mask = torch.zeros(edge_bucket, dtype=dtype, device=device)
    edge_mask[:real_num_edges] = 1
    padded["_node_mask"] = node_mask
    padded["_edge_mask"] = edge_mask
    padded["_real_num_atoms"] = torch.tensor(real_num_atoms, dtype=torch.int64, device=device)
    return padded


def _dynamic_edge_force_shape(name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    dynamic_dims = {
        "positions": {0},
        "edge_index": {1},
        "shifts": {0},
        "unit_shifts": {0},
        "node_attrs": {0},
        "batch": {0},
        "_node_mask": {0},
        "_edge_mask": {0},
    }.get(name, set())
    return tuple(-1 if dim_index in dynamic_dims else int(dim) for dim_index, dim in enumerate(shape))


def edge_force_compile_dynamic_cache_key(
    *,
    input_shapes: dict[str, tuple[int, ...]],
    abi_signature: tuple | None = None,
) -> tuple:
    cache_key = (
        "dynamic",
        tuple(
            (name, _dynamic_edge_force_shape(name, tuple(shape)))
            for name, shape in sorted(input_shapes.items())
        ),
    )
    return _with_cache_abi_signature(cache_key, abi_signature)


def edge_force_compile_bucket_cache_key(
    *,
    num_atoms: int,
    num_edges: int,
    input_shapes: dict[str, tuple[int, ...]],
    bucket_atoms: tuple[int, ...],
    bucket_edges: tuple[int, ...],
    bucket_margin: float,
    abi_signature: tuple | None = None,
) -> tuple | None:
    atom_bucket = _select_edge_force_bucket(
        int(num_atoms), bucket_atoms, bucket_margin=float(bucket_margin)
    )
    edge_bucket = _select_edge_force_bucket(
        int(num_edges), bucket_edges, bucket_margin=float(bucket_margin)
    )
    if atom_bucket is None or edge_bucket is None:
        return None

    bucketed_shapes: dict[str, tuple[int, ...]] = {}
    for name, shape in input_shapes.items():
        bucketed_shape = tuple(
            atom_bucket
            if int(dim) == int(num_atoms)
            else edge_bucket
            if int(dim) == int(num_edges)
            else int(dim)
            for dim in shape
        )
        bucketed_shapes[name] = bucketed_shape

    cache_key = (
        "bucket",
        atom_bucket,
        edge_bucket,
        tuple((name, tuple(shape)) for name, shape in sorted(bucketed_shapes.items())),
    )
    return _with_cache_abi_signature(cache_key, abi_signature)


def _edge_force_bucket_sizes_from_cache_key(cache_key: tuple) -> tuple[int, int] | None:
    if len(cache_key) >= 3 and cache_key[0] == "bucket":
        return int(cache_key[1]), int(cache_key[2])
    return None


def _edge_force_model_r_max(model: torch.nn.Module) -> float:
    value = getattr(model, "r_max", None)
    if value is None:
        raise AttributeError("edge-force bucket padding requires model.r_max")
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


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


def _compiled_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(name for name, param in model.named_parameters() if param.requires_grad)


def _parameter_tensors_by_name(
    model: torch.nn.Module, param_names: tuple[str, ...]
) -> tuple[torch.Tensor, ...]:
    named_parameters = dict(model.named_parameters())
    return tuple(named_parameters[name] for name in param_names)



def _edge_force_executable_inputs(
    *,
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    param_names: tuple[str, ...],
    input_names: tuple[str, ...],
    loss_input_names: tuple[str, ...] = (),
    loss_fn: torch.nn.Module | None = None,
) -> list[torch.Tensor]:
    loss_weight_tensors: tuple[torch.Tensor, ...] = ()
    if loss_input_names:
        if loss_fn is None:
            raise TypeError("compiled tensor loss requires a loss_fn")
        loss_weight_tensors = _compiled_tensor_loss_weights(loss_fn)
    return [
        *_parameter_tensors_by_name(model, param_names),
        *(data_dict[name] for name in input_names),
        *(data_dict[name] for name in loss_input_names),
        *loss_weight_tensors,
    ]


def _parameter_dict_from_tensors(
    param_names: tuple[str, ...], param_tensors: tuple[torch.Tensor, ...]
) -> dict[str, torch.Tensor]:
    return dict(zip(param_names, param_tensors, strict=True))


def _parameter_module_and_local_name(
    model: torch.nn.Module, parameter_name: str
) -> tuple[torch.nn.Module, str]:
    if "." not in parameter_name:
        return model, parameter_name
    module_name, local_name = parameter_name.rsplit(".", 1)
    return model.get_submodule(module_name), local_name


def _replace_module_parameters_with_tensors(
    model: torch.nn.Module,
    param_names: tuple[str, ...],
    param_tensors: tuple[torch.Tensor, ...],
) -> list[tuple[torch.nn.Module, str, torch.Tensor | None]]:
    if len(param_names) != len(param_tensors):
        raise ValueError(
            "parameter placeholder count mismatch: "
            f"expected {len(param_names)}, got {len(param_tensors)}"
        )
    saved: list[tuple[torch.nn.Module, str, torch.Tensor | None]] = []
    try:
        for name, tensor in zip(param_names, param_tensors, strict=True):
            module, local_name = _parameter_module_and_local_name(model, name)
            saved.append((module, local_name, module._parameters[local_name]))
            module._parameters[local_name] = tensor
    except Exception:
        _restore_module_parameters(saved)
        raise
    return saved


def _restore_module_parameters(
    saved: list[tuple[torch.nn.Module, str, torch.Tensor | None]]
) -> None:
    for module, local_name, original in reversed(saved):
        module._parameters[local_name] = original


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


def _select_by_node_heads(values: torch.Tensor, node_heads: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, node_heads.to(torch.int64).reshape(-1, 1)).squeeze(1)


def _edge_force_spherical_harmonics_polynomial(
    lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    if lmax == 0:
        return torch.stack([sh_0_0], dim=-1)

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
    if lmax == 2:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
            ],
            dim=-1,
        )

    sh_3_0 = math.sqrt(5.0 / 6.0) * (sh_2_0 * z + sh_2_4 * x)
    sh_3_1 = math.sqrt(5.0) * sh_2_0 * y
    sh_3_2 = math.sqrt(3.0 / 8.0) * (4.0 * y2 - x2z2) * x
    sh_3_3 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh_3_4 = math.sqrt(3.0 / 8.0) * z * (4.0 * y2 - x2z2)
    sh_3_5 = math.sqrt(5.0) * sh_2_4 * y
    sh_3_6 = math.sqrt(5.0 / 6.0) * (sh_2_4 * z - sh_2_0 * x)
    if lmax == 3:
        return torch.stack(
            [
                sh_0_0,
                sh_1_0,
                sh_1_1,
                sh_1_2,
                sh_2_0,
                sh_2_1,
                sh_2_2,
                sh_2_3,
                sh_2_4,
                sh_3_0,
                sh_3_1,
                sh_3_2,
                sh_3_3,
                sh_3_4,
                sh_3_5,
                sh_3_6,
            ],
            dim=-1,
        )

    raise NotImplementedError(
        "edge-force symbolic compile currently supports spherical harmonics up to lmax=3"
    )


def _edge_force_spherical_harmonics(
    spherical_harmonics: torch.nn.Module,
    vectors: torch.Tensor,
    *,
    use_e3nn: bool = False,
) -> torch.Tensor:
    lmax = int(getattr(spherical_harmonics, "_lmax"))
    if use_e3nn or lmax > 3:
        return spherical_harmonics(vectors)

    x = vectors
    if bool(getattr(spherical_harmonics, "normalize")):
        x = torch.nn.functional.normalize(x, dim=-1)

    sh = _edge_force_spherical_harmonics_polynomial(
        lmax, x[..., 0], x[..., 1], x[..., 2]
    )
    ls_list = list(getattr(spherical_harmonics, "_ls_list"))
    if not bool(getattr(spherical_harmonics, "_is_range_lmax")):
        sh = torch.cat([sh[..., l * l : (l + 1) * (l + 1)] for l in ls_list], dim=-1)

    normalization = str(getattr(spherical_harmonics, "normalization"))
    if normalization == "integral":
        scale_values = [
            math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)
            for l in ls_list
            for _ in range(2 * l + 1)
        ]
    elif normalization == "component":
        scale_values = [
            math.sqrt(2 * l + 1) for l in ls_list for _ in range(2 * l + 1)
        ]
    elif normalization == "norm":
        scale_values = []
    else:
        raise ValueError(f"unsupported spherical harmonics normalization: {normalization}")
    if scale_values:
        scale = sh.new_tensor(scale_values)
        sh = sh * scale
    return sh


def _apply_first_dim_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    view_shape = (mask.shape[0],) + (1,) * (values.dim() - 1)
    return values * mask.to(dtype=values.dtype, device=values.device).reshape(view_shape)


def _mace_energy_from_edge_vectors(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    *,
    vectors: torch.Tensor,
    lengths: torch.Tensor,
    use_e3nn_spherical_harmonics: bool = False,
) -> dict[str, torch.Tensor]:
    if not _is_scale_shift_mace(model):
        raise TypeError("edge-force compiled loss currently supports ScaleShiftMACE only")

    num_graphs = int(data["ptr"].numel() - 1)
    if "head" in data:
        node_heads = torch.gather(
            data["head"].to(torch.int64), 0, data["batch"].to(torch.int64)
        )
    else:
        node_heads = torch.zeros_like(data["batch"], dtype=torch.int64)
    node_mask = data.get(
        "_node_mask",
        torch.ones(data["positions"].shape[0], dtype=vectors.dtype, device=vectors.device),
    ).to(device=vectors.device, dtype=vectors.dtype)
    edge_mask = data.get(
        "_edge_mask",
        torch.ones(data["edge_index"].shape[1], dtype=vectors.dtype, device=vectors.device),
    ).to(device=vectors.device, dtype=vectors.dtype)

    node_e0 = _select_by_node_heads(
        model.atomic_energies_fn(data["node_attrs"]), node_heads
    )
    node_e0 = _apply_first_dim_mask(node_e0, node_mask)
    e0 = scatter_sum(src=node_e0, index=data["batch"], dim=0, dim_size=num_graphs).to(
        vectors.dtype
    )

    node_feats = model.node_embedding(data["node_attrs"])
    edge_attrs = _edge_force_spherical_harmonics(
        model.spherical_harmonics,
        vectors,
        use_e3nn=use_e3nn_spherical_harmonics,
    )
    edge_feats, cutoff = model.radial_embedding(
        lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
    )
    edge_attrs = _apply_first_dim_mask(edge_attrs, edge_mask)
    edge_feats = _apply_first_dim_mask(edge_feats, edge_mask)
    if cutoff is not None:
        cutoff = _apply_first_dim_mask(cutoff, edge_mask)

    if hasattr(model, "pair_repulsion"):
        pair_node_energy = model.pair_repulsion_fn(
            lengths, data["node_attrs"], data["edge_index"], model.atomic_numbers
        )
        pair_node_energy = _apply_first_dim_mask(pair_node_energy, node_mask)
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
            embedding_node_energy = _apply_first_dim_mask(
                embedding_node_energy, node_mask
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
            _apply_first_dim_mask(
                _select_by_node_heads(readout(node_feats_list[feat_idx], node_heads), node_heads),
                node_mask,
            )
        )

    node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
    node_inter_es = model.scale_shift(node_inter_es, node_heads)
    node_inter_es = _apply_first_dim_mask(node_inter_es, node_mask)
    inter_e = scatter_sum(node_inter_es, data["batch"], dim=-1, dim_size=num_graphs)
    total_energy = e0 + inter_e
    node_energy = node_e0.clone().double() + node_inter_es.clone().double()
    return {
        "energy": total_energy,
        "node_energy": node_energy,
        "interaction_energy": inter_e,
    }


def _edge_force_energy_and_edge_grad(
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    vectors: torch.Tensor,
    *,
    use_e3nn_spherical_harmonics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = vectors.detach().requires_grad_(True)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    output = _mace_energy_from_edge_vectors(
        model,
        data_dict,
        vectors=vectors,
        lengths=lengths,
        use_e3nn_spherical_harmonics=use_e3nn_spherical_harmonics,
    )
    edge_grad = torch.autograd.grad(
        outputs=[output["energy"]],
        inputs=[vectors],
        grad_outputs=[torch.ones_like(output["energy"])],
        retain_graph=True,
        create_graph=True,
        allow_unused=False,
    )[0]
    return output["energy"], edge_grad


def _atomic_forces_from_edge_grad(
    data_dict: dict[str, torch.Tensor], edge_grad: torch.Tensor
) -> torch.Tensor:
    return edge_gradient_to_atomic_forces(
        edge_grad,
        edge_index=data_dict["edge_index"],
        num_atoms=data_dict["positions"].shape[0],
    )


def _position_force_energy_and_forces(
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    positions: torch.Tensor,
    *,
    use_e3nn_spherical_harmonics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.detach().requires_grad_(True)
    current_data = dict(data_dict)
    current_data["positions"] = positions
    from mace.modules.utils import get_edge_vectors_and_lengths

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=current_data["edge_index"],
        shifts=current_data["shifts"].detach(),
    )
    output = _mace_energy_from_edge_vectors(
        model,
        current_data,
        vectors=vectors,
        lengths=lengths,
        use_e3nn_spherical_harmonics=use_e3nn_spherical_harmonics,
    )
    gradient = torch.autograd.grad(
        outputs=[output["energy"]],
        inputs=[positions],
        grad_outputs=[torch.ones_like(output["energy"])],
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]
    forces = torch.zeros_like(positions) if gradient is None else -1.0 * gradient
    return output["energy"], forces


def _position_model_outputs(
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    positions: torch.Tensor,
    *,
    compute_virials: bool = False,
    compute_stress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    positions = positions.detach().requires_grad_(True)
    current_data = dict(data_dict)
    current_data["positions"] = positions
    output = model(
        current_data,
        training=True,
        compute_force=True,
        compute_virials=compute_virials,
        compute_stress=compute_stress,
    )
    return output["energy"], output["forces"], output.get("stress"), output.get("virials")


def _forces_matching_reference(
    data_dict: dict[str, torch.Tensor], forces: torch.Tensor
) -> torch.Tensor:
    ref_forces = data_dict["forces"]
    if forces.shape[0] != ref_forces.shape[0]:
        return forces[: ref_forces.shape[0]]
    return forces


def _weighted_energy_mse_raw(
    data_dict: dict[str, torch.Tensor], energy: torch.Tensor
) -> torch.Tensor:
    num_atoms = data_dict["ptr"][1:] - data_dict["ptr"][:-1]
    return (
        data_dict["weight"]
        * data_dict["energy_weight"]
        * torch.square((data_dict["energy"] - energy) / num_atoms)
    )


def _weighted_energy_mae_raw(
    data_dict: dict[str, torch.Tensor], energy: torch.Tensor
) -> torch.Tensor:
    num_atoms = data_dict["ptr"][1:] - data_dict["ptr"][:-1]
    return (
        data_dict["weight"]
        * data_dict["energy_weight"]
        * torch.abs((data_dict["energy"] - energy) / num_atoms)
    )


def _per_atom_config_values(
    data_dict: dict[str, torch.Tensor], key: str, *, like: torch.Tensor
) -> torch.Tensor:
    if "batch" in data_dict:
        per_atom = data_dict[key][data_dict["batch"]].unsqueeze(-1)
        node_mask = data_dict.get("_node_mask")
        if node_mask is not None and node_mask.shape[0] == per_atom.shape[0]:
            per_atom = per_atom * node_mask.to(
                dtype=per_atom.dtype, device=per_atom.device
            ).unsqueeze(-1)
        return per_atom.to(dtype=like.dtype, device=like.device)
    num_atoms = data_dict["ptr"][1:] - data_dict["ptr"][:-1]
    return torch.repeat_interleave(data_dict[key], num_atoms).unsqueeze(-1).to(
        dtype=like.dtype, device=like.device
    )


def _weighted_forces_mse_raw(
    data_dict: dict[str, torch.Tensor], forces: torch.Tensor
) -> torch.Tensor:
    ref_forces = data_dict["forces"]
    forces = _forces_matching_reference(data_dict, forces)
    configs_weight = _per_atom_config_values(data_dict, "weight", like=forces)
    configs_forces_weight = _per_atom_config_values(
        data_dict, "forces_weight", like=forces
    )
    return configs_weight * configs_forces_weight * torch.square(ref_forces - forces)


def _forces_norm_raw(
    data_dict: dict[str, torch.Tensor], forces: torch.Tensor
) -> torch.Tensor:
    ref_forces = data_dict["forces"]
    forces = _forces_matching_reference(data_dict, forces)
    return torch.linalg.vector_norm(ref_forces - forces, ord=2, dim=-1)


def _node_masked_mean(
    data_dict: dict[str, torch.Tensor], values: torch.Tensor
) -> torch.Tensor:
    node_mask = data_dict.get("_node_mask")
    if node_mask is None or values.shape[0] != node_mask.shape[0]:
        return values.mean()
    mask = node_mask.to(dtype=values.dtype, device=values.device)
    view_shape = (mask.shape[0],) + (1,) * (values.dim() - 1)
    masked_values = values * mask.reshape(view_shape)
    values_per_node = max(1, values.numel() // int(values.shape[0]))
    denom = mask.sum() * values_per_node
    return masked_values.sum() / denom.clamp_min(1)


def _edge_force_weighted_energy_forces_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    return (
        energy_scale * _weighted_energy_mse_raw(data_dict, energy).mean()
        + forces_scale
        * _node_masked_mean(data_dict, _weighted_forces_mse_raw(data_dict, forces))
    )


def _edge_force_weighted_forces_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    forces: torch.Tensor,
    forces_loss_weight: torch.Tensor,
) -> torch.Tensor:
    forces_scale = forces_loss_weight.to(device=forces.device)
    return forces_scale * _node_masked_mean(
        data_dict, _weighted_forces_mse_raw(data_dict, forces)
    )


def _edge_force_weighted_energy_forces_l1l2_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    return (
        energy_scale * _weighted_energy_mae_raw(data_dict, energy).mean()
        + forces_scale * _node_masked_mean(data_dict, _forces_norm_raw(data_dict, forces))
    )


def _weighted_stress_mse_raw(
    data_dict: dict[str, torch.Tensor], stress: torch.Tensor
) -> torch.Tensor:
    configs_weight = data_dict["weight"].view(-1, 1, 1)
    configs_stress_weight = data_dict["stress_weight"].view(-1, 1, 1)
    return (
        configs_weight
        * configs_stress_weight
        * torch.square(data_dict["stress"] - stress)
    )


def _weighted_virials_mse_raw(
    data_dict: dict[str, torch.Tensor], virials: torch.Tensor
) -> torch.Tensor:
    configs_weight = data_dict["weight"].view(-1, 1, 1)
    configs_virials_weight = data_dict["virials_weight"].view(-1, 1, 1)
    num_atoms = (data_dict["ptr"][1:] - data_dict["ptr"][:-1]).view(-1, 1, 1)
    return (
        configs_weight
        * configs_virials_weight
        * torch.square((data_dict["virials"] - virials) / num_atoms)
    )


def _edge_force_weighted_energy_forces_stress_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
    stress_loss_weight: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    stress_scale = stress_loss_weight.to(device=energy.device)
    return (
        energy_scale * _weighted_energy_mse_raw(data_dict, energy).mean()
        + forces_scale
        * _node_masked_mean(data_dict, _weighted_forces_mse_raw(data_dict, forces))
        + stress_scale * _weighted_stress_mse_raw(data_dict, stress).mean()
    )


def _edge_force_weighted_energy_forces_virials_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    virials: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
    virials_loss_weight: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    virials_scale = virials_loss_weight.to(device=energy.device)
    return (
        energy_scale * _weighted_energy_mse_raw(data_dict, energy).mean()
        + forces_scale
        * _node_masked_mean(data_dict, _weighted_forces_mse_raw(data_dict, forces))
        + virials_scale * _weighted_virials_mse_raw(data_dict, virials).mean()
    )


def _edge_force_weighted_huber_energy_forces_stress_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
    stress_loss_weight: torch.Tensor,
    huber_delta: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    stress_scale = stress_loss_weight.to(device=energy.device)
    delta = huber_delta.to(device=energy.device, dtype=energy.dtype)
    num_atoms = data_dict["ptr"][1:] - data_dict["ptr"][:-1]

    def huber_mean(reference: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        error = reference - prediction
        abs_error = torch.abs(error)
        quadratic = 0.5 * torch.square(error)
        linear = delta * (abs_error - 0.5 * delta)
        return torch.where(abs_error <= delta, quadratic, linear).mean()

    loss_energy = huber_mean(data_dict["energy"] / num_atoms, energy / num_atoms)
    loss_forces = huber_mean(data_dict["forces"], forces)
    loss_stress = huber_mean(data_dict["stress"], stress)
    return (
        energy_scale * loss_energy
        + forces_scale * loss_forces
        + stress_scale * loss_stress
    )


def _huber_mean(
    reference: torch.Tensor, prediction: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    error = reference - prediction
    abs_error = torch.abs(error)
    quadratic = 0.5 * torch.square(error)
    linear = delta * (abs_error - 0.5 * delta)
    return torch.where(abs_error <= delta, quadratic, linear).mean()


def _conditional_huber_forces_mean(
    reference: torch.Tensor, prediction: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    norm_forces = torch.linalg.vector_norm(reference, ord=2, dim=-1, keepdim=True)
    factor = torch.where(
        norm_forces < 100,
        torch.ones_like(norm_forces),
        torch.where(
            norm_forces < 200,
            torch.full_like(norm_forces, 0.7),
            torch.where(
                norm_forces < 300,
                torch.full_like(norm_forces, 0.4),
                torch.full_like(norm_forces, 0.1),
            ),
        ),
    )
    return _huber_mean(reference, prediction, delta * factor)


def _edge_force_universal_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    stress: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
    stress_loss_weight: torch.Tensor,
    huber_delta: torch.Tensor,
) -> torch.Tensor:
    energy_scale = energy_loss_weight.to(device=energy.device)
    forces_scale = forces_loss_weight.to(device=energy.device)
    stress_scale = stress_loss_weight.to(device=energy.device)
    delta = huber_delta.to(device=energy.device, dtype=energy.dtype)
    num_atoms = data_dict["ptr"][1:] - data_dict["ptr"][:-1]
    configs_energy_weight = data_dict["energy_weight"]
    configs_forces_weight = torch.repeat_interleave(
        data_dict["forces_weight"], num_atoms
    ).unsqueeze(-1)
    configs_stress_weight = data_dict["stress_weight"].view(-1, 1, 1)

    loss_energy = _huber_mean(
        configs_energy_weight * data_dict["energy"] / num_atoms,
        configs_energy_weight * energy / num_atoms,
        delta,
    )
    loss_forces = _conditional_huber_forces_mean(
        configs_forces_weight * data_dict["forces"],
        configs_forces_weight * forces,
        delta,
    )
    loss_stress = _huber_mean(
        configs_stress_weight * data_dict["stress"],
        configs_stress_weight * stress,
        delta,
    )
    return (
        energy_scale * loss_energy
        + forces_scale * loss_forces
        + stress_scale * loss_stress
    )


def _edge_force_compiled_tensor_loss(
    *,
    loss_kind: str,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    loss_weights: tuple[torch.Tensor, ...],
    stress: torch.Tensor | None = None,
    virials: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_kind == "weighted_energy_forces":
        return _edge_force_weighted_energy_forces_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
        )
    if loss_kind == "weighted_forces":
        return _edge_force_weighted_forces_loss(
            data_dict=data_dict,
            forces=forces,
            forces_loss_weight=loss_weights[0],
        )
    if loss_kind == "weighted_energy_forces_l1l2":
        return _edge_force_weighted_energy_forces_l1l2_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
        )
    if loss_kind == "weighted_energy_forces_stress":
        if stress is None:
            raise RuntimeError("compiled stress loss requires stress output")
        return _edge_force_weighted_energy_forces_stress_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            stress=stress,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
            stress_loss_weight=loss_weights[2],
        )
    if loss_kind == "weighted_energy_forces_virials":
        if virials is None:
            raise RuntimeError("compiled virials loss requires virials output")
        return _edge_force_weighted_energy_forces_virials_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            virials=virials,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
            virials_loss_weight=loss_weights[2],
        )
    if loss_kind == "weighted_huber_energy_forces_stress":
        if stress is None:
            raise RuntimeError("compiled Huber stress loss requires stress output")
        return _edge_force_weighted_huber_energy_forces_stress_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            stress=stress,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
            stress_loss_weight=loss_weights[2],
            huber_delta=loss_weights[3],
        )
    if loss_kind == "universal":
        if stress is None:
            raise RuntimeError("compiled UniversalLoss requires stress output")
        return _edge_force_universal_loss(
            data_dict=data_dict,
            energy=energy,
            forces=forces,
            stress=stress,
            energy_loss_weight=loss_weights[0],
            forces_loss_weight=loss_weights[1],
            stress_loss_weight=loss_weights[2],
            huber_delta=loss_weights[3],
        )
    raise RuntimeError(f"unsupported compiled tensor loss kind: {loss_kind}")


def _position_force_weighted_energy_forces_loss(
    *,
    data_dict: dict[str, torch.Tensor],
    energy: torch.Tensor,
    forces: torch.Tensor,
    energy_loss_weight: torch.Tensor,
    forces_loss_weight: torch.Tensor,
) -> torch.Tensor:
    return _edge_force_weighted_energy_forces_loss(
        data_dict=data_dict,
        energy=energy,
        forces=forces,
        energy_loss_weight=energy_loss_weight,
        forces_loss_weight=forces_loss_weight,
    )


def _edge_force_outputs(
    model: torch.nn.Module,
    data_dict: dict[str, torch.Tensor],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    vectors: torch.Tensor,
    *,
    use_e3nn_spherical_harmonics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    energy, edge_grad = _edge_force_energy_and_edge_grad(
        model,
        data_dict,
        vectors,
        use_e3nn_spherical_harmonics=use_e3nn_spherical_harmonics,
    )
    forces = edge_gradient_to_atomic_forces(
        edge_grad,
        edge_index=edge_index,
        num_atoms=positions.shape[0],
    )
    return energy, forces


def _loss_required_output_flags(loss_fn: torch.nn.Module) -> tuple[bool, bool]:
    capability = edge_force_loss_output_capability(loss_fn)
    required = set(capability.required_outputs)
    return "virials" in required, "stress" in required


def _loss_from_energy_forces(
    *, batch, loss_fn, energy: torch.Tensor, forces: torch.Tensor
) -> torch.Tensor:
    if hasattr(batch, "forces") and forces.shape[0] != batch.forces.shape[0]:
        forces = forces[: batch.forces.shape[0]]
    output = {
        "energy": energy,
        "forces": forces,
        "virials": None,
        "stress": None,
    }
    return loss_fn(pred=output, ref=batch)


def _edge_vector_inputs(
    batch,
    *,
    atom_bucket: int | None = None,
    edge_bucket: int | None = None,
    r_max: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    data = batch.to_dict()
    if atom_bucket is not None or edge_bucket is not None:
        if atom_bucket is None or edge_bucket is None or r_max is None:
            raise ValueError("atom_bucket, edge_bucket, and r_max must be set together")
        data = _pad_edge_force_data_to_bucket(
            data, atom_bucket=atom_bucket, edge_bucket=edge_bucket, r_max=float(r_max)
        )
    positions = data["positions"].detach()
    edge_index = data["edge_index"]
    from mace.modules.utils import get_edge_vectors_and_lengths

    vectors, _ = get_edge_vectors_and_lengths(
        positions=positions,
        edge_index=edge_index,
        shifts=data["shifts"].detach(),
    )
    vectors = vectors.detach().clone()
    return data, positions, edge_index, vectors


def _edge_force_snapshot_from_executable(
    *,
    model: torch.nn.Module,
    batch,
    loss_fn,
    executable,
    input_names: tuple[str, ...],
    bucket_sizes: tuple[int, int] | None = None,
    param_names: tuple[str, ...] = (),
    loss_input_names: tuple[str, ...] = (),
    force_gradient_mode: str = "edge",
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    if bucket_sizes is None:
        data_dict, positions, _, vectors = _edge_vector_inputs(batch)
    else:
        data_dict, positions, _, vectors = _edge_vector_inputs(
            batch,
            atom_bucket=bucket_sizes[0],
            edge_bucket=bucket_sizes[1],
            r_max=_edge_force_model_r_max(model),
        )
    gradient_input = positions if force_gradient_mode == "positions" else vectors
    gradient_input = gradient_input.detach().clone()
    inputs = _edge_force_executable_inputs(
        model=model,
        data_dict=data_dict,
        param_names=param_names,
        input_names=input_names,
        loss_input_names=loss_input_names,
        loss_fn=loss_fn,
    )
    executable_outputs = executable(gradient_input, *inputs)
    if len(executable_outputs) == 3:
        energy, forces, loss = executable_outputs
    else:
        energy, force_like = executable_outputs
        forces = (
            force_like
            if force_gradient_mode == "positions"
            else _atomic_forces_from_edge_grad(data_dict, force_like)
        )
        loss = _loss_from_energy_forces(
            batch=batch,
            loss_fn=loss_fn,
            energy=energy,
            forces=forces,
        )
    loss.backward()
    if hasattr(batch, "forces") and forces.shape[0] != batch.forces.shape[0]:
        forces = forces[: batch.forces.shape[0]]
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
    bucket_sizes: tuple[int, int] | None = None,
    param_names: tuple[str, ...] = (),
    loss_input_names: tuple[str, ...] = (),
    force_gradient_mode: str = "edge",
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    if bucket_sizes is None:
        data_dict, positions, _, vectors = _edge_vector_inputs(batch)
    else:
        data_dict, positions, _, vectors = _edge_vector_inputs(
            batch,
            atom_bucket=bucket_sizes[0],
            edge_bucket=bucket_sizes[1],
            r_max=_edge_force_model_r_max(model),
        )
    gradient_input = positions if force_gradient_mode == "positions" else vectors
    gradient_input = gradient_input.detach().clone()
    inputs = _edge_force_executable_inputs(
        model=model,
        data_dict=data_dict,
        param_names=param_names,
        input_names=input_names,
        loss_input_names=loss_input_names,
        loss_fn=loss_fn,
    )
    executable_outputs = executable(gradient_input, *inputs)
    if len(executable_outputs) == 3:
        energy, forces, loss = executable_outputs
    else:
        energy, force_like = executable_outputs
        forces = (
            force_like
            if force_gradient_mode == "positions"
            else _atomic_forces_from_edge_grad(data_dict, force_like)
        )
        loss = _loss_from_energy_forces(
            batch=batch,
            loss_fn=loss_fn,
            energy=energy,
            forces=forces,
        )
    if hasattr(batch, "forces") and forces.shape[0] != batch.forces.shape[0]:
        forces = forces[: batch.forces.shape[0]]
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
    compute_virials, compute_stress = _loss_required_output_flags(loss_fn)
    output = model(
        batch.to_dict(),
        training=True,
        compute_force=True,
        compute_virials=compute_virials,
        compute_stress=compute_stress,
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
    compute_virials, compute_stress = _loss_required_output_flags(loss_fn)
    output = model(
        batch.to_dict(),
        training=True,
        compute_force=True,
        compute_virials=compute_virials,
        compute_stress=compute_stress,
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
    graph_module: torch.fx.GraphModule
    gate_result: EdgeForceCompileGateResult
    cache_key: tuple
    input_names: tuple[str, ...]
    param_names: tuple[str, ...]
    loss_input_names: tuple[str, ...] = ()
    returns_loss: bool = False
    force_gradient_mode: str = "edge"
    setup_phase_seconds: dict[str, float] = dataclasses.field(default_factory=dict)


class EdgeForceCompiledLossModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, config: EdgeForceCompileConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.cache: BoundedLRUCache[tuple, _CompiledEdgeForceStep] = BoundedLRUCache(
            max_entries=config.max_cache_entries
        )
        self.cache_policy_state = EdgeForceCachePolicyState()
        self.parity_check_step = 0
        self.fixed_probe_step = 0
        self.fixed_probe_batch: _EdgeForceFrozenBatch | None = None
        self.fixed_probe_cache_key: tuple | None = None
        self.disabled = False
        self.functorch_donated_buffer_disabled = False


    def _cached_compiled_step(self, cache_key: tuple) -> _CompiledEdgeForceStep | None:
        return self.cache.get_lru(cache_key)

    def _store_compiled_step(
        self, cache_key: tuple, compiled: _CompiledEdgeForceStep
    ) -> None:
        self.cache.store(cache_key, compiled)

    def disable_compile_fallback(self, exc: Exception) -> bool:
        if not self.config.allow_fallback or not _is_safe_compile_fallback_exception(
            exc
        ):
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
        output_args: dict[str, bool] | None = None,
        disabled_reason: str = "disabled",
        policy_decision: EdgeForceCachePolicyDecision | None = None,
    ):
        if output_args is None:
            output_args = {"forces": True, "virials": False, "stress": False}
        output = self.model(
            batch.to_dict(),
            training=True,
            compute_force=bool(output_args.get("forces", True)),
            compute_virials=bool(output_args.get("virials", False)),
            compute_stress=bool(output_args.get("stress", False)),
        )
        metrics = {
            "edge_force_compile": False,
            "edge_force_compile_disabled": True,
            "edge_force_compile_disabled_reason": disabled_reason,
            "edge_force_num_atoms": int(batch.positions.shape[0]),
            "edge_force_num_edges": int(batch.edge_index.shape[1]),
        }
        if policy_decision is not None:
            metrics.update(
                {
                    "edge_force_compile_cache_policy": policy_decision.cache_policy,
                    "edge_force_cache_seen_count": policy_decision.seen_count,
                    "edge_force_cache_compile_count": policy_decision.compile_count,
                    "edge_force_cache_hit_count": policy_decision.cache_hit_count,
                    **_break_even_policy_metrics(policy_decision),
                }
            )
        return loss_fn(pred=output, ref=batch), metrics

    def _should_run_parity_check(self) -> bool:
        interval = int(self.config.parity_check_interval)
        if interval <= 0:
            return False
        self.parity_check_step += 1
        return self.parity_check_step % interval == 0

    def _should_run_fixed_probe(self) -> bool:
        interval = int(self.config.fixed_probe_interval)
        if interval <= 0:
            return False
        self.fixed_probe_step += 1
        return self.fixed_probe_step % interval == 0

    def _run_parity_check(
        self,
        *,
        batch,
        loss_fn,
        executable,
        compiled: _CompiledEdgeForceStep,
        bucket_sizes: tuple[int, int] | None,
    ) -> dict[str, Any]:
        try:
            if self.config.parity_check_gradients:
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
                    input_names=compiled.input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=compiled.param_names,
                    loss_input_names=compiled.loss_input_names,
                    force_gradient_mode=compiled.force_gradient_mode,
                )
            else:
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
                    input_names=compiled.input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=compiled.param_names,
                    loss_input_names=compiled.loss_input_names,
                    force_gradient_mode=compiled.force_gradient_mode,
                )
            return _compare_edge_force_snapshots(
                reference,
                candidate,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )
        finally:
            self.model.zero_grad(set_to_none=True)

    def _run_fixed_probe(
        self,
        *,
        loss_fn,
    ) -> dict[str, Any]:
        if self.fixed_probe_batch is None or self.fixed_probe_cache_key is None:
            return {"skipped": True, "skip_reason": "not_captured"}
        compiled = self.cache.get(self.fixed_probe_cache_key)
        if compiled is None:
            return {"skipped": True, "skip_reason": "cache_entry_missing"}
        if compiled.executable is None:
            return {"skipped": True, "skip_reason": "executable_released"}
        device = next(self.model.parameters()).device
        probe_batch = self.fixed_probe_batch.to(device)
        bucket_sizes = _edge_force_bucket_sizes_from_cache_key(compiled.cache_key)
        try:
            if self.config.fixed_probe_gradients:
                reference = _position_force_snapshot(
                    model=self.model,
                    batch=probe_batch,
                    loss_fn=loss_fn,
                )
                candidate = _edge_force_snapshot_from_executable(
                    model=self.model,
                    batch=probe_batch,
                    loss_fn=loss_fn,
                    executable=compiled.executable,
                    input_names=compiled.input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=compiled.param_names,
                    loss_input_names=compiled.loss_input_names,
                    force_gradient_mode=compiled.force_gradient_mode,
                )
            else:
                reference = _position_force_value_snapshot(
                    model=self.model,
                    batch=probe_batch,
                    loss_fn=loss_fn,
                )
                candidate = _edge_force_value_snapshot_from_executable(
                    model=self.model,
                    batch=probe_batch,
                    loss_fn=loss_fn,
                    executable=compiled.executable,
                    input_names=compiled.input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=compiled.param_names,
                    loss_input_names=compiled.loss_input_names,
                    force_gradient_mode=compiled.force_gradient_mode,
                )
            comparison = _compare_edge_force_snapshots(
                reference,
                candidate,
                atol=self.config.atol,
                rtol=self.config.rtol,
            )
            comparison["num_atoms"] = int(probe_batch.positions.shape[0])
            comparison["num_edges"] = int(probe_batch.edge_index.shape[1])
            return comparison
        finally:
            self.model.zero_grad(set_to_none=True)

    @staticmethod
    def _fixed_probe_metrics_from_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
        if comparison.get("skipped", False):
            return {
                "edge_force_fixed_probe_check": False,
                "edge_force_fixed_probe_skipped": True,
                "edge_force_fixed_probe_skip_reason": str(
                    comparison.get("skip_reason", "unknown")
                ),
            }
        grad_diffs = comparison.get("param_grad_max_abs_diff", {}) or {}
        worst_grad_name = "none"
        worst_grad_diff = 0.0
        for name, diff in grad_diffs.items():
            diff_value = float(diff)
            if worst_grad_name == "none" or diff_value > worst_grad_diff:
                worst_grad_name = str(name)
                worst_grad_diff = diff_value
        failed_checks = comparison.get("failed_checks", []) or []
        return {
            "edge_force_fixed_probe_check": True,
            "edge_force_fixed_probe_skipped": False,
            "edge_force_fixed_probe_accepted": bool(comparison.get("ok", False)),
            "edge_force_fixed_probe_num_atoms": int(comparison.get("num_atoms", 0)),
            "edge_force_fixed_probe_num_edges": int(comparison.get("num_edges", 0)),
            "edge_force_fixed_probe_energy_max_abs_diff": float(
                comparison.get("energy_max_abs_diff", 0.0)
            ),
            "edge_force_fixed_probe_forces_max_abs_diff": float(
                comparison.get("forces_max_abs_diff", 0.0)
            ),
            "edge_force_fixed_probe_loss_abs_diff": float(
                comparison.get("loss_abs_diff", 0.0)
            ),
            "edge_force_fixed_probe_param_grad_max_abs_diff": worst_grad_diff,
            "edge_force_fixed_probe_param_grad_worst": worst_grad_name,
            "edge_force_fixed_probe_failed_checks": len(failed_checks),
            "edge_force_fixed_probe_failed_check_names": ",".join(
                map(str, failed_checks[:8])
            ),
        }


    def _cache_abi_signature(self) -> tuple:
        parameter_signature = tuple(
            (
                name,
                tuple(int(dim) for dim in param.shape),
                str(param.dtype),
                bool(param.requires_grad),
            )
            for name, param in self.model.named_parameters()
        )
        buffer_signature = tuple(
            (
                name,
                tuple(int(dim) for dim in buffer.shape),
                str(buffer.dtype),
            )
            for name, buffer in self.model.named_buffers()
        )
        return (
            "abi",
            ("model_type", type(self.model).__module__, type(self.model).__qualname__),
            ("torch", str(torch.__version__)),
            ("cuda", str(getattr(torch.version, "cuda", None))),
            _edge_force_dependency_abi_signature(),
            _edge_force_distributed_abi_signature(),
            _edge_force_autocast_abi_signature(),
            ("deterministic", bool(torch.are_deterministic_algorithms_enabled())),
            ("compile_graph", bool(self.config.compile_graph)),
            ("compile_mode", self.config.compile_mode),
            ("compile_dynamic", bool(self.config.compile_dynamic)),
            ("compile_shape_padding", bool(self.config.compile_shape_padding)),
            ("compile_max_fusion_size", int(self.config.compile_max_fusion_size)),
            ("strip_detach", bool(self.config.strip_detach)),
            (
                "use_e3nn_spherical_harmonics",
                bool(self.config.use_e3nn_spherical_harmonics),
            ),
            ("force_gradient_mode", self.config.force_gradient_mode),
            ("parameters", parameter_signature),
            ("buffers", buffer_signature),
        )

    @staticmethod
    def _parity_metrics_from_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
        grad_diffs = comparison.get("param_grad_max_abs_diff", {}) or {}
        worst_grad_name = "none"
        worst_grad_diff = 0.0
        for name, diff in grad_diffs.items():
            diff_value = float(diff)
            if worst_grad_name == "none" or diff_value > worst_grad_diff:
                worst_grad_name = str(name)
                worst_grad_diff = diff_value
        failed_checks = comparison.get("failed_checks", []) or []
        return {
            "edge_force_parity_check": True,
            "edge_force_parity_accepted": bool(comparison.get("ok", False)),
            "edge_force_parity_energy_max_abs_diff": float(
                comparison.get("energy_max_abs_diff", 0.0)
            ),
            "edge_force_parity_forces_max_abs_diff": float(
                comparison.get("forces_max_abs_diff", 0.0)
            ),
            "edge_force_parity_loss_abs_diff": float(
                comparison.get("loss_abs_diff", 0.0)
            ),
            "edge_force_parity_param_grad_max_abs_diff": worst_grad_diff,
            "edge_force_parity_param_grad_worst": worst_grad_name,
            "edge_force_parity_failed_checks": len(failed_checks),
            "edge_force_parity_failed_check_names": ",".join(map(str, failed_checks[:8])),
        }

    def _cache_key(
        self,
        *,
        batch,
        input_names: tuple[str, ...],
        data_dict: dict[str, torch.Tensor],
        loss_fn: torch.nn.Module | None = None,
        output_args: dict[str, bool] | None = None,
    ) -> tuple | None:
        input_shapes = {name: tuple(data_dict[name].shape) for name in input_names}
        input_abi_signature = (
            "inputs",
            tuple(
                (
                    name,
                    str(data_dict[name].dtype),
                    _edge_force_device_abi_signature(data_dict[name].device),
                    bool(data_dict[name].requires_grad),
                )
                for name in input_names
            ),
        )
        loss_abi_signature = (
            ("compiled_loss", _compiled_tensor_loss_kind(loss_fn))
            if loss_fn is not None and _edge_force_can_compile_loss(loss_fn, data_dict.keys())
            else ("compiled_loss", "none")
        )
        requested_outputs_signature = (
            "requested_outputs",
            tuple(
                (name, bool((output_args or {}).get(name, default)))
                for name, default in (
                    ("forces", True),
                    ("stress", False),
                    ("virials", False),
                )
            ),
        )
        abi_signature = (
            *self._cache_abi_signature(),
            loss_abi_signature,
            requested_outputs_signature,
            input_abi_signature,
        )
        if self.config.cache_policy == "bucket":
            return edge_force_compile_bucket_cache_key(
                num_atoms=batch.positions.shape[0],
                num_edges=batch.edge_index.shape[1],
                input_shapes=input_shapes,
                bucket_atoms=self.config.bucket_atoms,
                bucket_edges=self.config.bucket_edges,
                bucket_margin=self.config.bucket_margin,
                abi_signature=abi_signature,
            )
        if self.config.cache_policy == "dynamic":
            return edge_force_compile_dynamic_cache_key(
                input_shapes=input_shapes,
                abi_signature=abi_signature,
            )
        return edge_force_compile_shape_cache_key(
            num_atoms=batch.positions.shape[0],
            num_edges=batch.edge_index.shape[1],
            input_shapes=input_shapes,
            abi_signature=abi_signature,
        )

    def _compile_step(self, *, batch, loss_fn, cache_key: tuple) -> _CompiledEdgeForceStep:
        phase_seconds: dict[str, float] = {}

        def log_phase_start(phase_name: str) -> float:
            logging.info(
                "Edge-force compile setup phase %s start: mode=%s cache_key=%s",
                phase_name,
                self.config.force_gradient_mode,
                cache_key,
            )
            return time.perf_counter()

        def log_phase_done(phase_name: str, phase_start_time: float) -> None:
            phase_seconds[phase_name] = time.perf_counter() - phase_start_time
            logging.info(
                "Edge-force compile setup phase %s done: %.3fs mode=%s cache_key=%s",
                phase_name,
                phase_seconds[phase_name],
                self.config.force_gradient_mode,
                cache_key,
            )

        phase_start = log_phase_start("input_prep")
        bucket_sizes = _edge_force_bucket_sizes_from_cache_key(cache_key)
        if bucket_sizes is None:
            data_dict, positions, _, vectors = _edge_vector_inputs(batch)
        else:
            data_dict, positions, _, vectors = _edge_vector_inputs(
                batch,
                atom_bucket=bucket_sizes[0],
                edge_bucket=bucket_sizes[1],
                r_max=_edge_force_model_r_max(self.model),
            )
        gradient_input = positions if self.config.force_gradient_mode == "positions" else vectors
        input_names = edge_force_compile_input_names(
            data_dict.keys(), force_gradient_mode=self.config.force_gradient_mode
        )
        compile_loss = _edge_force_can_compile_loss(loss_fn, data_dict.keys())
        compiled_loss_kind = (
            _compiled_tensor_loss_kind(loss_fn) if compile_loss else ""
        )
        loss_input_names = (
            edge_force_compile_loss_input_names(data_dict.keys(), loss_fn=loss_fn)
            if compile_loss
            else ()
        )
        param_names = _compiled_parameter_names(self.model)
        example_inputs = tuple(
            _edge_force_executable_inputs(
                model=self.model,
                data_dict=data_dict,
                param_names=param_names,
                input_names=input_names,
                loss_input_names=loss_input_names,
                loss_fn=loss_fn,
            )
        )
        log_phase_done("input_prep", phase_start)

        def closure(gradient_arg: torch.Tensor, *all_tensors: torch.Tensor):
            param_end = len(param_names)
            data_end = param_end + len(input_names)
            loss_end = data_end + len(loss_input_names)
            param_tensors = all_tensors[:param_end]
            data_tensors = all_tensors[param_end:data_end]
            loss_tensors = all_tensors[data_end:loss_end]
            loss_weights = all_tensors[loss_end:]
            current_data = dict(data_dict)
            current_data.update(zip(input_names, data_tensors, strict=True))
            current_data.update(zip(loss_input_names, loss_tensors, strict=True))
            saved_params = _replace_module_parameters_with_tensors(
                self.model, param_names, param_tensors
            )
            try:
                if self.config.force_gradient_mode == "positions":
                    compute_stress_loss = compiled_loss_kind in {
                        "weighted_energy_forces_stress",
                        "weighted_huber_energy_forces_stress",
                        "universal",
                    }
                    compute_virials_loss = compiled_loss_kind == "weighted_energy_forces_virials"
                    if compute_stress_loss or compute_virials_loss:
                        energy, forces, stress, virials = _position_model_outputs(
                            self.model,
                            current_data,
                            gradient_arg,
                            compute_virials=compute_virials_loss,
                            compute_stress=compute_stress_loss,
                        )
                    else:
                        energy, forces = _position_force_energy_and_forces(
                            self.model,
                            current_data,
                            gradient_arg,
                            use_e3nn_spherical_harmonics=self.config.use_e3nn_spherical_harmonics,
                        )
                        stress = None
                        virials = None
                    if compile_loss:
                        loss = _edge_force_compiled_tensor_loss(
                            loss_kind=compiled_loss_kind,
                            data_dict=current_data,
                            energy=energy,
                            forces=forces,
                            stress=stress,
                            virials=virials,
                            loss_weights=loss_weights,
                        )
                        return energy, forces, loss
                    return energy, forces
                energy, edge_grad = _edge_force_energy_and_edge_grad(
                    self.model,
                    current_data,
                    gradient_arg,
                    use_e3nn_spherical_harmonics=self.config.use_e3nn_spherical_harmonics,
                )
                if compile_loss:
                    forces = _atomic_forces_from_edge_grad(current_data, edge_grad)
                    loss = _edge_force_compiled_tensor_loss(
                        loss_kind=compiled_loss_kind,
                        data_dict=current_data,
                        energy=energy,
                        forces=forces,
                        loss_weights=loss_weights,
                    )
                    return energy, forces, loss
                return energy, edge_grad
            finally:
                _restore_module_parameters(saved_params)

        phase_start = log_phase_start("trace")
        trace_result = trace_force_closure(
            closure,
            (gradient_input, *example_inputs),
            tracing_mode=self.config.tracing_mode,
            strip_detach=self.config.strip_detach,
            strip_all_detach=True,
        )
        log_phase_done("trace", phase_start)

        gate_graph_module = (
            rebuild_fx_graph_module(trace_result.graph_module)
            if self.config.compile_graph
            else trace_result.graph_module
        )
        phase_start = log_phase_start("gate_compile")
        executable, compile_kwargs = compile_fx_graph_module(
            gate_graph_module,
            compile_graph=self.config.compile_graph,
            compile_mode=self.config.compile_mode,
            compile_dynamic=self.config.compile_dynamic,
            shape_padding=self.config.compile_shape_padding,
            max_fusion_size=self.config.compile_max_fusion_size,
        )
        log_phase_done("gate_compile", phase_start)

        if self.config.setup_gate == "none":
            phase_seconds["gate_reference"] = 0.0
            phase_seconds["gate_candidate"] = 0.0
            gate_result = EdgeForceCompileGateResult(
                enabled=True,
                accepted=None,
                fallback_reason="setup_gate_skipped",
                detach_nodes_before=trace_result.detach_nodes_before,
                detach_nodes_after=trace_result.detach_nodes_after,
                node_count=len(list(trace_result.graph_module.graph.nodes)),
                comparison={"setup_gate": "none", "skipped": True},
                compile_kwargs=compile_kwargs,
            )
        else:
            phase_start = log_phase_start("gate_reference")
            if self.config.compile_graph:
                reference = _position_force_value_snapshot(
                    model=self.model,
                    batch=batch,
                    loss_fn=loss_fn,
                )
            else:
                reference = _position_force_snapshot(
                    model=self.model,
                    batch=batch,
                    loss_fn=loss_fn,
                )
            log_phase_done("gate_reference", phase_start)

            phase_start = log_phase_start("gate_candidate")
            if self.config.compile_graph:
                candidate = _edge_force_value_snapshot_from_executable(
                    model=self.model,
                    batch=batch,
                    loss_fn=loss_fn,
                    executable=executable,
                    input_names=input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=param_names,
                    loss_input_names=loss_input_names,
                    force_gradient_mode=self.config.force_gradient_mode,
                )
            else:
                candidate = _edge_force_snapshot_from_executable(
                    model=self.model,
                    batch=batch,
                    loss_fn=loss_fn,
                    executable=executable,
                    input_names=input_names,
                    bucket_sizes=bucket_sizes,
                    param_names=param_names,
                    loss_input_names=loss_input_names,
                    force_gradient_mode=self.config.force_gradient_mode,
                )
            log_phase_done("gate_candidate", phase_start)

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

        cached_graph_module = trace_result.graph_module
        training_executable = executable
        phase_seconds["training_compile"] = 0.0
        if self.config.compile_graph:
            # Each torch.compile call gets its own GraphModule copy.  PyTorch 2.10
            # can retain higher-order autograd state on compiled force callables,
            # so the cached source graph must never be one already handed to compile.
            cached_graph_module = rebuild_fx_graph_module(trace_result.graph_module)
            training_graph_module = rebuild_fx_graph_module(trace_result.graph_module)
            phase_start = log_phase_start("training_compile")
            training_executable, _ = compile_fx_graph_module(
                training_graph_module,
                compile_graph=self.config.compile_graph,
                compile_mode=self.config.compile_mode,
                compile_dynamic=self.config.compile_dynamic,
                shape_padding=self.config.compile_shape_padding,
                max_fusion_size=self.config.compile_max_fusion_size,
            )
            log_phase_done("training_compile", phase_start)

        self.model.zero_grad(set_to_none=True)
        return _CompiledEdgeForceStep(
            executable=training_executable,
            graph_module=cached_graph_module,
            gate_result=gate_result,
            cache_key=cache_key,
            input_names=input_names,
            param_names=param_names,
            loss_input_names=loss_input_names,
            returns_loss=compile_loss,
            force_gradient_mode=self.config.force_gradient_mode,
            setup_phase_seconds=phase_seconds,
        )

    def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
        if self.disabled:
            return self._eager_force_loss(
                batch=batch, loss_fn=loss_fn, output_args=output_args
            )
        requested_extra_outputs = bool(output_args.get("virials", False)) or bool(
            output_args.get("stress", False)
        )
        if requested_extra_outputs and self.config.force_gradient_mode != "positions":
            return self._eager_force_loss(
                batch=batch,
                loss_fn=loss_fn,
                output_args=output_args,
                disabled_reason="unsupported_outputs",
            )
        loss_capability = edge_force_loss_output_capability(loss_fn)
        if not loss_capability.edge_force_supported and not (
            self.config.force_gradient_mode == "positions"
            and loss_capability.compiled_tensor_loss_supported
        ):
            return self._eager_force_loss(
                batch=batch,
                loss_fn=loss_fn,
                output_args=output_args,
                disabled_reason=loss_capability.unsupported_reason or "unsupported_loss",
            )
        try:
            data_dict, _, _, vectors = _edge_vector_inputs(batch)
            input_names = edge_force_compile_input_names(
                data_dict.keys(), force_gradient_mode=self.config.force_gradient_mode
            )
            if requested_extra_outputs and not _edge_force_can_compile_loss(
                loss_fn, data_dict.keys()
            ):
                return self._eager_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    output_args=output_args,
                    disabled_reason="unsupported_loss_inputs",
                )
            cache_key = self._cache_key(
                batch=batch,
                input_names=input_names,
                data_dict=data_dict,
                loss_fn=loss_fn,
                output_args=output_args,
            )
            if self.config.cache_policy == "bucket":
                if not self.config.bucket_atoms or not self.config.bucket_edges:
                    return self._eager_force_loss(
                        batch=batch,
                        loss_fn=loss_fn,
                        output_args=output_args,
                        disabled_reason="no_bucket",
                    )
                if cache_key is None:
                    return self._eager_force_loss(
                        batch=batch,
                        loss_fn=loss_fn,
                        output_args=output_args,
                        disabled_reason="no_bucket_match",
                    )
                bucket_sizes = _edge_force_bucket_sizes_from_cache_key(cache_key)
                if bucket_sizes is None:
                    raise RuntimeError(f"invalid edge-force bucket cache key: {cache_key}")
                data_dict, _, _, vectors = _edge_vector_inputs(
                    batch,
                    atom_bucket=bucket_sizes[0],
                    edge_bucket=bucket_sizes[1],
                    r_max=_edge_force_model_r_max(self.model),
                )
                input_names = edge_force_compile_input_names(
                    data_dict.keys(), force_gradient_mode=self.config.force_gradient_mode
                )
                cache_key = self._cache_key(
                    batch=batch,
                    input_names=input_names,
                    data_dict=data_dict,
                    loss_fn=loss_fn,
                    output_args=output_args,
                )
            expected_returns_loss = _edge_force_can_compile_loss(loss_fn, data_dict.keys())
            compiled = self._cached_compiled_step(cache_key)
            if compiled is not None and compiled.returns_loss != expected_returns_loss:
                self.cache.pop(cache_key, None)
                compiled = None
            cache_hit = compiled is not None
            policy_decision = self.cache_policy_state.record_and_decide(
                cache_key,
                policy=self.config.cache_policy,
                min_repeats=self.config.min_repeats,
                cache_hit=cache_hit,
                break_even_expected_remaining_hits=(
                    self.config.break_even_expected_remaining_hits
                ),
            )
            if not policy_decision.compile_allowed:
                eager_start = time.perf_counter()
                loss, metrics = self._eager_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    output_args=output_args,
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
                bucket_sizes = _edge_force_bucket_sizes_from_cache_key(cache_key)
                if bucket_sizes is not None:
                    metrics["edge_force_bucket_atoms"] = bucket_sizes[0]
                    metrics["edge_force_bucket_edges"] = bucket_sizes[1]
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
                self._store_compiled_step(cache_key, compiled)
            if self.fixed_probe_batch is None and int(self.config.fixed_probe_interval) > 0:
                self.fixed_probe_batch = _freeze_edge_force_batch(batch)
                self.fixed_probe_cache_key = cache_key
            if compiled.input_names != input_names:
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
                        bucket_sizes=_edge_force_bucket_sizes_from_cache_key(compiled.cache_key),
                        param_names=compiled.param_names,
                        loss_input_names=compiled.loss_input_names,
                        force_gradient_mode=compiled.force_gradient_mode,
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
                        bucket_sizes=_edge_force_bucket_sizes_from_cache_key(compiled.cache_key),
                        param_names=compiled.param_names,
                        loss_input_names=compiled.loss_input_names,
                        force_gradient_mode=compiled.force_gradient_mode,
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
            runtime_recompile_seconds: float | None = None
            release_cached_executable = (
                cache_hit
                and self.config.compile_graph
                and self.config.refresh_executable_each_step
            )
            if release_cached_executable:
                compiled.executable = None
            runtime_executable = compiled.executable
            if release_cached_executable:
                runtime_recompile_start = time.perf_counter()
                runtime_graph_module = rebuild_fx_graph_module(compiled.graph_module)
                runtime_executable, _ = compile_fx_graph_module(
                    runtime_graph_module,
                    compile_graph=self.config.compile_graph,
                    compile_mode=self.config.compile_mode,
                    compile_dynamic=self.config.compile_dynamic,
                    shape_padding=self.config.compile_shape_padding,
                    max_fusion_size=self.config.compile_max_fusion_size,
                )
                runtime_recompile_seconds = time.perf_counter() - runtime_recompile_start
            if runtime_executable is None:
                raise RuntimeError("edge-force compiled executable is unavailable")

            parity_comparison = None
            fixed_probe_comparison = None
            bucket_sizes = _edge_force_bucket_sizes_from_cache_key(cache_key)
            if self._should_run_parity_check():
                parity_comparison = self._run_parity_check(
                    batch=batch,
                    loss_fn=loss_fn,
                    executable=runtime_executable,
                    compiled=compiled,
                    bucket_sizes=bucket_sizes,
                )
                if (
                    self.config.parity_check_strict
                    and not bool(parity_comparison.get("ok", False))
                ):
                    raise RuntimeError(
                        f"edge-force compile periodic parity check failed: {parity_comparison}"
                    )
            if self._should_run_fixed_probe():
                fixed_probe_comparison = self._run_fixed_probe(loss_fn=loss_fn)
                if (
                    self.config.fixed_probe_strict
                    and not bool(fixed_probe_comparison.get("ok", False))
                ):
                    raise RuntimeError(
                        f"edge-force compile fixed probe failed: {fixed_probe_comparison}"
                    )

            compiled_start = time.perf_counter()
            gradient_input = (
                data_dict["positions"]
                if compiled.force_gradient_mode == "positions"
                else vectors
            )
            gradient_input = gradient_input.detach().clone()
            inputs = _edge_force_executable_inputs(
                model=self.model,
                data_dict=data_dict,
                param_names=compiled.param_names,
                input_names=compiled.input_names,
                loss_input_names=compiled.loss_input_names,
                loss_fn=loss_fn,
            )
            executable_outputs = runtime_executable(gradient_input, *inputs)
            if compiled.returns_loss:
                energy, forces, loss = executable_outputs
            else:
                energy, force_like = executable_outputs
                forces = (
                    force_like
                    if compiled.force_gradient_mode == "positions"
                    else _atomic_forces_from_edge_grad(data_dict, force_like)
                )
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
                "edge_force_compile_loss": compiled.returns_loss,
                "edge_force_gradient_mode": compiled.force_gradient_mode,
                "edge_force_cache_hit": cache_hit,
                "edge_force_gate_accepted": compiled.gate_result.accepted,
                "edge_force_setup_gate": self.config.setup_gate,
                "edge_force_num_atoms": int(batch.positions.shape[0]),
                "edge_force_num_edges": int(batch.edge_index.shape[1]),
                "edge_force_compile_cache_policy": policy_decision.cache_policy,
                "edge_force_cache_seen_count": policy_decision.seen_count,
                "edge_force_cache_compile_count": stats.compile_count,
                "edge_force_cache_hit_count": stats.cache_hit_count,
                **_break_even_policy_metrics(policy_decision),
                "edge_force_compile_setup_seconds": stats.compile_setup_seconds,
                **{
                    f"edge_force_compile_{phase_name}_seconds": phase_seconds
                    for phase_name, phase_seconds in compiled.setup_phase_seconds.items()
                },
                "edge_force_compiled_step_seconds_ema": stats.compiled_step_seconds_ema,
                "edge_force_eager_step_seconds_ema": stats.eager_step_seconds_ema,
                "edge_force_runtime_recompile": runtime_recompile_seconds is not None,
                "edge_force_runtime_recompile_seconds": runtime_recompile_seconds,
            }
            if bucket_sizes is not None:
                metrics["edge_force_bucket_atoms"] = bucket_sizes[0]
                metrics["edge_force_bucket_edges"] = bucket_sizes[1]
            if self.config.compile_graph and compiled.returns_loss:
                metrics["_retain_graph_for_backward"] = True
            if parity_comparison is not None:
                metrics.update(self._parity_metrics_from_comparison(parity_comparison))
            if fixed_probe_comparison is not None:
                metrics.update(
                    self._fixed_probe_metrics_from_comparison(fixed_probe_comparison)
                )
            if cache_hit_gate_accepted is not None:
                metrics["edge_force_cache_hit_gate_accepted"] = cache_hit_gate_accepted
            return loss, metrics
        except Exception as exc:
            if not self.disable_compile_fallback(exc):
                raise
            return self._eager_force_loss(
                batch=batch, loss_fn=loss_fn, output_args=output_args
            )


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
        if not self.allow_fallback or not _is_safe_compile_fallback_exception(exc):
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
        full_compiled_outputs = bool(
            kwargs.get("compute_virials", False)
            or kwargs.get("compute_stress", False)
        )
        unsupported_force_outputs = any(
            kwargs.get(name, False)
            for name in (
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
            if full_compiled_outputs or not compute_force:
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
            from mace.modules.utils import get_outputs, prepare_graph

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
    if not _is_scale_shift_mace(model):
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
