from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.optim import _functional as optim_functional


_NS_EPS = 1.0e-7
_NS_COEFF_FAST = (3.4445, -4.7750, 2.0315)
_NS_COEFF_POLISH = (2.0, -1.5, 0.5)
_NS_STEPS_FAST = 8
_NS_STEPS_POLISH = 2
_MAGMA_EPS = 1.0e-12
_MAGMA_TAU = 2.0
_MAGMA_EMA_DECAY = 0.9
_MAGMA_MIN_SCALE = 0.1
_MAGMA_SIGMOID_MIN = 1.0 / (1.0 + math.exp(1.0 / _MAGMA_TAU))
_MAGMA_SIGMOID_MAX = 1.0 / (1.0 + math.exp(-1.0 / _MAGMA_TAU))


@dataclass(frozen=True)
class OptimSpec:
    route: str
    matrix_axes: tuple[int, int] | None = None
    batch_axes: tuple[int, ...] = ()
    slice_specs: tuple[dict, ...] = ()
    semantic_axes: tuple[str, ...] = ()
    matrix_structure: str = "real"
    min_matrix_dim: int = 1
    max_aspect_ratio: float | None = None
    lr_scale: float = 1.0
    weight_decay: float | None = None
    spec_version: int = 1

    def __post_init__(self) -> None:
        if self.route not in {"muon", "adam", "adamw"}:
            raise ValueError("OptimSpec.route must be 'muon', 'adam', or 'adamw'")
        if self.matrix_structure not in _OPTIM_SPEC_MATRIX_STRUCTURES:
            raise ValueError(
                "OptimSpec.matrix_structure must be one of "
                f"{sorted(_OPTIM_SPEC_MATRIX_STRUCTURES)}"
            )
        if self.min_matrix_dim <= 0:
            raise ValueError("OptimSpec.min_matrix_dim must be positive")
        if self.max_aspect_ratio is not None and self.max_aspect_ratio <= 0.0:
            raise ValueError("OptimSpec.max_aspect_ratio must be positive")
        if self.lr_scale <= 0.0:
            raise ValueError("OptimSpec.lr_scale must be positive")
        if self.weight_decay is not None and self.weight_decay < 0.0:
            raise ValueError("OptimSpec.weight_decay must be non-negative")
        if self.spec_version <= 0:
            raise ValueError("OptimSpec.spec_version must be positive")


@dataclass(frozen=True)
class RouteRecord:
    name: str
    shape: tuple[int, ...]
    numel: int
    route: str
    reason: str
    muon_mode: str | None = None
    matrix_shape: tuple[int, int] | None = None
    matrix_batch: int | None = None
    lr_scale: float | None = None

    def as_summary(self) -> dict:
        item = {
            "name": self.name,
            "shape": self.shape,
            "numel": self.numel,
            "route": self.route,
            "reason": self.reason,
        }
        if self.route == "muon":
            item["muon_mode"] = self.muon_mode
            item["matrix_shape"] = self.matrix_shape
            item["matrix_batch"] = self.matrix_batch
            if self.lr_scale is not None:
                item["lr_scale"] = self.lr_scale
        return item


_ADAM_NAME_TOKENS = (
    "bias",
    "norm",
    "scale",
    "shift",
    "atomic_energies",
    "atomic_energy",
    "embedding",
    "readout",
    "readouts",
    "selector",
    "core",
    "products",
    "symmetric_contractions",
    "contractions",
    "skip_tp",
    "linear_up",
    "linear_down",
)

_MUON_NAME_TOKENS = (
    "radial",
    "mlp",
    "fitting",
)

_EQUIVARIANT_SLICE_TOKENS = (
    "symmetric_contractions",
    "contractions",
)

_MUON_MODES = {"2d", "slice"}
_ROUTINGS = {"mace", "tace", "module"}
_ADAM_VARIANTS = {"adam", "adamw"}
_MUON_LR_SCALE_MODES = {"original", "match_rms", "none"}
_OPTIM_SPEC_MATRIX_STRUCTURES = {
    "real",
    "complex",
    "shared_complex",
    "diagonal",
    "scalar_coeff",
}
_FORBIDDEN_MUON_MATRIX_SEMANTIC_AXES = {
    "degree",
    "ell",
    "l",
    "m",
    "parity",
    "path",
    "correlation",
    "species",
    "focus",
    "head",
    "expert",
}
_ALLOWED_MUON_MATRIX_SEMANTIC_AXES = {
    "channel",
    "channel_in",
    "channel_out",
    "multiplicity",
    "multiplicity_in",
    "multiplicity_out",
    "feature",
    "feature_in",
    "feature_out",
}
_MACE_HARD_ADAM_NAME_TOKENS = (
    "bias",
    "norm",
    "scale",
    "shift",
    "atomic_energies",
    "atomic_energy",
    "embedding",
    "readout",
    "readouts",
    "selector",
    "core",
    "alpha",
    "beta",
    "affine",
)


def _optim_spec_manifest_contract(
    optim_spec: OptimSpec, param: torch.nn.Parameter
) -> dict:
    ndim = int(param.ndim)
    return {
        "route": str(optim_spec.route),
        "matrix_axes": (
            [
                _normalize_axis(axis, ndim, label="matrix_axes")
                for axis in optim_spec.matrix_axes
            ]
            if optim_spec.matrix_axes is not None
            else None
        ),
        "batch_axes": [
            _normalize_axis(axis, ndim, label="batch_axes")
            for axis in optim_spec.batch_axes
        ],
        "semantic_axes": [str(axis) for axis in optim_spec.semantic_axes],
        "matrix_structure": str(optim_spec.matrix_structure),
        "min_matrix_dim": int(optim_spec.min_matrix_dim),
        "max_aspect_ratio": (
            float(optim_spec.max_aspect_ratio)
            if optim_spec.max_aspect_ratio is not None
            else None
        ),
        "lr_scale": float(optim_spec.lr_scale),
        "weight_decay": (
            float(optim_spec.weight_decay)
            if optim_spec.weight_decay is not None
            else None
        ),
        "spec_version": int(optim_spec.spec_version),
    }


def _normalize_tace_module_include(
    patterns: str | Iterable[str] | None,
) -> tuple[str, ...]:
    if patterns is None:
        return ("*",)
    if isinstance(patterns, str):
        items = patterns.split(",")
    else:
        items = list(patterns)
    normalized = tuple(str(item).strip() for item in items if str(item).strip())
    return normalized or ("*",)


def _matches_tace_module_include(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _effective_shape(param: torch.nn.Parameter | torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in param.shape if int(dim) != 1)


def _matrix_view_shape(
    shape: tuple[int, ...], muon_mode: str
) -> tuple[int, int, int] | None:
    effective_shape = tuple(int(dim) for dim in shape if int(dim) != 1)
    if len(effective_shape) < 2:
        return None
    if muon_mode == "2d":
        if len(effective_shape) != 2:
            return None
        return (1, effective_shape[-2], effective_shape[-1])
    if muon_mode == "slice":
        batch = math.prod(effective_shape[:-2]) if len(effective_shape) > 2 else 1
        return (int(batch), effective_shape[-2], effective_shape[-1])
    raise ValueError(f"unknown HybridMuon mode: {muon_mode}")


def _is_sharded_or_distributed_parameter(
    param: torch.nn.Parameter | torch.Tensor,
) -> bool:
    if bool(getattr(param, "_is_sharded", False)):
        return True
    tensor = getattr(param, "data", param)
    if bool(getattr(tensor, "_is_sharded", False)):
        return True
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None
    if DTensor is not None and (isinstance(param, DTensor) or isinstance(tensor, DTensor)):
        return True
    for attr in ("_dtensor_spec", "_local_tensor", "device_mesh", "placements"):
        if hasattr(param, attr) or hasattr(tensor, attr):
            return True
    class_names = {type(param).__name__, type(tensor).__name__}
    return bool(class_names & {"DTensor", "ShardedTensor"})


def _normalize_axis(axis: int, ndim: int, *, label: str) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{label} axis {axis} out of bounds for rank-{ndim} tensor")
    return axis


def _validate_optim_spec_muon_contract(
    name: str, param: torch.nn.Parameter, optim_spec: OptimSpec
) -> None:
    if optim_spec.route != "muon":
        return
    if optim_spec.matrix_structure != "real":
        raise ValueError(
            f"OptimSpec for {name!r} has matrix_structure="
            f"{optim_spec.matrix_structure!r}; route it to AdamW until a "
            "matching Muon structure is implemented"
        )
    if not optim_spec.semantic_axes:
        return
    ndim = int(param.ndim)
    if len(optim_spec.semantic_axes) != ndim:
        raise ValueError(
            f"OptimSpec.semantic_axes for {name!r} must have length {ndim}, "
            f"got {len(optim_spec.semantic_axes)}"
        )
    if optim_spec.matrix_axes is None:
        return
    matrix_axes = tuple(
        _normalize_axis(axis, ndim, label="matrix_axes")
        for axis in optim_spec.matrix_axes
    )
    for axis in matrix_axes:
        semantic_axis = str(optim_spec.semantic_axes[axis]).strip()
        if not semantic_axis:
            raise ValueError(
                f"OptimSpec.semantic_axes for {name!r} includes an empty axis label"
            )
        if semantic_axis in _FORBIDDEN_MUON_MATRIX_SEMANTIC_AXES:
            raise ValueError(
                f"OptimSpec for {name!r} cannot use semantic axis "
                f"{semantic_axis!r} as a Muon matrix axis"
            )
        if semantic_axis not in _ALLOWED_MUON_MATRIX_SEMANTIC_AXES:
            raise ValueError(
                f"OptimSpec for {name!r} cannot route unknown semantic axis "
                f"{semantic_axis!r} to Muon"
            )


def _normalize_optim_spec_slice_specs(
    name: str, param: torch.nn.Parameter, optim_spec: OptimSpec
) -> list[dict] | None:
    if not optim_spec.slice_specs:
        return None
    if optim_spec.route != "muon":
        raise ValueError(
            f"OptimSpec.slice_specs for {name!r} are only valid for route='muon'"
        )
    if optim_spec.matrix_axes is not None or optim_spec.batch_axes:
        raise ValueError(
            f"OptimSpec.slice_specs for {name!r} cannot be combined with "
            "matrix_axes or batch_axes"
        )

    specs: list[dict] = []
    for index, raw_spec in enumerate(optim_spec.slice_specs):
        if not isinstance(raw_spec, dict):
            raise ValueError(
                f"OptimSpec.slice_specs[{index}] for {name!r} must be a dict"
            )
        try:
            offset = int(raw_spec["offset"])
            numel = int(raw_spec["numel"])
            matrix_view_shape = tuple(
                int(dim) for dim in raw_spec["matrix_view_shape"]
            )
        except KeyError as exc:
            raise ValueError(
                f"OptimSpec.slice_specs[{index}] for {name!r} is missing {exc.args[0]!r}"
            ) from exc
        if offset < 0 or numel <= 0:
            raise ValueError(
                f"OptimSpec.slice_specs[{index}] for {name!r} has invalid "
                f"offset={offset} or numel={numel}"
            )
        if len(matrix_view_shape) != 3 or any(dim <= 0 for dim in matrix_view_shape):
            raise ValueError(
                f"OptimSpec.slice_specs[{index}] for {name!r} must declare a "
                "positive (batch, rows, cols) matrix_view_shape"
            )
        if math.prod(matrix_view_shape) != numel:
            raise ValueError(
                f"OptimSpec.slice_specs[{index}] for {name!r} has numel={numel} "
                f"but matrix_view_shape product={math.prod(matrix_view_shape)}"
            )
        normalized = {
            "offset": offset,
            "numel": numel,
            "matrix_view_shape": matrix_view_shape,
            "source": "module_slice_spec",
        }
        if "path" in raw_spec:
            path = tuple(int(item) for item in raw_spec["path"])
            normalized["path"] = path
        if "source_shape" in raw_spec:
            source_shape = tuple(int(dim) for dim in raw_spec["source_shape"])
            if not source_shape or any(dim <= 0 for dim in source_shape):
                raise ValueError(
                    f"OptimSpec.slice_specs[{index}] for {name!r} has invalid source_shape"
                )
            if math.prod(source_shape) != numel:
                raise ValueError(
                    f"OptimSpec.slice_specs[{index}] for {name!r} has numel={numel} "
                    f"but source_shape product={math.prod(source_shape)}"
                )
            try:
                permute = tuple(int(dim) for dim in raw_spec["permute"])
                inverse_permute = tuple(
                    int(dim) for dim in raw_spec["inverse_permute"]
                )
            except KeyError as exc:
                raise ValueError(
                    f"OptimSpec.slice_specs[{index}] for {name!r} with source_shape "
                    f"is missing {exc.args[0]!r}"
                ) from exc
            expected_axes = tuple(range(len(source_shape)))
            if sorted(permute) != list(expected_axes) or sorted(inverse_permute) != list(
                expected_axes
            ):
                raise ValueError(
                    f"OptimSpec.slice_specs[{index}] for {name!r} has invalid "
                    "permute/inverse_permute axes"
                )
            normalized.update(
                {
                    "source_shape": source_shape,
                    "permute": permute,
                    "inverse_permute": inverse_permute,
                }
            )
        specs.append(normalized)

    specs.sort(key=lambda spec: int(spec["offset"]))
    cursor = 0
    for index, spec in enumerate(specs):
        offset = int(spec["offset"])
        numel = int(spec["numel"])
        if offset != cursor:
            raise ValueError(
                f"OptimSpec.slice_specs for {name!r} must cover the flattened "
                f"parameter exactly once; slice {index} starts at {offset}, "
                f"expected {cursor}"
            )
        cursor = offset + numel
    if cursor != int(param.numel()):
        raise ValueError(
            f"OptimSpec.slice_specs for {name!r} cover {cursor} values, "
            f"but parameter has {int(param.numel())}"
        )
    return specs


def _optim_spec_matrix_layout(
    name: str, param: torch.nn.Parameter, optim_spec: OptimSpec
) -> dict | None:
    if optim_spec.slice_specs:
        return None
    if optim_spec.matrix_axes is None:
        if optim_spec.batch_axes:
            raise ValueError(
                f"OptimSpec.batch_axes for {name!r} require matrix_axes"
            )
        return None
    if optim_spec.route != "muon":
        raise ValueError(
            f"OptimSpec.matrix_axes for {name!r} are only valid for route='muon'"
        )
    ndim = int(param.ndim)
    matrix_axes = tuple(
        _normalize_axis(axis, ndim, label="matrix_axes")
        for axis in optim_spec.matrix_axes
    )
    if len(matrix_axes) != 2 or len(set(matrix_axes)) != 2:
        raise ValueError(
            f"OptimSpec.matrix_axes for {name!r} must contain exactly two unique axes"
        )
    batch_axes = tuple(
        _normalize_axis(axis, ndim, label="batch_axes")
        for axis in optim_spec.batch_axes
    )
    if len(set(batch_axes)) != len(batch_axes):
        raise ValueError(f"OptimSpec.batch_axes for {name!r} must be unique")
    if set(matrix_axes) & set(batch_axes):
        raise ValueError(
            f"OptimSpec.matrix_axes and batch_axes for {name!r} must not overlap"
        )
    covered_axes = set(matrix_axes) | set(batch_axes)
    missing_axes = tuple(axis for axis in range(ndim) if axis not in covered_axes)
    non_singleton_missing = [axis for axis in missing_axes if int(param.shape[axis]) != 1]
    if non_singleton_missing:
        raise ValueError(
            f"OptimSpec for {name!r} leaves non-singleton axes "
            f"{tuple(non_singleton_missing)} outside matrix_axes/batch_axes"
        )
    permute = batch_axes + tuple(missing_axes) + matrix_axes
    inverse_permute = tuple(permute.index(axis) for axis in range(ndim))
    source_shape = tuple(int(dim) for dim in param.shape)
    permuted_shape = tuple(source_shape[axis] for axis in permute)
    batch = math.prod(permuted_shape[:-2]) if len(permuted_shape) > 2 else 1
    rows = int(permuted_shape[-2])
    cols = int(permuted_shape[-1])
    if rows <= 0 or cols <= 0 or batch <= 0:
        raise ValueError(f"OptimSpec for {name!r} produced an empty matrix view")
    return {
        "source_shape": source_shape,
        "permute": permute,
        "inverse_permute": inverse_permute,
        "permuted_shape": permuted_shape,
        "matrix_view_shape": (int(batch), rows, cols),
    }


def _tensor_to_matrix_layout_view(tensor: torch.Tensor, layout: dict) -> torch.Tensor:
    return tensor.permute(tuple(layout["permute"])).reshape(
        tuple(layout["matrix_view_shape"])
    )


def _matrix_layout_view_to_tensor(
    matrix_update: torch.Tensor, layout: dict
) -> torch.Tensor:
    return (
        matrix_update.reshape(tuple(layout["permuted_shape"]))
        .permute(tuple(layout["inverse_permute"]))
        .reshape(tuple(layout["source_shape"]))
    )


def _normalize_module_optim_spec(name: str, spec) -> OptimSpec | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        spec = OptimSpec(**spec)
    if not isinstance(spec, OptimSpec):
        raise RuntimeError(
            f"HybridMuon routing='module' requires OptimSpec for parameter {name!r}"
        )
    return spec


def _optim_spec_from_module(
    module: torch.nn.Module, local_name: str, param: torch.nn.Parameter
) -> OptimSpec | None:
    getter = getattr(module, "hybrid_muon_optim_spec", None)
    spec = getter(local_name, param) if callable(getter) else None
    if spec is None:
        specs = getattr(module, "hybrid_muon_optim_specs", None)
        if specs is not None:
            spec = specs.get(local_name)
    return _normalize_module_optim_spec(local_name, spec)


def _owner_module_type_name(
    name: str,
    param: torch.nn.Parameter,
    module_map: dict[str, torch.nn.Module] | None,
) -> str | None:
    if module_map is None:
        return None
    module_name, _, local_name = name.rpartition(".")
    if not module_name:
        return None
    module = module_map.get(module_name)
    if module is not None and getattr(module, local_name, None) is param:
        module_type = type(module)
        return f"{module_type.__module__}.{module_type.__qualname__}"
    return None


def _module_declared_optim_spec(
    name: str,
    param: torch.nn.Parameter,
    module_map: dict[str, torch.nn.Module] | None,
) -> OptimSpec | None:
    if module_map is None:
        raise RuntimeError(
            "HybridMuon routing='module' requires a module_map with owning modules"
        )
    module_name, _, local_name = name.rpartition(".")
    if not module_name:
        raise RuntimeError(
            f"HybridMuon routing='module' cannot resolve owner module for {name!r}"
        )
    module = module_map.get(module_name)
    if module is None or getattr(module, local_name, None) is not param:
        raise RuntimeError(
            f"HybridMuon routing='module' requires owner module {module_name!r} "
            f"for parameter {name!r}"
        )

    spec = _optim_spec_from_module(module, local_name, param)
    if spec is not None:
        return spec

    parts = module_name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        ancestor_name = ".".join(parts[:index])
        ancestor = module_map.get(ancestor_name)
        if ancestor is None:
            continue
        relative_name = name[len(ancestor_name) + 1 :]
        spec = _optim_spec_from_module(ancestor, relative_name, param)
        if spec is not None:
            return spec

    return None


def _is_equivariant_slice_candidate(name: str) -> bool:
    return any(token in name for token in _EQUIVARIANT_SLICE_TOKENS)


def _flat_e3nn_linear_matrix_specs(
    name: str,
    param: torch.nn.Parameter,
    module_map: dict[str, torch.nn.Module] | None,
) -> tuple[list[dict], str] | None:
    if module_map is None or param.ndim != 1 or not name.endswith(".weight"):
        return None
    module_name = name.rsplit(".", 1)[0]
    module = module_map.get(module_name)
    if module is None or getattr(module, "weight", None) is not param:
        return None
    instructions = getattr(module, "instructions", None)
    if instructions is None:
        return None
    specs: list[dict] = []
    offset = 0
    lower = name.lower()
    for instruction_index, instruction in enumerate(instructions):
        path_shape = tuple(int(dim) for dim in getattr(instruction, "path_shape", ()))
        path = (
            int(instruction_index),
            int(getattr(instruction, "i_in", -1)),
            int(getattr(instruction, "i_out", -1)),
        )
        numel = math.prod(path_shape) if path_shape else 0
        if numel <= 0:
            return None
        if len(path_shape) == 2:
            rows, cols = path_shape
            specs.append(
                {
                    "offset": int(offset),
                    "numel": int(numel),
                    "matrix_view_shape": (1, int(rows), int(cols)),
                    "path": path,
                }
            )
        elif len(path_shape) == 3 and ".skip_tp.weight" in lower:
            channels_in, num_species, channels_out = path_shape
            specs.append(
                {
                    "offset": int(offset),
                    "numel": int(numel),
                    "source_shape": (
                        int(channels_in),
                        int(num_species),
                        int(channels_out),
                    ),
                    "permute": (1, 0, 2),
                    "inverse_permute": (1, 0, 2),
                    "matrix_view_shape": (
                        int(num_species),
                        int(channels_in),
                        int(channels_out),
                    ),
                    "path": path,
                }
            )
        else:
            return None
        offset += numel
    if offset != param.numel() or not specs:
        return None
    reason = (
        "tace-e3nn-skip-tp-species-muon"
        if ".skip_tp.weight" in lower
        else "tace-e3nn-linear-muon"
    )
    return specs, reason


def _flat_spec_to_matrix_view(flat_update: torch.Tensor, spec: dict) -> torch.Tensor:
    chunk = flat_update[spec["offset"] : spec["offset"] + spec["numel"]]
    if "source_shape" not in spec:
        return chunk.reshape(tuple(int(dim) for dim in spec["matrix_view_shape"]))
    return (
        chunk.reshape(tuple(int(dim) for dim in spec["source_shape"]))
        .permute(tuple(int(dim) for dim in spec["permute"]))
        .reshape(tuple(int(dim) for dim in spec["matrix_view_shape"]))
    )


def _matrix_view_to_flat_spec(matrix_update: torch.Tensor, spec: dict) -> torch.Tensor:
    matrix_view_shape = tuple(int(dim) for dim in spec["matrix_view_shape"])
    matrix_update = matrix_update.reshape(matrix_view_shape)
    if "source_shape" not in spec:
        return matrix_update.reshape(-1)
    return (
        matrix_update.permute(tuple(int(dim) for dim in spec["inverse_permute"]))
        .reshape(tuple(int(dim) for dim in spec["source_shape"]))
        .reshape(-1)
    )


def _muon_lr_scale(
    rows: int,
    cols: int,
    *,
    mode: str,
    match_rms_coeff: float,
) -> float:
    if mode == "original":
        return math.sqrt(max(1.0, rows / max(cols, 1)))
    if mode == "match_rms":
        return float(match_rms_coeff) * math.sqrt(max(rows, cols))
    if mode == "none":
        return 1.0
    raise ValueError(f"hybrid_muon_lr_scale_mode must be one of {sorted(_MUON_LR_SCALE_MODES)}")


def _canonical_wide_matrix_view(matrix_update: torch.Tensor) -> tuple[torch.Tensor, bool]:
    transposed = matrix_update.shape[-2] > matrix_update.shape[-1]
    if transposed:
        return matrix_update.transpose(-2, -1), True
    return matrix_update, False


def _spec_summary_shape(specs: list[dict]) -> tuple[int, tuple[int, int] | None]:
    matrix_shapes = {tuple(spec["matrix_view_shape"][-2:]) for spec in specs}
    matrix_batch = sum(int(spec["matrix_view_shape"][0]) for spec in specs)
    matrix_shape = matrix_shapes.pop() if len(matrix_shapes) == 1 else None
    return matrix_batch, matrix_shape


def _optim_spec_matrix_gate_reason(
    param: torch.nn.Parameter,
    optim_spec: OptimSpec,
    *,
    flat_specs: list[dict] | None,
    matrix_layout: dict | None,
    muon_mode: str,
) -> str | None:
    if optim_spec.route != "muon":
        return None
    matrix_shapes: list[tuple[int, int]] = []
    if flat_specs is not None:
        matrix_shapes.extend(
            tuple(int(dim) for dim in spec["matrix_view_shape"][-2:])
            for spec in flat_specs
        )
    elif matrix_layout is not None:
        matrix_shapes.append(
            tuple(int(dim) for dim in matrix_layout["matrix_view_shape"][-2:])
        )
    else:
        matrix_view = _matrix_view_shape(tuple(int(dim) for dim in param.shape), muon_mode)
        if matrix_view is not None:
            matrix_shapes.append(tuple(int(dim) for dim in matrix_view[-2:]))
    for rows, cols in matrix_shapes:
        short_side = min(rows, cols)
        long_side = max(rows, cols)
        if short_side < int(optim_spec.min_matrix_dim):
            return "module-matrix-dim-gate"
        if optim_spec.max_aspect_ratio is not None:
            aspect_ratio = long_side / max(short_side, 1)
            if aspect_ratio > float(optim_spec.max_aspect_ratio):
                return "module-matrix-aspect-gate"
    return None


def _route_parameter(
    name: str,
    param: torch.nn.Parameter,
    *,
    muon_mode: str = "2d",
    routing: str = "mace",
) -> tuple[str, str]:
    if muon_mode not in _MUON_MODES:
        raise ValueError(f"hybrid_muon_mode must be one of {sorted(_MUON_MODES)}")
    if routing not in _ROUTINGS:
        raise ValueError(f"hybrid_muon_routing must be one of {sorted(_ROUTINGS)}")
    lower = name.lower()
    if not param.requires_grad:
        return "frozen", "requires_grad=False"
    if "readout" in lower or "readouts" in lower:
        return "adam", "sensitive-name"
    if routing == "tace" and any(
        token in lower for token in _MACE_HARD_ADAM_NAME_TOKENS
    ):
        return "adam", "mace-sensitive-name"
    if param.ndim < 2:
        return "adam", "rank<2"
    effective_shape = _effective_shape(param)
    if len(effective_shape) < 2:
        return "adam", "effective-rank<2"
    if routing == "tace":
        if (
            len(effective_shape) == 2
            and ".conv_tp_weights." in lower
            and lower.endswith(".weight")
        ):
            return "muon", "radial-tp-weight-mlp"
        if len(effective_shape) == 2 and any(
            token in lower for token in _MUON_NAME_TOKENS
        ):
            return "muon", "safe-dense-name"
        return "adamw", "tace-unknown-adamw"
    if (
        len(effective_shape) == 2
        and ".conv_tp_weights." in lower
        and lower.endswith(".weight")
    ):
        return "muon", "radial-tp-weight-mlp"
    if any(token in lower for token in _ADAM_NAME_TOKENS):
        if not any(token in lower for token in _MUON_NAME_TOKENS):
            return "adam", "sensitive-name"
    if len(effective_shape) == 2 and any(token in lower for token in _MUON_NAME_TOKENS):
        return "muon", "safe-dense-name"
    return "adam", "ambiguous"


def build_hybrid_muon_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    muon_weight_decay: float,
    muon_lr_factor: float = 0.1,
    beta: float = 0.9,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    adam_variant: str = "adamw",
    muon_lr_scale_mode: str = "original",
    muon_match_rms_coeff: float = 0.18,
    eps: float = 1.0e-8,
    amsgrad: bool = False,
    muon_mode: str = "2d",
    routing: str = "mace",
    module_map: dict[str, torch.nn.Module] | None = None,
    tace_module_include: str | Iterable[str] | None = "*",
    tace_module_lr_scale: float = 1.0,
    magma_lite: bool = False,
    magma_initial_score: float = 0.5,
    magma_warmup_steps: int = 0,
    magma_bypass_first_step: bool = False,
    adam_param_options_by_id: dict[int, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    if muon_lr_factor <= 0.0:
        raise ValueError("muon_lr_factor must be positive")
    if muon_mode not in _MUON_MODES:
        raise ValueError(f"hybrid_muon_mode must be one of {sorted(_MUON_MODES)}")
    if routing not in _ROUTINGS:
        raise ValueError(f"hybrid_muon_routing must be one of {sorted(_ROUTINGS)}")
    if adam_variant not in _ADAM_VARIANTS:
        raise ValueError(
            f"hybrid_muon_adam_variant must be one of {sorted(_ADAM_VARIANTS)}"
        )
    if muon_lr_scale_mode not in _MUON_LR_SCALE_MODES:
        raise ValueError(
            f"hybrid_muon_lr_scale_mode must be one of {sorted(_MUON_LR_SCALE_MODES)}"
        )
    if muon_match_rms_coeff <= 0.0:
        raise ValueError("muon_match_rms_coeff must be positive")
    if tace_module_lr_scale <= 0.0:
        raise ValueError("tace_module_lr_scale must be positive")
    if not 0.0 <= magma_initial_score <= 1.0:
        raise ValueError("magma_initial_score must be in [0, 1]")
    if magma_warmup_steps < 0:
        raise ValueError("magma_warmup_steps must be non-negative")
    tace_module_include_patterns = _normalize_tace_module_include(tace_module_include)

    muon_params: list[torch.nn.Parameter] = []
    muon_param_names: list[str] = []
    muon_param_lr_scales: dict[str, float] = {}
    muon_param_weight_decays: dict[str, float] = {}
    muon_param_matrix_layouts: dict[str, dict] = {}
    muon_param_route_reasons: dict[str, str] = {}
    muon_param_module_types: dict[str, str] = {}
    muon_param_optim_spec_versions: dict[str, int] = {}
    muon_param_matrix_structures: dict[str, str] = {}
    muon_param_optim_spec_contracts: dict[str, dict] = {}
    adam_group_buckets: dict[tuple, dict] = {}
    muon_matrix_specs: dict[str, list[dict]] = {}
    summary: list[dict] = []
    seen_names: set[str] = set()
    seen_param_ids: dict[int, str] = {}
    for name, param in named_parameters:
        if name in seen_names:
            raise ValueError(f"Duplicate parameter name in HybridMuon routing: {name!r}")
        seen_names.add(name)
        if param.requires_grad:
            param_id = id(param)
            previous_name = seen_param_ids.get(param_id)
            if previous_name is not None:
                raise ValueError(
                    f"Trainable parameter {name!r} appears more than once in "
                    f"HybridMuon routing; first seen as {previous_name!r}"
                )
            seen_param_ids[param_id] = name
        flat_specs = None
        flat_reason = None
        adam_variant_override = None
        optim_spec = None
        matrix_layout = None
        effective_muon_lr_scale: float | None = None
        lower_name = name.lower()
        if routing == "module":
            if not param.requires_grad:
                route, reason = "frozen", "requires_grad=False"
            else:
                optim_spec = _module_declared_optim_spec(name, param, module_map)
                if optim_spec is None:
                    default_route, default_reason = _route_parameter(
                        name, param, muon_mode=muon_mode, routing="mace"
                    )
                    if default_route == "muon" and default_reason == "radial-tp-weight-mlp":
                        route, reason = "muon", f"module-default-{default_reason}"
                    else:
                        route, reason = "adamw", "module-default-adamw"
                        adam_variant_override = "adamw"
                else:
                    route, reason = optim_spec.route, "module-declared"
                    _validate_optim_spec_muon_contract(name, param, optim_spec)
                    flat_specs = _normalize_optim_spec_slice_specs(name, param, optim_spec)
                    matrix_layout = _optim_spec_matrix_layout(name, param, optim_spec)
                    if flat_specs is not None:
                        muon_matrix_specs[name] = flat_specs
                    if optim_spec.route in _ADAM_VARIANTS:
                        adam_variant_override = optim_spec.route
        else:
            if routing == "tace" and not any(
                token in lower_name for token in _MACE_HARD_ADAM_NAME_TOKENS
            ):
                if module_map is not None:
                    optim_spec = _module_declared_optim_spec(name, param, module_map)
                if optim_spec is not None:
                    if not _matches_tace_module_include(
                        name, tace_module_include_patterns
                    ):
                        route, reason = "adamw", "module-spec-filtered"
                        adam_variant_override = "adamw"
                    else:
                        route, reason = optim_spec.route, "module-declared"
                        _validate_optim_spec_muon_contract(name, param, optim_spec)
                        if optim_spec.route == "muon":
                            effective_muon_lr_scale = (
                                float(optim_spec.lr_scale) * float(tace_module_lr_scale)
                            )
                        flat_specs = _normalize_optim_spec_slice_specs(
                            name, param, optim_spec
                        )
                        matrix_layout = _optim_spec_matrix_layout(
                            name, param, optim_spec
                        )
                        if flat_specs is not None:
                            muon_matrix_specs[name] = flat_specs
                        if optim_spec.route in _ADAM_VARIANTS:
                            adam_variant_override = optim_spec.route
                else:
                    flat_spec_result = _flat_e3nn_linear_matrix_specs(
                        name, param, module_map
                    )
                    if flat_spec_result is not None:
                        flat_specs, flat_reason = flat_spec_result
            if optim_spec is None:
                if flat_specs is not None:
                    route, reason = "muon", flat_reason
                    muon_matrix_specs[name] = flat_specs
                else:
                    route, reason = _route_parameter(
                        name, param, muon_mode=muon_mode, routing=routing
                    )
        if route == "muon" and optim_spec is not None:
            gate_reason = _optim_spec_matrix_gate_reason(
                param,
                optim_spec,
                flat_specs=flat_specs,
                matrix_layout=matrix_layout,
                muon_mode=muon_mode,
            )
            if gate_reason is not None:
                route, reason = "adamw", gate_reason
                adam_variant_override = "adamw"
                effective_muon_lr_scale = None
                flat_specs = None
                matrix_layout = None
                muon_matrix_specs.pop(name, None)
        if route == "frozen":
            continue
        module_type_name = _owner_module_type_name(name, param, module_map)
        if route == "muon" and _is_sharded_or_distributed_parameter(param):
            raise RuntimeError(
                f"HybridMuon cannot safely route sharded/DTensor parameter {name!r} "
                "to Muon. Route it to AdamW or use a full-matrix/distributed Gram "
                "Muon implementation."
            )
        if route == "muon":
            muon_params.append(param)
            muon_param_names.append(name)
            muon_param_route_reasons[name] = str(reason)
            if module_type_name is not None:
                muon_param_module_types[name] = module_type_name
            if optim_spec is not None:
                muon_param_optim_spec_versions[name] = int(optim_spec.spec_version)
                muon_param_matrix_structures[name] = str(optim_spec.matrix_structure)
                muon_param_optim_spec_contracts[name] = (
                    _optim_spec_manifest_contract(optim_spec, param)
                )
                if effective_muon_lr_scale is None:
                    effective_muon_lr_scale = float(optim_spec.lr_scale)
                if effective_muon_lr_scale != 1.0:
                    muon_param_lr_scales[name] = float(effective_muon_lr_scale)
                if optim_spec.weight_decay is not None:
                    muon_param_weight_decays[name] = float(optim_spec.weight_decay)
                if matrix_layout is not None:
                    muon_param_matrix_layouts[name] = matrix_layout
        else:
            adam_options = (adam_param_options_by_id or {}).get(id(param), {})
            group_lr = adam_options.get("lr", lr)
            group_weight_decay = adam_options.get("weight_decay", weight_decay)
            if optim_spec is not None and optim_spec.weight_decay is not None:
                group_weight_decay = float(optim_spec.weight_decay)
            group_betas = tuple(adam_options.get("betas", adam_betas))
            group_eps = adam_options.get("eps", eps)
            group_amsgrad = bool(adam_options.get("amsgrad", amsgrad))
            group_adam_variant = adam_variant_override or adam_options.get(
                "adam_variant", adam_variant
            )
            if group_adam_variant not in _ADAM_VARIANTS:
                raise ValueError(
                    f"hybrid_muon_adam_variant must be one of {sorted(_ADAM_VARIANTS)}"
                )
            key = (
                float(group_lr),
                float(group_weight_decay),
                group_betas,
                float(group_eps),
                group_amsgrad,
                group_adam_variant,
            )
            bucket = adam_group_buckets.setdefault(
                key,
                {
                    "params": [],
                    "param_names": [],
                    "route": "adam",
                    "adam_variant": group_adam_variant,
                    "lr": group_lr,
                    "weight_decay": group_weight_decay,
                    "betas": group_betas,
                    "eps": group_eps,
                    "amsgrad": group_amsgrad,
                    "param_route_reasons": {},
                    "param_module_types": {},
                    "param_optim_spec_versions": {},
                    "param_matrix_structures": {},
                    "param_optim_spec_contracts": {},
                },
            )
            bucket["params"].append(param)
            bucket["param_names"].append(name)
            bucket["param_route_reasons"][name] = str(reason)
            if module_type_name is not None:
                bucket["param_module_types"][name] = module_type_name
            if optim_spec is not None:
                bucket["param_optim_spec_versions"][name] = int(optim_spec.spec_version)
                bucket["param_matrix_structures"][name] = str(optim_spec.matrix_structure)
                bucket["param_optim_spec_contracts"][name] = (
                    _optim_spec_manifest_contract(optim_spec, param)
                )
        matrix_view = _matrix_view_shape(tuple(int(dim) for dim in param.shape), muon_mode)
        matrix_batch = matrix_view[0] if route == "muon" and matrix_view else None
        matrix_shape = matrix_view[-2:] if route == "muon" and matrix_view else None
        if route == "muon" and matrix_layout is not None:
            matrix_batch = matrix_layout["matrix_view_shape"][0]
            matrix_shape = matrix_layout["matrix_view_shape"][-2:]
        if route == "muon" and flat_specs is not None:
            matrix_batch, matrix_shape = _spec_summary_shape(flat_specs)
        summary.append(
            RouteRecord(
                name=name,
                shape=tuple(int(dim) for dim in param.shape),
                numel=param.numel(),
                route=route,
                reason=reason,
                muon_mode=muon_mode if route == "muon" else None,
                matrix_shape=matrix_shape,
                matrix_batch=matrix_batch,
                lr_scale=(
                    float(effective_muon_lr_scale)
                    if route == "muon"
                    and effective_muon_lr_scale is not None
                    and effective_muon_lr_scale != 1.0
                    else None
                ),
            ).as_summary()
        )

    groups: list[dict] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "route": "muon",
                "lr": lr * muon_lr_factor,
                "hybrid_muon_base_lr": lr,
                "hybrid_muon_lr_factor": muon_lr_factor,
                "weight_decay": muon_weight_decay,
                "beta": beta,
                "muon_mode": muon_mode,
                "routing": routing,
                "matrix_specs": muon_matrix_specs,
                "param_names": muon_param_names,
                "param_lr_scales": muon_param_lr_scales,
                "param_weight_decays": muon_param_weight_decays,
                "param_matrix_layouts": muon_param_matrix_layouts,
                "param_route_reasons": muon_param_route_reasons,
                "param_module_types": muon_param_module_types,
                "param_optim_spec_versions": muon_param_optim_spec_versions,
                "param_matrix_structures": muon_param_matrix_structures,
                "param_optim_spec_contracts": muon_param_optim_spec_contracts,
                "muon_lr_scale_mode": muon_lr_scale_mode,
                "muon_match_rms_coeff": float(muon_match_rms_coeff),
                "magma_lite": bool(magma_lite),
                "magma_initial_score": float(magma_initial_score),
                "magma_warmup_steps": int(magma_warmup_steps),
                "magma_bypass_first_step": bool(magma_bypass_first_step),
            }
        )
    groups.extend(adam_group_buckets.values())
    return groups, summary


def summarize_hybrid_muon_routes(summary: list[dict]) -> str:
    muon = [item for item in summary if item["route"] == "muon"]
    adam = [item for item in summary if item["route"] in {"adam", "adamw"}]
    lines = [
        "HybridMuon parameter routing",
        f"Muon tensors: {len(muon)} ({sum(item['numel'] for item in muon)} parameters)",
        f"Adam tensors: {len(adam)} ({sum(item['numel'] for item in adam)} parameters)",
    ]
    if any(str(item.get("reason", "")).startswith("tace-") for item in summary):
        lines.append(
            "Warning: routing='tace' is deprecated compatibility routing; "
            "prefer module-declared OptimSpec routing for TECE/DPA4-style blocks."
        )
    for title, items in (("Muon", muon), ("Adam", adam)):
        for item in items:
            suffix = ""
            if item.get("matrix_shape") is not None:
                suffix = (
                    f" mode={item.get('muon_mode')}"
                    f" matrix_batch={item.get('matrix_batch')}"
                    f" matrix_shape={item.get('matrix_shape')}"
                )
            if item.get("lr_scale") is not None:
                suffix += f" lr_scale={item.get('lr_scale')}"
            lines.append(
                f"{title}: {item['name']} shape={item['shape']} reason={item['reason']}{suffix}"
            )
    return "\n".join(lines)



def _magma_lite_scale(
    state: dict,
    score_key: str,
    grad_matrix: torch.Tensor,
    momentum_matrix: torch.Tensor,
    *,
    initial_score: float = 0.5,
    force_scale_one: bool = False,
    bypass_first_step: bool = False,
) -> torch.Tensor:
    batch = int(grad_matrix.shape[0])
    grad_view = grad_matrix.reshape(batch, -1).to(dtype=torch.float32)
    momentum_view = momentum_matrix.reshape(batch, -1).to(dtype=torch.float32)
    score = state.get(score_key)
    is_new_score = (
        score is None
        or not torch.is_tensor(score)
        or score.ndim != 1
        or score.numel() != batch
        or score.device != grad_matrix.device
    )
    if is_new_score:
        score = torch.full(
            (batch,), float(initial_score), dtype=torch.float32, device=grad_matrix.device
        )
    elif score.dtype != torch.float32:
        score = score.to(dtype=torch.float32, device=grad_matrix.device)
    dot = (momentum_view * grad_view).sum(dim=1)
    denom = (momentum_view.norm(dim=1) * grad_view.norm(dim=1)).clamp_min(_MAGMA_EPS)
    cosine = (dot / denom).clamp(min=-1.0, max=1.0)
    raw = torch.sigmoid(cosine / _MAGMA_TAU)
    raw = (raw - _MAGMA_SIGMOID_MIN) / (_MAGMA_SIGMOID_MAX - _MAGMA_SIGMOID_MIN)
    raw = raw.clamp(min=0.0, max=1.0)
    score.mul_(_MAGMA_EMA_DECAY).add_(raw, alpha=1.0 - _MAGMA_EMA_DECAY)
    state[score_key] = score
    scale = _MAGMA_MIN_SCALE + (1.0 - _MAGMA_MIN_SCALE) * score
    if force_scale_one or (bypass_first_step and is_new_score):
        return torch.ones_like(scale)
    return scale

def _orthogonalize_newton_schulz(update: torch.Tensor, steps: int | None = None) -> torch.Tensor:
    original_dtype = update.dtype
    x = update.float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / x.norm().clamp_min(_NS_EPS)
    schedule = (
        (_NS_COEFF_FAST, _NS_STEPS_FAST if steps is None else steps),
        (_NS_COEFF_POLISH, _NS_STEPS_POLISH if steps is None else 0),
    )
    for (a, b, c), count in schedule:
        for _ in range(count):
            gram = x @ x.T
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def _orthogonalize_newton_schulz_batched(
    updates: torch.Tensor, steps: int | None = None
) -> torch.Tensor:
    original_dtype = updates.dtype
    x = updates.float()
    if x.ndim != 3:
        raise ValueError(
            "batched Newton-Schulz input must have shape (batch, rows, cols)"
        )
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    norm = x.flatten(start_dim=1).norm(dim=1).clamp_min(_NS_EPS)
    x = x / norm.view(-1, 1, 1)
    schedule = (
        (_NS_COEFF_FAST, _NS_STEPS_FAST if steps is None else steps),
        (_NS_COEFF_POLISH, _NS_STEPS_POLISH if steps is None else 0),
    )
    for (a, b, c), count in schedule:
        for _ in range(count):
            gram = x @ x.transpose(-2, -1)
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.to(dtype=original_dtype)


_RUNTIME_PARAM_GROUP_METADATA_KEYS = (
    "matrix_specs",
    "param_lr_scales",
    "param_weight_decays",
    "param_matrix_layouts",
    "param_optim_spec_versions",
    "param_matrix_structures",
    "param_optim_spec_contracts",
)
_ROUTE_MANIFEST_STATE_KEYS = (
    "hybrid_muon_route_manifest",
    "hybrid_muon_route_manifest_hash",
)


def _jsonify_route_value(value):
    if isinstance(value, dict):
        return {str(key): _jsonify_route_value(val) for key, val in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonify_route_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _route_manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_allows_stage_two_route_switch(
    saved_manifest: dict | None, current_manifest: dict, state_dict: dict
) -> bool:
    if not isinstance(saved_manifest, dict):
        return False
    if not any(
        bool(group.get("hybrid_muon_stage_two_route_applied", False))
        for group in state_dict.get("param_groups", [])
    ):
        return False
    if saved_manifest.get("spec_version") != current_manifest.get("spec_version"):
        return False
    saved_parameters = saved_manifest.get("parameters")
    current_parameters = current_manifest.get("parameters")
    if not isinstance(saved_parameters, dict) or not isinstance(current_parameters, dict):
        return False
    if set(saved_parameters) != set(current_parameters):
        return False
    stable_keys = {
        "shape",
        "reason",
        "module_type",
        "optim_spec_version",
        "structure",
        "optim_spec_contract",
    }
    for name, saved_item in saved_parameters.items():
        current_item = current_parameters[name]
        if not isinstance(saved_item, dict) or not isinstance(current_item, dict):
            return False
        for key in stable_keys:
            if saved_item.get(key) != current_item.get(key):
                return False
        if saved_item.get("route") == current_item.get("route"):
            if saved_item.get("group_route") != current_item.get("group_route"):
                return False
            if saved_item.get("matrix_views") != current_item.get("matrix_views"):
                return False
            continue
        if not (
            current_item.get("route") == "muon"
            and current_item.get("group_route") == "muon"
            and saved_item.get("route") in {"adam", "adamw"}
            and saved_item.get("group_route") == "adam"
        ):
            return False
        if saved_item.get("matrix_views") != current_item.get("matrix_views"):
            return False
        if saved_item.get("muon_mode") != current_item.get("muon_mode"):
            return False
    return True


def _hybrid_muon_route_manifest_from_groups(param_groups: Iterable[dict]) -> dict:
    parameters: dict[str, dict] = {}
    for group_index, group in enumerate(param_groups):
        route = group.get("route", "adam")
        param_names = group.get("param_names")
        if not isinstance(param_names, (list, tuple)):
            param_names = [
                f"<unnamed group {group_index} parameter {param_index}>"
                for param_index, _ in enumerate(group.get("params", []))
            ]
        matrix_specs = group.get("matrix_specs", {})
        if not isinstance(matrix_specs, dict):
            matrix_specs = {}
        matrix_layouts = group.get("param_matrix_layouts", {})
        if not isinstance(matrix_layouts, dict):
            matrix_layouts = {}
        route_reasons = group.get("param_route_reasons", {})
        if not isinstance(route_reasons, dict):
            route_reasons = {}
        module_types = group.get("param_module_types", {})
        if not isinstance(module_types, dict):
            module_types = {}
        optim_spec_versions = group.get("param_optim_spec_versions", {})
        if not isinstance(optim_spec_versions, dict):
            optim_spec_versions = {}
        matrix_structures = group.get("param_matrix_structures", {})
        if not isinstance(matrix_structures, dict):
            matrix_structures = {}
        optim_spec_contracts = group.get("param_optim_spec_contracts", {})
        if not isinstance(optim_spec_contracts, dict):
            optim_spec_contracts = {}
        muon_mode = str(group.get("muon_mode", "2d"))
        adam_variant = str(group.get("adam_variant", "adamw"))
        for param_index, param in enumerate(group.get("params", [])):
            name = str(param_names[param_index])
            if route == "muon":
                parameter_route = "muon"
            else:
                parameter_route = adam_variant if adam_variant in _ADAM_VARIANTS else "adam"
            item = {
                "shape": [int(dim) for dim in param.shape],
                "route": parameter_route,
                "group_route": str(route),
                "matrix_views": [],
                "structure": str(matrix_structures.get(name, "real")),
            }
            if name in route_reasons:
                item["reason"] = str(route_reasons[name])
            if name in module_types:
                item["module_type"] = str(module_types[name])
            if name in optim_spec_versions:
                item["optim_spec_version"] = int(optim_spec_versions[name])
            if name in optim_spec_contracts:
                item["optim_spec_contract"] = optim_spec_contracts[name]
            records_muon_matrix_views = parameter_route == "muon" or bool(
                group.get("hybrid_muon_stage_two_route_applied", False)
            )
            if records_muon_matrix_views:
                if name in matrix_specs:
                    item["matrix_views"] = []
                    for spec in matrix_specs[name]:
                        view_kind = str(spec.get("source", "flat_spec"))
                        if view_kind not in {"flat_spec", "module_slice_spec"}:
                            view_kind = "flat_spec"
                        view = {
                            "kind": view_kind,
                            "offset": int(spec["offset"]),
                            "numel": int(spec["numel"]),
                            "shape": [
                                int(dim) for dim in spec["matrix_view_shape"][-2:]
                            ],
                        }
                        if "path" in spec:
                            view["path"] = [int(item) for item in spec["path"]]
                        item["matrix_views"].append(view)
                elif name in matrix_layouts:
                    layout = matrix_layouts[name]
                    item["matrix_views"] = [
                        {
                            "kind": "layout",
                            "shape": [
                                int(dim) for dim in layout["matrix_view_shape"][-2:]
                            ],
                        }
                    ]
                else:
                    matrix_view = _matrix_view_shape(
                        tuple(int(dim) for dim in param.shape), muon_mode
                    )
                    if matrix_view is not None:
                        item["matrix_views"] = [
                            {
                                "kind": "view",
                                "shape": [int(dim) for dim in matrix_view[-2:]],
                            }
                        ]
                if item["matrix_views"]:
                    item["muon_mode"] = muon_mode
                    if parameter_route != "muon":
                        item["stage_two_from_route"] = "muon"
            parameters[name] = item
    return {
        "spec_version": 1,
        "parameters": _jsonify_route_value(parameters),
    }


class HybridMuon(Optimizer):
    def __init__(
        self,
        params: Iterable[dict],
        *,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        beta: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta": beta,
            "betas": betas,
            "eps": eps,
            "route": "adam",
        }
        super().__init__(params, defaults)

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict.get("param_groups", []):
            for key in _RUNTIME_PARAM_GROUP_METADATA_KEYS:
                group.pop(key, None)
        manifest = _hybrid_muon_route_manifest_from_groups(self.param_groups)
        state_dict["hybrid_muon_route_manifest"] = manifest
        state_dict["hybrid_muon_route_manifest_hash"] = _route_manifest_hash(manifest)
        return state_dict

    def load_state_dict(self, state_dict):
        saved_route_hash = state_dict.get("hybrid_muon_route_manifest_hash")
        if saved_route_hash is not None:
            current_manifest = _hybrid_muon_route_manifest_from_groups(self.param_groups)
            current_hash = _route_manifest_hash(current_manifest)
            if (
                str(saved_route_hash) != current_hash
                and not _manifest_allows_stage_two_route_switch(
                    state_dict.get("hybrid_muon_route_manifest"),
                    current_manifest,
                    state_dict,
                )
            ):
                raise ValueError(
                    "HybridMuon route manifest hash mismatch; checkpoint routes "
                    "do not match the current optimizer routing"
                )
        runtime_metadata = [
            {key: group.get(key) for key in _RUNTIME_PARAM_GROUP_METADATA_KEYS}
            for group in self.param_groups
        ]
        sanitized_state_dict = dict(state_dict)
        for key in _ROUTE_MANIFEST_STATE_KEYS:
            sanitized_state_dict.pop(key, None)
        sanitized_groups = []
        for group in state_dict.get("param_groups", []):
            sanitized_group = dict(group)
            for key in _RUNTIME_PARAM_GROUP_METADATA_KEYS:
                sanitized_group.pop(key, None)
            sanitized_groups.append(sanitized_group)
        sanitized_state_dict["param_groups"] = sanitized_groups
        result = super().load_state_dict(sanitized_state_dict)
        for group, metadata in zip(
            self.param_groups, runtime_metadata, strict=False
        ):
            for key, value in metadata.items():
                if value is not None:
                    group[key] = value
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            route = group.get("route", "adam")
            if route == "muon":
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group: dict) -> None:
        lr = group["lr"]
        beta = group.get("beta", 0.9)
        weight_decay = group.get("weight_decay", 0.0)
        muon_mode = group.get("muon_mode", "2d")
        matrix_specs = group.get("matrix_specs", {})
        muon_lr_scale_mode = group.get("muon_lr_scale_mode", "original")
        muon_match_rms_coeff = float(group.get("muon_match_rms_coeff", 0.18))
        if muon_lr_scale_mode not in _MUON_LR_SCALE_MODES:
            raise ValueError(
                f"hybrid_muon_lr_scale_mode must be one of {sorted(_MUON_LR_SCALE_MODES)}"
            )
        if muon_match_rms_coeff <= 0.0:
            raise ValueError("muon_match_rms_coeff must be positive")
        param_names = group.get("param_names")
        if not isinstance(param_names, (list, tuple)) or len(param_names) != len(
            group["params"]
        ):
            param_names = [
                f"<unnamed parameter {index}>"
                for index, _ in enumerate(group["params"])
            ]
        param_lr_scales = group.get("param_lr_scales", {})
        if not isinstance(param_lr_scales, dict):
            param_lr_scales = {}
        param_weight_decays = group.get("param_weight_decays", {})
        if not isinstance(param_weight_decays, dict):
            param_weight_decays = {}
        param_matrix_layouts = group.get("param_matrix_layouts", {})
        if not isinstance(param_matrix_layouts, dict):
            param_matrix_layouts = {}
        magma_lite = bool(group.get("magma_lite", False))
        magma_initial_score = float(group.get("magma_initial_score", 0.5))
        magma_warmup_steps = int(group.get("magma_warmup_steps", 0))
        magma_bypass_first_step = bool(group.get("magma_bypass_first_step", False))
        if not 0.0 <= magma_initial_score <= 1.0:
            raise ValueError("magma_initial_score must be in [0, 1]")
        if magma_warmup_steps < 0:
            raise ValueError("magma_warmup_steps must be non-negative")
        updates_by_shape = {}
        for param_index, param in enumerate(group["params"]):
            param_name = str(param_names[param_index])
            if param.grad is None:
                continue
            grad = param.grad
            param_weight_decay = float(param_weight_decays.get(param_name, weight_decay))
            param_lr_scale = float(param_lr_scales.get(param_name, 1.0))
            if param_weight_decay:
                param.mul_(1.0 - lr * param_weight_decay)
            state = self.state[param]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(param)
            magma_step = int(state.get("muon_step", 0))
            magma_force_scale_one = magma_lite and magma_step < magma_warmup_steps
            momentum = state["momentum"]
            momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
            update = momentum.mul(beta).add(grad, alpha=1.0 - beta)
            specs = matrix_specs.get(param_name)
            matrix_layout = param_matrix_layouts.get(param_name)
            if specs and matrix_layout is not None:
                raise RuntimeError(
                    f"Muon-routed parameter {param_name!r} has both MatrixSpec "
                    "and OptimSpec matrix_axes layout"
                )
            if specs:
                flat_update = update.reshape(-1)
                for spec in specs:
                    matrix_update = _flat_spec_to_matrix_view(flat_update, spec)
                    grad_matrix = _flat_spec_to_matrix_view(grad.reshape(-1), spec)
                    momentum_matrix = _flat_spec_to_matrix_view(momentum.reshape(-1), spec)
                    rows, cols = matrix_update.shape[-2:]
                    scale = param_lr_scale * _muon_lr_scale(
                        rows,
                        cols,
                        mode=muon_lr_scale_mode,
                        match_rms_coeff=muon_match_rms_coeff,
                    )
                    magma_scale = None
                    if magma_lite:
                        magma_scale = _magma_lite_scale(
                            state,
                            f"magma_score_{int(spec['offset'])}",
                            grad_matrix,
                            momentum_matrix,
                            initial_score=magma_initial_score,
                            force_scale_one=magma_force_scale_one,
                            bypass_first_step=magma_bypass_first_step,
                        )
                    canonical_update, transposed = _canonical_wide_matrix_view(
                        matrix_update
                    )
                    key = (
                        canonical_update.shape[-2],
                        matrix_update.device,
                        matrix_update.dtype,
                    )
                    updates_by_shape.setdefault(key, []).append(
                        (param, canonical_update, scale, spec, magma_scale, transposed)
                    )
                if magma_lite:
                    state["muon_step"] = magma_step + 1
                continue
            if matrix_layout is not None:
                matrix_update = _tensor_to_matrix_layout_view(update, matrix_layout)
                grad_matrix = _tensor_to_matrix_layout_view(grad, matrix_layout)
                momentum_matrix = _tensor_to_matrix_layout_view(momentum, matrix_layout)
                rows, cols = matrix_update.shape[-2:]
                scale = param_lr_scale * _muon_lr_scale(
                    rows,
                    cols,
                    mode=muon_lr_scale_mode,
                    match_rms_coeff=muon_match_rms_coeff,
                )
                magma_scale = None
                if magma_lite:
                    magma_scale = _magma_lite_scale(
                        state,
                        "magma_score",
                        grad_matrix,
                        momentum_matrix,
                        initial_score=magma_initial_score,
                        force_scale_one=magma_force_scale_one,
                        bypass_first_step=magma_bypass_first_step,
                    )
                canonical_update, transposed = _canonical_wide_matrix_view(matrix_update)
                key = (
                    canonical_update.shape[-2],
                    matrix_update.device,
                    matrix_update.dtype,
                )
                updates_by_shape.setdefault(key, []).append(
                    (param, canonical_update, scale, matrix_layout, magma_scale, transposed)
                )
                if magma_lite:
                    state["muon_step"] = magma_step + 1
                continue
            matrix_view_shape = _matrix_view_shape(
                tuple(int(dim) for dim in update.shape), muon_mode
            )
            if matrix_view_shape is None:
                raise RuntimeError(
                    f"Muon-routed parameter {param_name!r} has no valid MatrixSpec "
                    f"or matrix view for shape {tuple(int(dim) for dim in param.shape)}"
                )
            matrix_update = update.reshape(matrix_view_shape)
            grad_matrix = grad.reshape(matrix_view_shape)
            momentum_matrix = momentum.reshape(matrix_view_shape)
            rows, cols = matrix_update.shape[-2:]
            scale = param_lr_scale * _muon_lr_scale(
                rows,
                cols,
                mode=muon_lr_scale_mode,
                match_rms_coeff=muon_match_rms_coeff,
            )
            magma_scale = None
            if magma_lite:
                magma_scale = _magma_lite_scale(
                    state,
                    "magma_score",
                    grad_matrix,
                    momentum_matrix,
                    initial_score=magma_initial_score,
                    force_scale_one=magma_force_scale_one,
                    bypass_first_step=magma_bypass_first_step,
                )
            canonical_update, transposed = _canonical_wide_matrix_view(matrix_update)
            key = (
                canonical_update.shape[-2],
                matrix_update.device,
                matrix_update.dtype,
            )
            updates_by_shape.setdefault(key, []).append(
                (param, canonical_update, scale, None, magma_scale, transposed)
            )
            if magma_lite:
                state["muon_step"] = magma_step + 1

        flat_deltas: dict[torch.nn.Parameter, torch.Tensor] = {}
        for records in updates_by_shape.values():
            total_batch = sum(record[1].shape[0] for record in records)
            short_side = int(records[0][1].shape[-2])
            max_long_side = max(int(record[1].shape[-1]) for record in records)
            needs_padding = any(
                int(record[1].shape[-1]) != max_long_side for record in records
            )
            if total_batch == 1 and not needs_padding:
                param, matrix_update, scale, spec, magma_scale, transposed = records[0]
                ortho = _orthogonalize_newton_schulz(matrix_update[0])
                if magma_scale is not None:
                    ortho = ortho * magma_scale.reshape(()).to(
                        dtype=ortho.dtype, device=ortho.device
                    )
                if transposed:
                    ortho = ortho.transpose(-2, -1)
                if spec is None:
                    param.add_(ortho.reshape_as(param), alpha=-lr * scale)
                elif "offset" in spec:
                    delta = flat_deltas.setdefault(
                        param, torch.zeros_like(param).reshape(-1)
                    )
                    delta[spec["offset"] : spec["offset"] + spec["numel"]].add_(
                        _matrix_view_to_flat_spec(ortho, spec), alpha=scale
                    )
                else:
                    param.add_(
                        _matrix_layout_view_to_tensor(ortho, spec),
                        alpha=-lr * scale,
                    )
                continue

            stacked_updates = records[0][1].new_zeros(
                (total_batch, short_side, max_long_side)
            )
            offset = 0
            for _, matrix_update, _, _, _, _ in records:
                batch = matrix_update.shape[0]
                long_side = matrix_update.shape[-1]
                stacked_updates[offset : offset + batch, :, :long_side].copy_(
                    matrix_update
                )
                offset += batch
            orthogonalized = _orthogonalize_newton_schulz_batched(stacked_updates)
            offset = 0
            for param, matrix_update, scale, spec, magma_scale, transposed in records:
                batch = matrix_update.shape[0]
                long_side = matrix_update.shape[-1]
                ortho = orthogonalized[offset : offset + batch, :, :long_side]
                offset += batch
                if magma_scale is not None:
                    ortho = ortho * magma_scale.view(batch, 1, 1).to(
                        dtype=ortho.dtype, device=ortho.device
                    )
                if transposed:
                    ortho = ortho.transpose(-2, -1)
                if spec is None:
                    param.add_(ortho.reshape_as(param), alpha=-lr * scale)
                elif "offset" in spec:
                    delta = flat_deltas.setdefault(
                        param, torch.zeros_like(param).reshape(-1)
                    )
                    delta[spec["offset"] : spec["offset"] + spec["numel"]].add_(
                        _matrix_view_to_flat_spec(ortho, spec), alpha=scale
                    )
                else:
                    param.add_(
                        _matrix_layout_view_to_tensor(ortho, spec),
                        alpha=-lr * scale,
                    )
        for param, delta in flat_deltas.items():
            param.add_(delta.reshape_as(param), alpha=-lr)

    def _step_adam_group(self, group: dict) -> None:
        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps = []
        amsgrad = group.get("amsgrad", False)
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError(
                    "HybridMuon Adam route does not support sparse gradients"
                )
            state = self.state[param]
            if "step" not in state:
                state["step"] = torch.tensor(0.0)
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            if amsgrad and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, dtype=torch.float32
                )
            if not torch.is_tensor(state["step"]):
                state["step"] = torch.tensor(float(state["step"]))
            elif state["step"].dtype not in (torch.float32, torch.float64):
                state["step"] = state["step"].detach().to(
                    device="cpu", dtype=torch.float32
                )
            params.append(param)
            grads.append(param.grad)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            state_steps.append(state["step"])
        if not params:
            return
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        adam_variant = group.get("adam_variant", "adamw")
        if adam_variant not in _ADAM_VARIANTS:
            raise ValueError(
                f"hybrid_muon_adam_variant must be one of {sorted(_ADAM_VARIANTS)}"
            )
        optim_functional.adam(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            decoupled_weight_decay=(adam_variant == "adamw"),
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group.get("weight_decay", 0.0),
            eps=group.get("eps", 1.0e-8),
            maximize=False,
        )
