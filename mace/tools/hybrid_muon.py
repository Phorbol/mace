from __future__ import annotations

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
class RouteRecord:
    name: str
    shape: tuple[int, ...]
    numel: int
    route: str
    reason: str
    muon_mode: str | None = None
    matrix_shape: tuple[int, int] | None = None
    matrix_batch: int | None = None

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
        return item


_ADAM_NAME_TOKENS = (
    "bias",
    "norm",
    "scale",
    "shift",
    "atomic_energies",
    "atomic_energy",
    "embedding",
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
    "readout",
    "readouts",
    "mlp",
    "fitting",
)

_EQUIVARIANT_SLICE_TOKENS = (
    "symmetric_contractions",
    "contractions",
)

_MUON_MODES = {"2d", "slice"}
_ROUTINGS = {"mace", "tace"}
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
    for instruction in instructions:
        path_shape = tuple(int(dim) for dim in getattr(instruction, "path_shape", ()))
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


def _spec_summary_shape(specs: list[dict]) -> tuple[int, tuple[int, int] | None]:
    matrix_shapes = {tuple(spec["matrix_view_shape"][-2:]) for spec in specs}
    matrix_batch = sum(int(spec["matrix_view_shape"][0]) for spec in specs)
    matrix_shape = matrix_shapes.pop() if len(matrix_shapes) == 1 else None
    return matrix_batch, matrix_shape


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
        if _matrix_view_shape(tuple(int(dim) for dim in param.shape), muon_mode) is None:
            return "adam", "non-matrix-for-mode"
        return "muon", "tace-matrix-muon"
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
    eps: float = 1.0e-8,
    amsgrad: bool = False,
    muon_mode: str = "2d",
    routing: str = "mace",
    module_map: dict[str, torch.nn.Module] | None = None,
    magma_lite: bool = False,
    adam_param_options_by_id: dict[int, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    if muon_lr_factor <= 0.0:
        raise ValueError("muon_lr_factor must be positive")
    if muon_mode not in _MUON_MODES:
        raise ValueError(f"hybrid_muon_mode must be one of {sorted(_MUON_MODES)}")
    if routing not in _ROUTINGS:
        raise ValueError(f"hybrid_muon_routing must be one of {sorted(_ROUTINGS)}")

    muon_params: list[torch.nn.Parameter] = []
    adam_group_buckets: dict[tuple, dict] = {}
    muon_matrix_specs: dict[int, list[dict]] = {}
    summary: list[dict] = []
    for name, param in named_parameters:
        flat_specs = None
        flat_reason = None
        lower_name = name.lower()
        if routing == "tace" and not any(
            token in lower_name for token in _MACE_HARD_ADAM_NAME_TOKENS
        ):
            flat_spec_result = _flat_e3nn_linear_matrix_specs(name, param, module_map)
            if flat_spec_result is not None:
                flat_specs, flat_reason = flat_spec_result
        if flat_specs is not None:
            route, reason = "muon", flat_reason
            muon_matrix_specs[id(param)] = flat_specs
        else:
            route, reason = _route_parameter(
                name, param, muon_mode=muon_mode, routing=routing
            )
        if route == "frozen":
            continue
        if route == "muon":
            muon_params.append(param)
        else:
            adam_options = (adam_param_options_by_id or {}).get(id(param), {})
            group_lr = adam_options.get("lr", lr)
            group_weight_decay = adam_options.get("weight_decay", weight_decay)
            group_betas = tuple(adam_options.get("betas", adam_betas))
            group_eps = adam_options.get("eps", eps)
            group_amsgrad = bool(adam_options.get("amsgrad", amsgrad))
            key = (
                float(group_lr),
                float(group_weight_decay),
                group_betas,
                float(group_eps),
                group_amsgrad,
            )
            bucket = adam_group_buckets.setdefault(
                key,
                {
                    "params": [],
                    "route": "adam",
                    "lr": group_lr,
                    "weight_decay": group_weight_decay,
                    "betas": group_betas,
                    "eps": group_eps,
                    "amsgrad": group_amsgrad,
                },
            )
            bucket["params"].append(param)
        matrix_view = _matrix_view_shape(tuple(int(dim) for dim in param.shape), muon_mode)
        matrix_batch = matrix_view[0] if route == "muon" and matrix_view else None
        matrix_shape = matrix_view[-2:] if route == "muon" and matrix_view else None
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
            ).as_summary()
        )

    groups: list[dict] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "route": "muon",
                "lr": lr * muon_lr_factor,
                "weight_decay": muon_weight_decay,
                "beta": beta,
                "muon_mode": muon_mode,
                "routing": routing,
                "matrix_specs": muon_matrix_specs,
                "magma_lite": bool(magma_lite),
            }
        )
    groups.extend(adam_group_buckets.values())
    return groups, summary


def summarize_hybrid_muon_routes(summary: list[dict]) -> str:
    muon = [item for item in summary if item["route"] == "muon"]
    adam = [item for item in summary if item["route"] == "adam"]
    lines = [
        "HybridMuon parameter routing",
        f"Muon tensors: {len(muon)} ({sum(item['numel'] for item in muon)} parameters)",
        f"Adam tensors: {len(adam)} ({sum(item['numel'] for item in adam)} parameters)",
    ]
    for title, items in (("Muon", muon), ("Adam", adam)):
        for item in items:
            suffix = ""
            if item.get("matrix_shape") is not None:
                suffix = (
                    f" mode={item.get('muon_mode')}"
                    f" matrix_batch={item.get('matrix_batch')}"
                    f" matrix_shape={item.get('matrix_shape')}"
                )
            lines.append(
                f"{title}: {item['name']} shape={item['shape']} reason={item['reason']}{suffix}"
            )
    return "\n".join(lines)



def _magma_lite_scale(
    state: dict,
    score_key: str,
    grad_matrix: torch.Tensor,
    momentum_matrix: torch.Tensor,
) -> torch.Tensor:
    batch = int(grad_matrix.shape[0])
    grad_view = grad_matrix.reshape(batch, -1).to(dtype=torch.float32)
    momentum_view = momentum_matrix.reshape(batch, -1).to(dtype=torch.float32)
    score = state.get(score_key)
    if (
        score is None
        or not torch.is_tensor(score)
        or score.ndim != 1
        or score.numel() != batch
        or score.device != grad_matrix.device
    ):
        score = torch.full(
            (batch,), 0.5, dtype=torch.float32, device=grad_matrix.device
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
    return _MAGMA_MIN_SCALE + (1.0 - _MAGMA_MIN_SCALE) * score

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
        magma_lite = bool(group.get("magma_lite", False))
        updates_by_shape = {}
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            state = self.state[param]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(param)
            momentum = state["momentum"]
            momentum.mul_(beta).add_(grad, alpha=1.0 - beta)
            update = momentum.mul(beta).add(grad, alpha=1.0 - beta)
            specs = matrix_specs.get(id(param))
            if specs:
                flat_update = update.reshape(-1)
                for spec in specs:
                    matrix_update = _flat_spec_to_matrix_view(flat_update, spec)
                    grad_matrix = _flat_spec_to_matrix_view(grad.reshape(-1), spec)
                    momentum_matrix = _flat_spec_to_matrix_view(momentum.reshape(-1), spec)
                    rows, cols = matrix_update.shape[-2:]
                    scale = math.sqrt(max(1, rows / max(cols, 1)))
                    magma_scale = None
                    if magma_lite:
                        magma_scale = _magma_lite_scale(
                            state,
                            f"magma_score_{int(spec['offset'])}",
                            grad_matrix,
                            momentum_matrix,
                        )
                    key = ((rows, cols), matrix_update.device, matrix_update.dtype)
                    updates_by_shape.setdefault(key, []).append(
                        (param, matrix_update, scale, spec, magma_scale)
                    )
                continue
            matrix_view_shape = _matrix_view_shape(
                tuple(int(dim) for dim in update.shape), muon_mode
            )
            if matrix_view_shape is None:
                continue
            matrix_update = update.reshape(matrix_view_shape)
            grad_matrix = grad.reshape(matrix_view_shape)
            momentum_matrix = momentum.reshape(matrix_view_shape)
            rows, cols = matrix_update.shape[-2:]
            scale = math.sqrt(max(1, rows / max(cols, 1)))
            magma_scale = None
            if magma_lite:
                magma_scale = _magma_lite_scale(
                    state, "magma_score", grad_matrix, momentum_matrix
                )
            key = ((rows, cols), matrix_update.device, matrix_update.dtype)
            updates_by_shape.setdefault(key, []).append(
                (param, matrix_update, scale, None, magma_scale)
            )

        flat_deltas: dict[torch.nn.Parameter, torch.Tensor] = {}
        for records in updates_by_shape.values():
            total_batch = sum(record[1].shape[0] for record in records)
            if total_batch == 1:
                param, matrix_update, scale, spec, magma_scale = records[0]
                ortho = _orthogonalize_newton_schulz(matrix_update[0])
                if magma_scale is not None:
                    ortho = ortho * magma_scale.reshape(()).to(
                        dtype=ortho.dtype, device=ortho.device
                    )
                if spec is None:
                    param.add_(ortho.reshape_as(param), alpha=-lr * scale)
                else:
                    delta = flat_deltas.setdefault(param, torch.zeros_like(param).reshape(-1))
                    delta[spec["offset"] : spec["offset"] + spec["numel"]].add_(
                        _matrix_view_to_flat_spec(ortho, spec), alpha=scale
                    )
                continue

            stacked_updates = torch.cat([record[1] for record in records], dim=0)
            orthogonalized = _orthogonalize_newton_schulz_batched(stacked_updates)
            offset = 0
            for param, matrix_update, scale, spec, magma_scale in records:
                batch = matrix_update.shape[0]
                ortho = orthogonalized[offset : offset + batch]
                offset += batch
                if magma_scale is not None:
                    ortho = ortho * magma_scale.view(batch, 1, 1).to(
                        dtype=ortho.dtype, device=ortho.device
                    )
                if spec is None:
                    param.add_(ortho.reshape_as(param), alpha=-lr * scale)
                else:
                    delta = flat_deltas.setdefault(param, torch.zeros_like(param).reshape(-1))
                    delta[spec["offset"] : spec["offset"] + spec["numel"]].add_(
                        _matrix_view_to_flat_spec(ortho, spec), alpha=scale
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
            if not state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, dtype=torch.float32
                    )
            elif not torch.is_tensor(state["step"]):
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
            decoupled_weight_decay=False,
            amsgrad=amsgrad,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group.get("weight_decay", 0.0),
            eps=group.get("eps", 1.0e-8),
            maximize=False,
        )
