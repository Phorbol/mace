from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.optim import Optimizer
from torch.optim import _functional as optim_functional


@dataclass(frozen=True)
class RouteRecord:
    name: str
    shape: tuple[int, ...]
    numel: int
    route: str
    reason: str


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



def _route_parameter(name: str, param: torch.nn.Parameter) -> tuple[str, str]:
    lower = name.lower()
    if not param.requires_grad:
        return "frozen", "requires_grad=False"
    if param.ndim < 2:
        return "adam", "rank<2"
    effective_shape = tuple(int(dim) for dim in param.shape if int(dim) != 1)
    if len(effective_shape) < 2:
        return "adam", "effective-rank<2"
    if (
        param.ndim == 2
        and ".conv_tp_weights." in lower
        and lower.endswith(".weight")
    ):
        return "muon", "radial-tp-weight-mlp"
    if any(token in lower for token in _ADAM_NAME_TOKENS):
        if not any(token in lower for token in _MUON_NAME_TOKENS):
            return "adam", "sensitive-name"
    if param.ndim == 2 and any(token in lower for token in _MUON_NAME_TOKENS):
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
) -> tuple[list[dict], list[dict]]:
    if muon_lr_factor <= 0.0:
        raise ValueError("muon_lr_factor must be positive")

    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    summary: list[dict] = []
    for name, param in named_parameters:
        route, reason = _route_parameter(name, param)
        if route == "frozen":
            continue
        target = muon_params if route == "muon" else adam_params
        target.append(param)
        summary.append(
            RouteRecord(
                name=name,
                shape=tuple(int(dim) for dim in param.shape),
                numel=param.numel(),
                route=route,
                reason=reason,
            ).__dict__
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
            }
        )
    if adam_params:
        groups.append(
            {
                "params": adam_params,
                "route": "adam",
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": adam_betas,
                "eps": eps,
                "amsgrad": amsgrad,
            }
        )
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
            lines.append(
                f"{title}: {item['name']} shape={item['shape']} reason={item['reason']}"
            )
    return "\n".join(lines)


def _orthogonalize_newton_schulz(update: torch.Tensor, steps: int = 5) -> torch.Tensor:
    original_dtype = update.dtype
    x = update.float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    norm = x.norm().clamp_min(1.0e-7)
    x = x / norm
    for _ in range(steps):
        a = x @ x.T
        x = 1.5 * x - 0.5 * (a @ x)
    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


def _orthogonalize_newton_schulz_batched(
    updates: torch.Tensor, steps: int = 5
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
    norm = x.flatten(start_dim=1).norm(dim=1).clamp_min(1.0e-7)
    x = x / norm.view(-1, 1, 1)
    for _ in range(steps):
        a = x @ x.transpose(-2, -1)
        x = 1.5 * x - 0.5 * (a @ x)
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
            flat_update = update.reshape(update.shape[0], -1)
            scale = math.sqrt(
                max(1, flat_update.shape[0] / max(flat_update.shape[1], 1))
            )
            key = (tuple(flat_update.shape), flat_update.device, flat_update.dtype)
            updates_by_shape.setdefault(key, []).append((param, flat_update, scale))

        for records in updates_by_shape.values():
            if len(records) == 1:
                param, flat_update, scale = records[0]
                ortho = _orthogonalize_newton_schulz(flat_update).reshape_as(param)
                param.add_(ortho, alpha=-lr * scale)
                continue

            stacked_updates = torch.stack([record[1] for record in records])
            orthogonalized = _orthogonalize_newton_schulz_batched(stacked_updates)
            for (param, _flat_update, scale), ortho in zip(records, orthogonalized):
                param.add_(ortho.reshape_as(param), alpha=-lr * scale)

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
