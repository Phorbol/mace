from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.optim import Optimizer


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
            ortho = _orthogonalize_newton_schulz(flat_update).reshape_as(param)
            scale = math.sqrt(
                max(1, flat_update.shape[0] / max(flat_update.shape[1], 1))
            )
            param.add_(ortho, alpha=-lr * scale)

    def _step_adam_group(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1.0e-8)
        weight_decay = group.get("weight_decay", 0.0)
        amsgrad = group.get("amsgrad", False)
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.float()
            if weight_decay:
                grad = grad.add(param.float(), alpha=weight_decay)
            state = self.state[param]
            if not state:
                state["step"] = torch.tensor(0, device=param.device)
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
                if amsgrad:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param, dtype=torch.float32
                    )
            state["step"] += 1
            step = int(state["step"].item())
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom_source = exp_avg_sq
            if amsgrad:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom_source = max_exp_avg_sq
            denom = denom_source.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            update = exp_avg.div(bias_correction1).div_(denom)
            param.add_(update.to(dtype=param.dtype), alpha=-lr)
