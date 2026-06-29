from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Literal

import torch
import torch.distributed as dist


@dataclasses.dataclass(frozen=True)
class TrainingGuardConfig:
    loss_skip: bool = False
    loss_skip_nan: bool = True
    loss_skip_large: bool = True
    loss_skip_ema_window: int = 100
    loss_skip_multiplier: float = 3.0
    loss_skip_start_step: int = 1000
    loss_skip_threshold: float | None = None
    stable_grad_clip: bool = False
    nonfinite_grad_guard: bool = False


@dataclasses.dataclass(frozen=True)
class LossSkipResult:
    skip: bool
    reason: Literal["none", "nonfinite", "large"]
    threshold: float | None
    loss_ema: float | None


class LossSkipController:
    def __init__(
        self,
        *,
        manual_threshold: float | None,
        start_step: int,
        ema_window: int | None,
        multiplier: float,
        skip_nan: bool,
        skip_large: bool,
    ) -> None:
        if ema_window is not None and ema_window <= 0:
            raise ValueError("ema_window must be positive")
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.manual_threshold = manual_threshold
        self.start_step = start_step
        self.multiplier = multiplier
        self.skip_nan = skip_nan
        self.skip_large = skip_large
        self.ema_alpha = None if ema_window is None else 1.0 / float(ema_window)
        self.loss_ema: float | None = None

    def _dynamic_threshold(self) -> float | None:
        if self.loss_ema is None:
            return None
        return self.multiplier * self.loss_ema

    def _effective_threshold(self) -> float | None:
        thresholds = [
            value
            for value in (self.manual_threshold, self._dynamic_threshold())
            if value is not None
        ]
        return min(thresholds) if thresholds else None

    def _update_ema(self, value: float) -> None:
        if self.ema_alpha is None:
            return
        if self.loss_ema is None:
            self.loss_ema = value
            return
        alpha = self.ema_alpha
        self.loss_ema = (1.0 - alpha) * self.loss_ema + alpha * value

    def check(self, loss: torch.Tensor, global_step: int) -> LossSkipResult:
        loss_detached = loss.detach()
        finite = bool(torch.isfinite(loss_detached).item())
        threshold = self._effective_threshold()
        reason_code = 0

        if self.skip_nan and not finite:
            reason_code = 1
        elif (
            self.skip_large
            and global_step >= self.start_step
            and finite
            and threshold is not None
            and float(loss_detached.item()) > threshold
        ):
            reason_code = 2

        if dist.is_available() and dist.is_initialized():
            tensor = torch.tensor(reason_code, device=loss.device, dtype=torch.int32)
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            reason_code = int(tensor.item())

        skip = reason_code > 0
        if not skip and finite:
            self._update_ema(float(loss_detached.item()))
            threshold = self._effective_threshold()

        reason: Literal["none", "nonfinite", "large"]
        reason = "none" if reason_code == 0 else "nonfinite" if reason_code == 1 else "large"
        return LossSkipResult(
            skip=skip,
            reason=reason,
            threshold=threshold,
            loss_ema=self.loss_ema,
        )


def _grad_tensors(parameters: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.grad.detach() for param in parameters if param.grad is not None]


def _stable_total_norm(grads: list[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.tensor(0.0)

    device = grads[0].device
    dtype = torch.float64 if grads[0].dtype == torch.float64 else torch.float32
    max_abs = torch.zeros((), device=device, dtype=dtype)
    for grad in grads:
        grad_abs_max = grad.detach().abs().max().to(device=device, dtype=dtype)
        max_abs = torch.maximum(max_abs, grad_abs_max)

    if max_abs == 0 or not torch.isfinite(max_abs):
        return max_abs

    sum_squares = torch.zeros((), device=device, dtype=dtype)
    for grad in grads:
        scaled = grad.detach().to(dtype=dtype) / max_abs
        sum_squares = sum_squares + torch.sum(scaled * scaled)
    return max_abs * torch.sqrt(sum_squares)


def stable_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    stable: bool,
) -> torch.Tensor:
    if not stable:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    params = list(parameters)
    grads = _grad_tensors(params)
    total_norm = _stable_total_norm(grads)
    if not grads:
        return total_norm

    clip_coef = torch.as_tensor(
        max_norm,
        device=total_norm.device,
        dtype=total_norm.dtype,
    ) / (total_norm + 1.0e-6)
    if bool(clip_coef < 1.0):
        for grad in grads:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
    return total_norm


def raise_nonfinite_gradient_norm(
    norm: torch.Tensor,
    named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]],
) -> None:
    if torch.isfinite(norm):
        return
    bad_names = [
        name
        for name, param in named_parameters()
        if param.grad is not None and not torch.all(torch.isfinite(param.grad))
    ]
    suffix = ""
    if bad_names:
        suffix = f"; non-finite gradients in: {', '.join(bad_names[:8])}"
    raise RuntimeError(f"Non-finite gradient norm detected ({norm.item()}){suffix}")


class NonFiniteGradGuard:
    def __init__(self) -> None:
        self._nonfinite_norm: torch.Tensor | None = None

    def update(self, norm: torch.Tensor) -> None:
        if not torch.isfinite(norm):
            self._nonfinite_norm = norm.detach()

    def raise_if_nonfinite(
        self,
        named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]],
    ) -> None:
        norm = self._nonfinite_norm
        self._nonfinite_norm = None
        if norm is not None:
            raise_nonfinite_gradient_norm(norm, named_parameters)
