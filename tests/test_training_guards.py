import math

import pytest
import torch

from mace.tools.training_guards import (
    LossSkipController,
    NonFiniteGradGuard,
    stable_clip_grad_norm_,
)


def test_loss_skip_controller_accepts_finite_loss_and_updates_ema():
    controller = LossSkipController(
        manual_threshold=None,
        start_step=1,
        ema_window=2,
        multiplier=3.0,
        skip_nan=True,
        skip_large=True,
    )

    result = controller.check(torch.tensor(2.0), global_step=0)

    assert result.skip is False
    assert result.reason == "none"
    assert result.loss_ema == pytest.approx(2.0)


def test_loss_skip_controller_skips_nonfinite_loss_without_updating_ema():
    controller = LossSkipController(
        manual_threshold=None,
        start_step=0,
        ema_window=2,
        multiplier=3.0,
        skip_nan=True,
        skip_large=True,
    )

    result = controller.check(torch.tensor(float("nan")), global_step=1)

    assert result.skip is True
    assert result.reason == "nonfinite"
    assert result.loss_ema is None


def test_loss_skip_controller_skips_large_loss_after_warmup():
    controller = LossSkipController(
        manual_threshold=None,
        start_step=1,
        ema_window=2,
        multiplier=2.0,
        skip_nan=True,
        skip_large=True,
    )
    assert controller.check(torch.tensor(1.0), global_step=0).skip is False

    result = controller.check(torch.tensor(3.0), global_step=1)

    assert result.skip is True
    assert result.reason == "large"
    assert result.threshold == pytest.approx(2.0)


def test_stable_clip_grad_norm_returns_finite_norm_for_large_gradients():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([1.0e30])

    norm = stable_clip_grad_norm_([param], max_norm=1.0, stable=True)

    assert torch.isfinite(norm)
    assert norm.item() == pytest.approx(1.0e30)
    assert param.grad.abs().max().item() == pytest.approx(1.0)


def test_stable_clip_grad_norm_matches_torch_for_ordinary_gradients():
    ours = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    ref = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    ours.grad = torch.tensor([3.0, 4.0])
    ref.grad = ours.grad.clone()

    ours_norm = stable_clip_grad_norm_([ours], max_norm=2.0, stable=True)
    ref_norm = torch.nn.utils.clip_grad_norm_([ref], max_norm=2.0)

    assert ours_norm.item() == pytest.approx(ref_norm.item())
    torch.testing.assert_close(ours.grad, ref.grad)


def test_nonfinite_grad_guard_raises_and_resets():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([math.inf])
    guard = NonFiniteGradGuard()
    guard.update(torch.tensor(float("inf")))

    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        guard.raise_if_nonfinite(lambda: [("weight", param)])

    guard.raise_if_nonfinite(lambda: [("weight", param)])
