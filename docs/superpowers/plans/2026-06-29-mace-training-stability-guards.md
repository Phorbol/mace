# MACE Training Stability Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in loss skipping, stable gradient clipping, and non-finite gradient checkpoint guards to MACE training without changing default training behavior.

**Architecture:** Put guard primitives in `mace/tools/training_guards.py`, pass a compact `TrainingGuardConfig` through `run_train.py` into `mace/tools/train.py`, and keep the existing `take_step()` eager/compile-fallback structure intact. Tests are unit-first and use small fake models/batches before any RECIO sbatch smoke.

**Tech Stack:** Python, PyTorch, pytest, existing MACE CLI/configargparse, existing `mace_env` verification commands.

---

## File Structure

- Create: `mace/tools/training_guards.py`
  - Owns `TrainingGuardConfig`, `LossSkipController`, `stable_clip_grad_norm_`, `NonFiniteGradGuard`, and `raise_nonfinite_gradient_norm`.
- Modify: `mace/tools/train.py`
  - Accepts an optional guard config, constructs step/checkpoint guard state, applies skip/clip/guard logic in `take_step()`, and checks non-finite guard before checkpoint writes.
- Modify: `mace/tools/arg_parser.py`
  - Adds opt-in CLI/YAML flags for loss skip, stable grad clipping, and non-finite grad guard.
- Modify: `mace/cli/run_train.py`
  - Constructs `TrainingGuardConfig` from parsed args and passes it into `train()`.
- Create: `tests/test_training_guards.py`
  - Unit tests for guard primitives.
- Modify: `tests/test_compile.py`
  - Adds training-loop integration tests using existing fake model/batch patterns.

## Task 1: Guard Primitive Tests

**Files:**
- Create: `tests/test_training_guards.py`
- Create: `mace/tools/training_guards.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training_guards.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_training_guards.py -q
```

Expected: import failure because `mace.tools.training_guards` does not exist.

- [ ] **Step 3: Implement guard primitives**

Create `mace/tools/training_guards.py`:

```python
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


def stable_clip_grad_norm_(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    *,
    stable: bool = True,
) -> torch.Tensor:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return torch.zeros((), dtype=torch.float64)
    grads = [param.grad for param in params]
    if stable:
        scale = torch.stack(torch._foreach_norm(grads, float("inf"))).max()
        scale = torch.where(scale > 0, scale, scale.new_ones(()))
        scaled_norms = torch._foreach_norm(torch._foreach_div(grads, scale), 2.0)
        total_norm = scale.double() * torch.linalg.vector_norm(
            torch.stack(scaled_norms).double()
        )
    else:
        total_norm = torch.nn.utils.get_total_norm(grads, error_if_nonfinite=False)
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, total_norm)
    return total_norm


class NonFiniteGradGuard:
    def __init__(self) -> None:
        self._nonfinite: torch.Tensor | None = None

    def update(self, total_norm: torch.Tensor) -> None:
        nonfinite = ~torch.isfinite(total_norm)
        if self._nonfinite is not None:
            nonfinite = nonfinite | self._nonfinite.to(nonfinite.device)
        self._nonfinite = nonfinite

    def raise_if_nonfinite(
        self,
        named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]],
    ) -> None:
        if self._nonfinite is None:
            return
        should_raise = bool(self._nonfinite)
        self._nonfinite = None
        if should_raise:
            raise_nonfinite_gradient_norm(named_parameters())


def raise_nonfinite_gradient_norm(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    bad = []
    for name, param in named_parameters:
        if param.grad is None:
            continue
        norm = param.grad.detach().norm()
        if not torch.isfinite(norm):
            bad.append(f"  {name}: grad_norm={norm}, shape={list(param.shape)}")
    detail = "\n".join(bad) if bad else "  (all current individual gradients are finite)"
    raise RuntimeError(
        "Non-finite gradient norm; training has diverged.\n"
        f"Parameters with non-finite gradients:\n{detail}"
    )
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_training_guards.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit guard primitives**

Run:

```bash
git add mace/tools/training_guards.py tests/test_training_guards.py
git commit -m "feat: add training stability guard primitives"
```

## Task 2: `take_step()` Integration Tests

**Files:**
- Modify: `tests/test_compile.py`
- Modify: `mace/tools/train.py`

- [ ] **Step 1: Write failing integration tests**

Append to `tests/test_compile.py`:

```python
def test_take_step_skips_optimizer_and_ema_when_loss_skip_rejects_batch():
    from mace.tools.train import take_step
    from mace.tools.training_guards import TrainingGuardConfig

    class Batch:
        def __init__(self):
            self.value = torch.tensor([1.0])

        def to(self, device, non_blocking=False):
            self.value = self.value.to(device)
            return self

        def to_dict(self):
            return {"value": self.value}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch, **kwargs):
            return {"energy": self.weight * batch["value"]}

    class InfLoss(torch.nn.Module):
        def forward(self, pred, ref):
            return pred["energy"].sum() * torch.tensor(float("inf"))

    class RecordingOptimizer(torch.optim.SGD):
        def __init__(self, params):
            super().__init__(params, lr=0.1)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    class RecordingEMA:
        def __init__(self):
            self.calls = 0

        def update(self):
            self.calls += 1

    model = Model()
    optimizer = RecordingOptimizer(model.parameters())
    ema = RecordingEMA()

    loss, metrics = take_step(
        model=model,
        loss_fn=InfLoss(),
        batch=Batch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        guard_config=TrainingGuardConfig(loss_skip=True),
        global_step=0,
    )

    assert optimizer.step_calls == 0
    assert ema.calls == 0
    assert metrics["loss_skipped"] == 1
    assert metrics["loss_skip_reason"] == "nonfinite"
    assert model.weight.grad is None
    assert torch.isinf(loss)
```

Also add:

```python
def test_take_step_disabled_guards_preserve_optimizer_step():
    from mace.tools.train import take_step
    from mace.tools.training_guards import TrainingGuardConfig

    class Batch:
        def __init__(self):
            self.value = torch.tensor([1.0])

        def to(self, device, non_blocking=False):
            self.value = self.value.to(device)
            return self

        def to_dict(self):
            return {"value": self.value}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def forward(self, batch, **kwargs):
            return {"energy": self.weight * batch["value"]}

    class Loss(torch.nn.Module):
        def forward(self, pred, ref):
            return pred["energy"].sum()

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, metrics = take_step(
        model=model,
        loss_fn=Loss(),
        batch=Batch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=1.0,
        device=torch.device("cpu"),
        guard_config=TrainingGuardConfig(),
        global_step=0,
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(0.9)
    assert metrics["loss_skipped"] == 0
    assert metrics["loss_skip_reason"] == "none"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_take_step_skips_optimizer_and_ema_when_loss_skip_rejects_batch tests/test_compile.py::test_take_step_disabled_guards_preserve_optimizer_step -q
```

Expected: failure because `take_step()` does not accept `guard_config` or
`global_step`.

- [ ] **Step 3: Integrate guards into `take_step()`**

Modify imports in `mace/tools/train.py`:

```python
from .training_guards import (
    LossSkipController,
    NonFiniteGradGuard,
    TrainingGuardConfig,
    stable_clip_grad_norm_,
)
```

Add optional parameters to `train()`, `train_one_epoch()`, and `take_step()`:

```python
guard_config: Optional[TrainingGuardConfig] = None,
loss_skip_controller: Optional[LossSkipController] = None,
nonfinite_grad_guard: Optional[NonFiniteGradGuard] = None,
global_step: int = 0,
```

Inside `take_step()`, preserve the compile fallback but move the optimizer step
inside the non-skipped path:

```python
if guard_config is None:
    guard_config = TrainingGuardConfig()
if precision_config is None:
    precision_config = TrainingPrecisionConfig(enabled=False, dtype=None)

def closure():
    optimizer.zero_grad(set_to_none=True)
    with get_autocast_context(precision_config):
        output = model(...)
    loss = loss_fn(pred=output, ref=batch)
    skip_result = None
    if guard_config.loss_skip:
        if loss_skip_controller is None:
            raise RuntimeError("loss_skip requires a LossSkipController")
        skip_result = loss_skip_controller.check(loss, global_step=global_step)
        if skip_result.skip:
            optimizer.zero_grad(set_to_none=True)
            return loss, skip_result, None
    loss.backward()
    total_norm = None
    if max_grad_norm is not None:
        total_norm = stable_clip_grad_norm_(
            model.parameters(),
            max_norm=max_grad_norm,
            stable=guard_config.stable_grad_clip,
        )
        if guard_config.nonfinite_grad_guard:
            if nonfinite_grad_guard is None:
                raise RuntimeError("nonfinite_grad_guard requires guard state")
            nonfinite_grad_guard.update(total_norm)
    return loss, skip_result, total_norm
```

After fallback handling:

```python
loss, skip_result, total_norm = closure()
skipped = bool(skip_result and skip_result.skip)
if not skipped:
    optimizer.step()
    if ema is not None:
        ema.update()
```

Return metrics with default values:

```python
loss_dict = {
    "loss": to_numpy(loss),
    "time": time.time() - start_time,
    "loss_skipped": int(skipped),
    "loss_skip_reason": "none" if skip_result is None else skip_result.reason,
    "loss_skip_threshold": None if skip_result is None else skip_result.threshold,
    "loss_skip_ema": None if skip_result is None else skip_result.loss_ema,
    "grad_norm": None if total_norm is None else to_numpy(total_norm),
    "grad_norm_nonfinite": 0 if total_norm is None else int(not torch.isfinite(total_norm)),
}
```

- [ ] **Step 4: Thread guard state through `train_one_epoch()`**

In `train()`, construct guard state once:

```python
if guard_config is None:
    guard_config = TrainingGuardConfig()
loss_skip_controller = (
    LossSkipController(
        manual_threshold=guard_config.loss_skip_threshold,
        start_step=guard_config.loss_skip_start_step,
        ema_window=guard_config.loss_skip_ema_window,
        multiplier=guard_config.loss_skip_multiplier,
        skip_nan=guard_config.loss_skip_nan,
        skip_large=guard_config.loss_skip_large,
    )
    if guard_config.loss_skip
    else None
)
nonfinite_grad_guard = (
    NonFiniteGradGuard() if guard_config.nonfinite_grad_guard else None
)
global_step = 0
```

Pass state to `train_one_epoch()`, incrementing by its returned step count:

```python
steps_done = train_one_epoch(..., global_step_start=global_step, ...)
global_step += steps_done
```

Change `train_one_epoch()` to return the number of non-LBFGS batches processed.
For LBFGS return `1` after the existing closure path.

- [ ] **Step 5: Run integration tests**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_training_guards.py tests/test_compile.py::test_take_step_skips_optimizer_and_ema_when_loss_skip_rejects_batch tests/test_compile.py::test_take_step_disabled_guards_preserve_optimizer_step -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit training-loop integration**

Run:

```bash
git add mace/tools/train.py tests/test_compile.py
git commit -m "feat: apply training stability guards in take_step"
```

## Task 3: Checkpoint Guard Tests

**Files:**
- Modify: `tests/test_compile.py`
- Modify: `mace/tools/train.py`

- [ ] **Step 1: Write failing checkpoint guard test**

Add to `tests/test_compile.py`:

```python
def test_nonfinite_guard_checked_before_checkpoint_save(monkeypatch):
    import mace.tools.train as train_module
    from mace.tools.training_guards import NonFiniteGradGuard, TrainingGuardConfig

    class RaisingGuard(NonFiniteGradGuard):
        def raise_if_nonfinite(self, named_parameters):
            raise RuntimeError("guard checked")

    class Handler:
        def save(self, *args, **kwargs):
            raise AssertionError("checkpoint save should not run")

    guard = RaisingGuard()
    model = torch.nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="guard checked"):
        train_module._save_checkpoint_after_guard(
            checkpoint_handler=Handler(),
            state=train_module.CheckpointState(
                model,
                torch.optim.SGD(model.parameters(), lr=0.1),
                torch.optim.lr_scheduler.ExponentialLR(
                    torch.optim.SGD(model.parameters(), lr=0.1), gamma=0.9
                ),
            ),
            epochs=0,
            keep_last=False,
            nonfinite_grad_guard=guard,
            named_parameters=model.named_parameters,
        )
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_nonfinite_guard_checked_before_checkpoint_save -q
```

Expected: failure because `_save_checkpoint_after_guard` does not exist.

- [ ] **Step 3: Add checkpoint save helper and use it**

Add to `mace/tools/train.py`:

```python
def _save_checkpoint_after_guard(
    *,
    checkpoint_handler: CheckpointHandler,
    state: CheckpointState,
    epochs: int,
    keep_last: bool,
    nonfinite_grad_guard: Optional[NonFiniteGradGuard],
    named_parameters,
) -> None:
    if nonfinite_grad_guard is not None:
        nonfinite_grad_guard.raise_if_nonfinite(named_parameters)
    checkpoint_handler.save(state=state, epochs=epochs, keep_last=keep_last)
```

Replace every `checkpoint_handler.save(...)` in `train()` with this helper and
pass `model.named_parameters`.

- [ ] **Step 4: Run checkpoint guard test**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_nonfinite_guard_checked_before_checkpoint_save -q
```

Expected: pass.

- [ ] **Step 5: Commit checkpoint guard**

Run:

```bash
git add mace/tools/train.py tests/test_compile.py
git commit -m "feat: guard checkpoints against nonfinite gradients"
```

## Task 4: CLI and `run_train.py` Wiring

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Modify: `tests/test_run_train.py`

- [ ] **Step 1: Write failing parser test**

Add to `tests/test_run_train.py` or an existing parser-focused section:

```python
def test_training_guard_parser_flags(parser):
    args = parser.parse_args(
        [
            "--loss_skip",
            "--loss_skip_ema_window=8",
            "--loss_skip_multiplier=2.5",
            "--loss_skip_start_step=10",
            "--loss_skip_threshold=100.0",
            "--stable_grad_clip",
            "--nonfinite_grad_guard",
        ]
    )

    assert args.loss_skip is True
    assert args.loss_skip_ema_window == 8
    assert args.loss_skip_multiplier == 2.5
    assert args.loss_skip_start_step == 10
    assert args.loss_skip_threshold == 100.0
    assert args.stable_grad_clip is True
    assert args.nonfinite_grad_guard is True
```

If `tests/test_run_train.py` does not expose a reusable `parser` fixture, create
the parser with the same helper used by nearby tests.

- [ ] **Step 2: Run parser test to verify failure**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_run_train.py -q
```

Expected: failure for missing args.

- [ ] **Step 3: Add CLI flags**

In `mace/tools/arg_parser.py`, add near `--clip_grad`:

```python
parser.add_argument("--loss_skip", action="store_true", help="Skip non-finite or spike-loss training batches")
parser.add_argument("--loss_skip_nan", action=argparse.BooleanOptionalAction, default=True, help="Skip batches with non-finite loss when loss_skip is enabled")
parser.add_argument("--loss_skip_large", action=argparse.BooleanOptionalAction, default=True, help="Skip large-loss batches when loss_skip is enabled")
parser.add_argument("--loss_skip_ema_window", type=int, default=100, help="EMA window for dynamic loss-skip threshold")
parser.add_argument("--loss_skip_multiplier", type=float, default=3.0, help="Multiplier for dynamic loss-skip threshold")
parser.add_argument("--loss_skip_start_step", type=int, default=1000, help="Global step before large-loss skipping starts")
parser.add_argument("--loss_skip_threshold", type=float, default=None, help="Optional absolute loss-skip threshold")
parser.add_argument("--stable_grad_clip", action="store_true", help="Use overflow-stable gradient norm calculation for clipping")
parser.add_argument("--nonfinite_grad_guard", action="store_true", help="Block checkpoint writes after non-finite gradient norms")
```

- [ ] **Step 4: Wire config in `run_train.py`**

Import and construct:

```python
from mace.tools.training_guards import TrainingGuardConfig

guard_config = TrainingGuardConfig(
    loss_skip=args.loss_skip,
    loss_skip_nan=args.loss_skip_nan,
    loss_skip_large=args.loss_skip_large,
    loss_skip_ema_window=args.loss_skip_ema_window,
    loss_skip_multiplier=args.loss_skip_multiplier,
    loss_skip_start_step=args.loss_skip_start_step,
    loss_skip_threshold=args.loss_skip_threshold,
    stable_grad_clip=args.stable_grad_clip,
    nonfinite_grad_guard=args.nonfinite_grad_guard,
)
```

Pass `guard_config=guard_config` into `train(...)`.

- [ ] **Step 5: Run parser and focused training tests**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_run_train.py tests/test_training_guards.py tests/test_compile.py::test_take_step_skips_optimizer_and_ema_when_loss_skip_rejects_batch tests/test_compile.py::test_take_step_disabled_guards_preserve_optimizer_step -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit CLI wiring**

Run:

```bash
git add mace/tools/arg_parser.py mace/cli/run_train.py tests/test_run_train.py
git commit -m "feat: expose training stability guard flags"
```

## Task 5: Verification and RECIO Smoke Hook

**Files:**
- Modify: `docs/acceleration/recio8k_training_acceleration.md`

- [ ] **Step 1: Run full focused regression**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_training_guards.py tests/test_compile.py tests/test_training_precision.py tests/test_hybrid_muon.py tests/test_edge_vector_force_equivalence.py tests/test_force_backward_compile_ops.py tests/test_training_compile_probe.py -q
python -m py_compile mace/tools/training_guards.py mace/tools/train.py mace/tools/arg_parser.py mace/cli/run_train.py
git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Document SAI smoke command**

Append to `docs/acceleration/recio8k_training_acceleration.md`:

    ## Stability Guard Smoke

    The first stability-guard SAI smoke should reuse the RECIO/8k generated case
    template and add:

    ```bash
    --loss_skip \
    --loss_skip_ema_window=100 \
    --loss_skip_multiplier=3.0 \
    --loss_skip_start_step=1000 \
    --stable_grad_clip \
    --nonfinite_grad_guard
    ```

    Acceptance is finite training, unchanged checkpoint behavior, and log metrics
    for `loss_skipped`, `loss_skip_reason`, and `grad_norm`.

- [ ] **Step 3: Commit docs**

Run:

```bash
git add docs/acceleration/recio8k_training_acceleration.md
git commit -m "docs: add stability guard smoke command"
```

- [ ] **Step 4: Submit SAI smoke after implementation**

Use a generated RECIO/8k short-run directory and the SAI-compliant sbatch rules:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke/<case>
sbatch mace-recio8k.sbatch
```

The job script must use:

```bash
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=rush-1o2gpu
```

Expected: Slurm `COMPLETED`, no NaNs, and metrics showing guard fields.

## Self-Review Checklist

- This plan implements the accepted stability-guards spec.
- It does not implement WSD, uncertainty loss, D3 validation, force-compiled
  training integration, or broader HybridMuon routing.
- Each production-code behavior change has a failing test first.
- Each task ends with focused tests and a commit.
- Existing force compile and HybridMuon tests remain part of final regression.
