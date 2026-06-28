# MACE DPA4-Inspired Training Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in MACE training acceleration paths for HybridMuon, bf16 AMP, and training compile, then verify them on the RECIO/8k SAI benchmark without overwriting historical runs.

**Architecture:** Keep the features independent and fail-closed: optimizer routing lives in a focused optimizer module, AMP lives in a small precision policy used by the training step, and compile lives in a training-compile helper called after cueq/OEQ conversion but before optimizer creation. RECIO/8k benchmark scripts generate separate run directories and Slurm scripts from the existing dataset/config.

**Tech Stack:** PyTorch, torch.compile/Dynamo, cuEquivariance, MACE training CLI, pytest, SAI Slurm/sbatch, `mace_env` conda environment.

---

## File Structure

- Create `mace/tools/hybrid_muon.py`
  - Owns MACE-aware parameter routing and the HybridMuon optimizer implementation.
  - Exposes `HybridMuon`, `build_hybrid_muon_param_groups`, and `summarize_hybrid_muon_routes`.
- Create `mace/tools/precision.py`
  - Owns AMP dtype validation and autocast context creation.
  - Exposes `TrainingPrecisionConfig` and `get_autocast_context`.
- Create `mace/tools/training_compile.py`
  - Owns opt-in training compile preparation and fallback behavior.
  - Exposes `prepare_model_for_training_compile`.
- Modify `mace/tools/arg_parser.py`
  - Adds CLI/YAML options for `hybrid_muon`, AMP, and training compile.
- Modify `mace/tools/scripts_utils.py`
  - Integrates `hybrid_muon` into `get_optimizer`.
- Modify `mace/cli/run_train.py`
  - Applies compile preparation after cueq/OEQ conversion and before optimizer creation.
  - Builds a precision config and passes it into training.
- Modify `mace/tools/train.py`
  - Uses the precision context in the normal Adam-style training step.
  - Keeps optimizer step, checkpointing, EMA/SWA, loss conversion, and gradient clipping outside autocast.
- Create `tests/test_hybrid_muon.py`
  - Tests routing, optimizer stepping, route summaries, and checkpoint state dict reload.
- Create `tests/test_training_precision.py`
  - Tests dtype validation and no-op/autocast context behavior.
- Modify `tests/test_compile.py`
  - Adds a small training-compile smoke test using existing helper model/batch patterns.
- Create `scripts/benchmarks/recio8k_accel/generate_cases.py`
  - Generates isolated RECIO/8k benchmark case directories.
- Create `scripts/benchmarks/recio8k_accel/mace-recio8k-template.sbatch`
  - SAI-compatible single-GPU sbatch template.
- Create `scripts/benchmarks/recio8k_accel/parse_metrics.py`
  - Parses run logs and MACE results files for speed and accuracy summaries.
- Create `docs/acceleration/recio8k_training_acceleration.md`
  - Documents flags, benchmark commands, and acceptance gates.

## Task 1: HybridMuon Routing Tests

**Files:**
- Create: `tests/test_hybrid_muon.py`
- Create later implementation target: `mace/tools/hybrid_muon.py`

- [ ] **Step 1: Write failing routing tests**

Create `tests/test_hybrid_muon.py` with:

```python
import torch

from mace.tools.hybrid_muon import (
    build_hybrid_muon_param_groups,
    summarize_hybrid_muon_routes,
)


class TinyMaceLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radial_embedding = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 8),
        )
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(8, 1)])
        self.scale_shift = torch.nn.Parameter(torch.ones(1))
        self.atomic_energies_fn = torch.nn.Linear(5, 1, bias=False)
        self.products = torch.nn.Parameter(torch.randn(2, 3, 4, 5))

    def forward(self, x):
        return self.readouts[0](self.radial_embedding(x)).sum()


def _route_names(summary, route):
    return {entry["name"] for entry in summary if entry["route"] == route}


def test_hybrid_muon_routes_only_safe_dense_mace_weights():
    model = TinyMaceLike()
    groups, summary = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
    )

    muon_names = _route_names(summary, "muon")
    adam_names = _route_names(summary, "adam")

    assert "radial_embedding.0.weight" in muon_names
    assert "radial_embedding.2.weight" in muon_names
    assert "readouts.0.weight" in muon_names
    assert "radial_embedding.0.bias" in adam_names
    assert "readouts.0.bias" in adam_names
    assert "scale_shift" in adam_names
    assert "atomic_energies_fn.weight" in adam_names
    assert "products" in adam_names
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


def test_hybrid_muon_route_summary_is_loggable():
    model = TinyMaceLike()
    _, summary = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
    )

    text = summarize_hybrid_muon_routes(summary)

    assert "HybridMuon parameter routing" in text
    assert "Muon tensors:" in text
    assert "Adam tensors:" in text
    assert "radial_embedding.0.weight" in text
```

- [ ] **Step 2: Run routing tests and verify failure**

Run:

```bash
pytest tests/test_hybrid_muon.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'mace.tools.hybrid_muon'`.

- [ ] **Step 3: Commit failing test**

```bash
git add tests/test_hybrid_muon.py
git commit -m "test: add hybrid muon routing expectations"
```

## Task 2: HybridMuon Optimizer Implementation

**Files:**
- Create: `mace/tools/hybrid_muon.py`
- Modify: `tests/test_hybrid_muon.py`

- [ ] **Step 1: Implement routing helpers and optimizer skeleton**

Create `mace/tools/hybrid_muon.py`:

```python
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


def _effective_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape if int(dim) != 1)


def _route_parameter(name: str, param: torch.nn.Parameter) -> tuple[str, str]:
    lower = name.lower()
    shape = _effective_shape(tuple(param.shape))
    if not param.requires_grad:
        return "frozen", "requires_grad=False"
    if len(shape) < 2:
        return "adam", "rank<2"
    if any(token in lower for token in _ADAM_NAME_TOKENS):
        if not any(token in lower for token in _MUON_NAME_TOKENS):
            return "adam", "sensitive-name"
    if len(shape) == 2 and any(token in lower for token in _MUON_NAME_TOKENS):
        return "muon", "safe-dense-name"
    return "adam", "ambiguous"


def build_hybrid_muon_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    lr: float,
    weight_decay: float,
    muon_weight_decay: float,
    beta: float = 0.9,
    adam_betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
) -> tuple[list[dict], list[dict]]:
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
                "lr": lr,
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
            scale = math.sqrt(max(1, flat_update.shape[0] / max(flat_update.shape[1], 1)))
            param.add_(ortho, alpha=-lr * scale)

    def _step_adam_group(self, group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1.0e-8)
        weight_decay = group.get("weight_decay", 0.0)
        for param in group["params"]:
            if param.grad is None:
                continue
            grad = param.grad.float()
            state = self.state[param]
            if not state:
                state["step"] = torch.tensor(0, device=param.device)
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
            state["step"] += 1
            step = int(state["step"].item())
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if weight_decay:
                param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            update = exp_avg.div(bias_correction1).div_(denom)
            param.add_(update.to(dtype=param.dtype), alpha=-lr)
```

- [ ] **Step 2: Run routing tests**

Run:

```bash
pytest tests/test_hybrid_muon.py -q
```

Expected: pass the routing tests.

- [ ] **Step 3: Add optimizer step and state reload tests**

Append to `tests/test_hybrid_muon.py`:

```python
from mace.tools.hybrid_muon import HybridMuon


def test_hybrid_muon_step_updates_params_and_state_dict_reloads():
    torch.manual_seed(5)
    model = TinyMaceLike()
    groups, _ = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    loss = model(torch.randn(3, 4))
    loss.backward()
    optimizer.step()

    assert any(
        not torch.allclose(before[name], param)
        for name, param in model.named_parameters()
        if param.requires_grad
    )

    reloaded = HybridMuon(groups, lr=1.0e-3)
    reloaded.load_state_dict(optimizer.state_dict())
    assert reloaded.state_dict()["state"]
```

- [ ] **Step 4: Run HybridMuon tests**

Run:

```bash
pytest tests/test_hybrid_muon.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit HybridMuon module**

```bash
git add mace/tools/hybrid_muon.py tests/test_hybrid_muon.py
git commit -m "feat: add mace hybrid muon optimizer"
```

## Task 3: Optimizer CLI Integration

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/tools/scripts_utils.py`
- Test: `tests/test_hybrid_muon.py`

- [ ] **Step 1: Add failing optimizer factory test**

Append to `tests/test_hybrid_muon.py`:

```python
import argparse

from mace.tools.scripts_utils import get_optimizer


def test_get_optimizer_builds_hybrid_muon():
    model = TinyMaceLike()
    args = argparse.Namespace(
        optimizer="hybrid_muon",
        lr=1.0e-3,
        weight_decay=1.0e-4,
        hybrid_muon_weight_decay=0.0,
        beta=0.9,
        amsgrad=False,
    )
    param_options = {
        "params": [{"name": "all", "params": list(model.parameters()), "lr": args.lr}],
        "lr": args.lr,
        "amsgrad": args.amsgrad,
        "betas": (args.beta, 0.999),
    }

    optimizer = get_optimizer(args, param_options, named_parameters=model.named_parameters())

    assert optimizer.__class__.__name__ == "HybridMuon"
    assert {group["route"] for group in optimizer.param_groups} == {"muon", "adam"}
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_hybrid_muon.py::test_get_optimizer_builds_hybrid_muon -q
```

Expected: fail because `get_optimizer` does not accept `named_parameters`.

- [ ] **Step 3: Add parser options**

In `mace/tools/arg_parser.py`, change optimizer choices:

```python
choices=["adam", "adamw", "schedulefree", "hybrid_muon"],
```

Add after `--weight_decay`:

```python
parser.add_argument(
    "--hybrid_muon_weight_decay",
    help="Decoupled weight decay for HybridMuon-routed matrix parameters",
    type=float,
    default=0.0,
)
```

- [ ] **Step 4: Integrate factory**

Change `get_optimizer` signature in `mace/tools/scripts_utils.py`:

```python
def get_optimizer(
    args: argparse.Namespace,
    param_options: Dict[str, Any],
    named_parameters=None,
) -> torch.optim.Optimizer:
```

Add branch before `schedulefree`:

```python
    elif args.optimizer == "hybrid_muon":
        if named_parameters is None:
            raise ValueError("HybridMuon requires named_parameters for safe MACE routing")
        from mace.tools.hybrid_muon import (
            HybridMuon,
            build_hybrid_muon_param_groups,
            summarize_hybrid_muon_routes,
        )

        groups, route_summary = build_hybrid_muon_param_groups(
            named_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            muon_weight_decay=args.hybrid_muon_weight_decay,
            beta=args.beta,
            adam_betas=(args.beta, 0.999),
        )
        logging.info(summarize_hybrid_muon_routes(route_summary))
        optimizer = HybridMuon(groups, lr=args.lr, weight_decay=args.weight_decay)
```

- [ ] **Step 5: Pass named parameters from training setup**

In `mace/cli/run_train.py`, replace:

```python
optimizer = get_optimizer(args, param_options)
```

with:

```python
optimizer = get_optimizer(args, param_options, named_parameters=model.named_parameters())
```

- [ ] **Step 6: Run optimizer integration tests**

Run:

```bash
pytest tests/test_hybrid_muon.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit optimizer integration**

```bash
git add mace/tools/arg_parser.py mace/tools/scripts_utils.py mace/cli/run_train.py tests/test_hybrid_muon.py
git commit -m "feat: wire hybrid muon into training cli"
```

## Task 4: Training Precision Policy

**Files:**
- Create: `mace/tools/precision.py`
- Create: `tests/test_training_precision.py`
- Modify later: `mace/tools/arg_parser.py`

- [ ] **Step 1: Write failing precision tests**

Create `tests/test_training_precision.py`:

```python
import pytest
import torch

from mace.tools.precision import TrainingPrecisionConfig, get_autocast_context


def test_precision_none_returns_disabled_config():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))

    assert config.enabled is False
    assert config.dtype is None


def test_precision_rejects_bf16_on_cpu():
    with pytest.raises(ValueError, match="bf16 AMP requires a CUDA device"):
        TrainingPrecisionConfig.from_name("bf16", torch.device("cpu"))


def test_autocast_context_none_is_noop():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))
    x = torch.ones(2, dtype=torch.float32)
    with get_autocast_context(config):
        y = x + 1
    assert y.dtype == torch.float32
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest tests/test_training_precision.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'mace.tools.precision'`.

- [ ] **Step 3: Implement precision module**

Create `mace/tools/precision.py`:

```python
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainingPrecisionConfig:
    enabled: bool
    dtype: torch.dtype | None

    @classmethod
    def from_name(cls, name: str, device: torch.device) -> "TrainingPrecisionConfig":
        normalized = str(name).lower()
        if normalized in {"none", "false", "off", "fp32"}:
            return cls(enabled=False, dtype=None)
        if normalized == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 AMP requires a CUDA device")
            if not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 AMP is not supported by this CUDA device/build")
            return cls(enabled=True, dtype=torch.bfloat16)
        if normalized == "fp16":
            if device.type != "cuda":
                raise ValueError("fp16 AMP requires a CUDA device")
            return cls(enabled=True, dtype=torch.float16)
        raise ValueError(f"Unknown training AMP dtype: {name!r}")


def get_autocast_context(config: TrainingPrecisionConfig):
    if not config.enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=config.dtype)
```

- [ ] **Step 4: Run precision tests**

Run:

```bash
pytest tests/test_training_precision.py -q
```

Expected: all tests pass on CPU-only hosts.

- [ ] **Step 5: Commit precision policy**

```bash
git add mace/tools/precision.py tests/test_training_precision.py
git commit -m "feat: add training precision policy"
```

## Task 5: AMP Training Loop Integration

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/tools/train.py`
- Modify: `mace/cli/run_train.py`
- Test: `tests/test_training_precision.py`

- [ ] **Step 1: Add parser option**

In `mace/tools/arg_parser.py`, near the training options, add:

```python
parser.add_argument(
    "--train_amp_dtype",
    help="Autocast dtype for training forward pass: none, bf16, or fp16",
    type=str,
    default="none",
    choices=["none", "bf16", "fp16"],
)
```

- [ ] **Step 2: Extend `take_step` signature**

In `mace/tools/train.py`, import:

```python
from mace.tools.precision import TrainingPrecisionConfig, get_autocast_context
```

Change `take_step` signature:

```python
def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    precision_config: Optional[TrainingPrecisionConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
```

Inside `take_step`, before `closure`:

```python
    if precision_config is None:
        precision_config = TrainingPrecisionConfig(enabled=False, dtype=None)
```

Wrap only model forward in `closure`:

```python
        with get_autocast_context(precision_config):
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
```

Keep:

```python
        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(...)
        optimizer.step()
```

outside autocast.

- [ ] **Step 3: Thread precision config through `train_one_epoch`**

Add optional `precision_config` to `train_one_epoch` signature and pass it to `take_step`:

```python
precision_config: Optional[TrainingPrecisionConfig] = None,
```

```python
                precision_config=precision_config,
```

Add optional `precision_config` to `train` signature and pass into `train_one_epoch`.

- [ ] **Step 4: Build precision config in CLI**

In `mace/cli/run_train.py`, after device initialization and before calling `train`, add:

```python
    from mace.tools.precision import TrainingPrecisionConfig

    precision_config = TrainingPrecisionConfig.from_name(args.train_amp_dtype, device)
    if precision_config.enabled:
        logging.info("Using training AMP dtype: %s", precision_config.dtype)
```

Pass into `train(...)`:

```python
        precision_config=precision_config,
```

- [ ] **Step 5: Add a CPU no-op integration test**

Append to `tests/test_training_precision.py`:

```python
def test_precision_config_constructs_for_default_training_path():
    config = TrainingPrecisionConfig.from_name("none", torch.device("cpu"))
    assert config.enabled is False
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
pytest tests/test_training_precision.py tests/test_hybrid_muon.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit AMP integration**

```bash
git add mace/tools/arg_parser.py mace/tools/train.py mace/cli/run_train.py tests/test_training_precision.py
git commit -m "feat: add opt-in training amp"
```

## Task 6: Training Compile Helper

**Files:**
- Create: `mace/tools/training_compile.py`
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Add failing helper test**

Append to `tests/test_compile.py`:

```python
def test_training_compile_helper_noop_cpu():
    from mace.tools.training_compile import prepare_model_for_training_compile

    model = create_mace("cpu")
    compiled = prepare_model_for_training_compile(
        model,
        enabled=False,
        mode="default",
        fullgraph=False,
        allow_fallback=True,
    )

    assert compiled is model
```

- [ ] **Step 2: Run helper test and verify failure**

Run:

```bash
pytest tests/test_compile.py::test_training_compile_helper_noop_cpu -q
```

Expected: fail with `ModuleNotFoundError: No module named 'mace.tools.training_compile'`.

- [ ] **Step 3: Implement helper**

Create `mace/tools/training_compile.py`:

```python
from __future__ import annotations

import logging

import torch

from mace.tools import compile as mace_compile


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
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        logging.info(
            "Enabled training torch.compile: mode=%s fullgraph=%s",
            mode,
            fullgraph,
        )
        return compiled
    except Exception as exc:
        message = f"training torch.compile setup failed: {exc}"
        if allow_fallback:
            logging.warning("%s; continuing without training compile", message)
            return model
        raise RuntimeError(message) from exc
```

- [ ] **Step 4: Add parser options**

In `mace/tools/arg_parser.py`, add:

```python
parser.add_argument(
    "--train_compile",
    help="Enable torch.compile for the training model after cueq/OEQ conversion",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--train_compile_mode",
    help="torch.compile mode for training",
    type=str,
    default="default",
    choices=["default", "reduce-overhead", "max-autotune"],
)
parser.add_argument(
    "--train_compile_fullgraph",
    help="Request fullgraph=True for training torch.compile",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--train_compile_allow_fallback",
    help="Continue eager training when training torch.compile setup fails",
    action=argparse.BooleanOptionalAction,
    default=True,
)
```

- [ ] **Step 5: Integrate after cueq/OEQ conversion**

In `mace/cli/run_train.py`, after cueq/OEQ conversion and before `param_options = get_params_options(args, model)`, add:

```python
    if args.train_compile:
        from mace.tools.training_compile import prepare_model_for_training_compile

        model = prepare_model_for_training_compile(
            model,
            enabled=args.train_compile,
            mode=args.train_compile_mode,
            fullgraph=args.train_compile_fullgraph,
            allow_fallback=args.train_compile_allow_fallback,
        )
```

- [ ] **Step 6: Run compile helper test**

Run:

```bash
pytest tests/test_compile.py::test_training_compile_helper_noop_cpu -q
```

Expected: pass.

- [ ] **Step 7: Run existing compile smoke tests on CPU**

Run:

```bash
pytest tests/test_compile.py::test_mace_cpu_fp32 -q
```

If the exact parametrized node id differs, run:

```bash
pytest tests/test_compile.py -q -k "test_mace and cpu and fp32"
```

Expected: pass or skip only for unavailable CUDA/cueq cases.

- [ ] **Step 8: Commit training compile helper**

```bash
git add mace/tools/training_compile.py mace/tools/arg_parser.py mace/cli/run_train.py tests/test_compile.py
git commit -m "feat: add opt-in training compile"
```

## Task 7: Combined Local Verification

**Files:**
- Current implementation files from Tasks 1-6

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
pytest tests/test_hybrid_muon.py tests/test_training_precision.py tests/test_compile.py::test_training_compile_helper_noop_cpu -q
```

Expected: all pass.

- [ ] **Step 2: Run formatting/lint check if available**

Run:

```bash
python -m compileall mace/tools/hybrid_muon.py mace/tools/precision.py mace/tools/training_compile.py
```

Expected: command exits with code 0.

- [ ] **Step 3: Inspect CLI help**

Run:

```bash
python -m mace.cli.run_train --help | rg "hybrid_muon|train_amp_dtype|train_compile"
```

Expected: output includes all new options.

- [ ] **Step 4: Commit any verification fixes**

If Step 1-3 required fixes:

```bash
git add mace/tools mace/cli tests
git commit -m "fix: stabilize acceleration test coverage"
```

If no fixes were required, do not create an empty commit.

## Task 8: RECIO/8k Benchmark Harness

**Files:**
- Create: `scripts/benchmarks/recio8k_accel/generate_cases.py`
- Create: `scripts/benchmarks/recio8k_accel/mace-recio8k-template.sbatch`
- Create: `scripts/benchmarks/recio8k_accel/parse_metrics.py`
- Create: `docs/acceleration/recio8k_training_acceleration.md`

- [ ] **Step 1: Create sbatch template**

Create `scripts/benchmarks/recio8k_accel/mace-recio8k-template.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=MACE-RECIO8K-ACCEL
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=improper-gpu

set -euo pipefail

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &
source /opt/envs/anaconda3.env
conda activate mace_env

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/mace/cli/run_train.py --config=config.yaml

exit
```

- [ ] **Step 2: Create case generator**

Create `scripts/benchmarks/recio8k_accel/generate_cases.py`:

```python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


CASES = {
    "baseline_fp32_adam_cueq": {},
    "bf16_adam_cueq": {"train_amp_dtype": "bf16"},
    "fp32_hybrid_muon_cueq": {"optimizer": "hybrid_muon"},
    "bf16_hybrid_muon_cueq": {"optimizer": "hybrid_muon", "train_amp_dtype": "bf16"},
    "compile_fp32_adam_cueq": {"train_compile": True, "train_compile_allow_fallback": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    template = Path(__file__).with_name("mace-recio8k-template.sbatch")
    base_config = yaml.safe_load((source / "config.yaml").read_text())

    for name, overrides in CASES.items():
        case_dir = output / name
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "train.xyz", case_dir / "train.xyz")
        shutil.copy2(source / "les_config.yaml", case_dir / "les_config.yaml")
        shutil.copy2(template, case_dir / "mace-recio8k.sbatch")
        config = dict(base_config)
        config.update(overrides)
        config["max_num_epochs"] = args.epochs
        config["start_swa"] = max(1, int(args.epochs * 0.75))
        config["restart_latest"] = False
        config["name"] = f"RECIO-8k-{name}"
        (case_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create metric parser**

Create `scripts/benchmarks/recio8k_accel/parse_metrics.py`:

```python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch (?P<epoch>\d+):.*MAE_E_per_atom=\s*(?P<mae_e>[0-9.]+) meV, MAE_F=\s*(?P<mae_f>[0-9.]+)"
)


def parse_log(path: Path) -> dict:
    epochs = []
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epochs.append(
                {
                    "epoch": int(match.group("epoch")),
                    "mae_e_mev_atom": float(match.group("mae_e")),
                    "mae_f_mev_a": float(match.group("mae_f")),
                }
            )
    return {"log": str(path), "epochs": epochs, "last": epochs[-1] if epochs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    summaries = {}
    for root_name in args.roots:
        root = Path(root_name)
        for log in root.glob("*/logs/*.log"):
            summaries[f"{root.name}/{log.parents[1].name}"] = parse_log(log)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create benchmark docs**

Create `docs/acceleration/recio8k_training_acceleration.md`:

```markdown
# RECIO/8k Training Acceleration Benchmark

Generate isolated short-run cases:

```bash
python scripts/benchmarks/recio8k_accel/generate_cases.py \
  --output /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke \
  --epochs 20
```

Submit one case from a clean login-node shell:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke/bf16_hybrid_muon_cueq
sbatch mace-recio8k.sbatch
```

Parse completed logs:

```bash
python scripts/benchmarks/recio8k_accel/parse_metrics.py \
  /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke
```

The historical full baseline final stage-two validation is about `22.1 meV/atom` energy and `260.2 meV/A` force. Smoke runs are used for finite-loss, speed, and trend checks; long runs are required before claiming no accuracy or generalization regression.
```

- [ ] **Step 5: Run generator syntax check**

Run:

```bash
python -m py_compile scripts/benchmarks/recio8k_accel/generate_cases.py scripts/benchmarks/recio8k_accel/parse_metrics.py
```

Expected: exits with code 0.

- [ ] **Step 6: Generate dry-run case directories in `/tmp`**

Run:

```bash
python scripts/benchmarks/recio8k_accel/generate_cases.py --output /tmp/recio8k-accel-plan-check --epochs 2
find /tmp/recio8k-accel-plan-check -maxdepth 2 -name config.yaml -print
```

Expected: five `config.yaml` paths are printed.

- [ ] **Step 7: Commit benchmark harness**

```bash
git add scripts/benchmarks/recio8k_accel docs/acceleration/recio8k_training_acceleration.md
git commit -m "tools: add recio8k acceleration benchmark harness"
```

## Task 9: SAI Smoke Submission and Evidence Collection

**Files:**
- No source changes required unless a smoke failure reveals an implementation bug.
- Generated run dirs under `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-*`.

- [ ] **Step 1: Generate smoke cases**

Run:

```bash
python scripts/benchmarks/recio8k_accel/generate_cases.py \
  --output /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-$(date +%Y%m%d-%H%M%S) \
  --epochs 20
```

Expected: five isolated case directories are created.

- [ ] **Step 2: Submit baseline and one accelerated case first**

Run from the generated root:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-*/baseline_fp32_adam_cueq
sbatch mace-recio8k.sbatch
cd ../bf16_hybrid_muon_cueq
sbatch mace-recio8k.sbatch
```

Expected: `sbatch` prints submitted job IDs. Use `squeue -u $USER` or SAI aliases to monitor.

- [ ] **Step 3: Inspect failures without overwriting historical data**

For each completed or failed case:

```bash
tail -120 slurm-*.out
tail -80 logs/*.log
```

Expected: no NaN, no missing-gradient error, no checkpoint state error. If a case fails, fix the source issue, commit it, regenerate that case, and resubmit.

- [ ] **Step 4: Submit remaining smoke cases**

Run:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-*/bf16_adam_cueq
sbatch mace-recio8k.sbatch
cd ../fp32_hybrid_muon_cueq
sbatch mace-recio8k.sbatch
cd ../compile_fp32_adam_cueq
sbatch mace-recio8k.sbatch
```

Expected: jobs are submitted. Compile fallback is acceptable only if logs explain the exact reason.

- [ ] **Step 5: Parse metrics**

Run:

```bash
python scripts/benchmarks/recio8k_accel/parse_metrics.py \
  /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-* \
  > /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-summary.json
```

Expected: summary JSON contains `last` epoch metrics for every completed case.

- [ ] **Step 6: Decide long-run candidates**

Promote only cases that satisfy all smoke gates:

- finite loss for all epochs,
- no missing gradients,
- successful checkpoint save,
- validation metric trend not clearly worse than baseline at the same smoke epoch,
- step time or max-memory benefit visible in logs or `nvdmon_job-*.log`.

- [ ] **Step 7: Document smoke evidence**

Update `docs/acceleration/recio8k_training_acceleration.md` with:

```markdown
## Smoke Evidence

| Case | Slurm job | Status | Last epoch | MAE E meV/atom | MAE F meV/A | Notes |
| --- | --- | --- | --- | --- | --- | --- |
```

Fill one row per submitted case using the actual Slurm IDs and parsed metrics.

- [ ] **Step 8: Commit evidence doc**

```bash
git add docs/acceleration/recio8k_training_acceleration.md
git commit -m "docs: record recio8k acceleration smoke evidence"
```

## Final Verification Before Claiming Completion

Run local checks:

```bash
pytest tests/test_hybrid_muon.py tests/test_training_precision.py tests/test_compile.py::test_training_compile_helper_noop_cpu -q
python -m compileall mace/tools/hybrid_muon.py mace/tools/precision.py mace/tools/training_compile.py
python -m mace.cli.run_train --help | rg "hybrid_muon|train_amp_dtype|train_compile"
```

Inspect SAI evidence:

```bash
python scripts/benchmarks/recio8k_accel/parse_metrics.py /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke-* 
```

The phase is not complete until the evidence proves:

- HybridMuon works and checkpoints/restarts.
- bf16 AMP works without NaNs or missing gradients.
- compile either works in at least one training configuration or has a precise tested fallback/minimal reproducer.
- RECIO/8k accelerated runs do not show accuracy/generalization regression relative to the fp32 Adam cueq baseline.
- Historical RECIO/8k data and logs remain untouched.
