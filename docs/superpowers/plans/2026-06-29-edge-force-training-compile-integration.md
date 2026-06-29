# Edge-Vector Force Training Compile Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in edge-vector force compile equivalence gate that records whether a compiled force-loss candidate is acceptable for MACE training.

**Architecture:** Keep the implementation fail-closed and separate from default training. `mace.tools.force_compile` owns generic make_fx/Inductor graph mechanics, while `mace.tools.training_compile` gains MACE-training-specific config and gate result objects. Probe scripts and future training code consume the same gate-result shape.

**Tech Stack:** Python, PyTorch autograd/make_fx/torch.compile, MACE `ScaleShiftMACE`, pytest, existing RECIO probe scripts.

---

## File Structure

- Modify: `mace/tools/training_compile.py`
  - Add `EdgeForceCompileConfig`, `EdgeForceCompileGateResult`, `edge_force_compile_gate()`, and `edge_force_compile_result_from_trace()`.
- Modify: `scripts/benchmarks/recio8k_accel/probe_edge_vector_force_equivalence.py`
  - Add gate metadata to the existing make_fx edge-vector payload by using `edge_force_compile_result_from_trace()`.
- Modify: `tests/test_compile.py`
  - Add lightweight config, disabled gate, unsupported-output, and trace-result adapter tests.
- Modify: `tests/test_edge_vector_force_equivalence.py`
  - Add a direct unit test for gate metadata payload construction using the existing helper module.

## Task 1: Config And Disabled Gate

**Files:**
- Modify: `mace/tools/training_compile.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_compile.py`:

```python
def test_edge_force_compile_config_defaults_to_disabled():
    from mace.tools.training_compile import EdgeForceCompileConfig

    config = EdgeForceCompileConfig()

    assert config.enabled is False
    assert config.tracing_mode == "real"
    assert config.strip_detach is True
    assert config.compile_graph is True
    assert config.compile_mode == "default"
    assert config.compile_dynamic is True
    assert config.allow_fallback is True
    assert config.atol == 1.0e-5
    assert config.rtol == 1.0e-4


def test_edge_force_compile_gate_rejects_disabled_config():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=False),
    )

    assert result.enabled is False
    assert result.accepted is False
    assert result.fallback_reason == "disabled"
    assert result.detach_nodes_before is None
    assert result.detach_nodes_after is None
    assert result.node_count is None
```

- [ ] **Step 2: Run RED test**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_edge_force_compile_config_defaults_to_disabled tests/test_compile.py::test_edge_force_compile_gate_rejects_disabled_config -q
```

Expected: import failure for `EdgeForceCompileConfig`.

- [ ] **Step 3: Implement minimal config and disabled gate**

Add near the top of `mace/tools/training_compile.py` after imports:

```python
import dataclasses
from typing import Any


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


def edge_force_compile_gate(*, model, batch, config: EdgeForceCompileConfig):
    if not config.enabled:
        return EdgeForceCompileGateResult(
            enabled=False,
            accepted=False,
            fallback_reason="disabled",
        )
    return EdgeForceCompileGateResult(
        enabled=True,
        accepted=False,
        fallback_reason="gate_not_run",
    )
```

- [ ] **Step 4: Run GREEN test**

Run the same two tests. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: add edge force compile gate config"
```

## Task 2: Unsupported Output Rejection

**Files:**
- Modify: `mace/tools/training_compile.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_compile.py`:

```python
def test_edge_force_compile_gate_rejects_unsupported_outputs():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=True),
        compute_virials=True,
    )

    assert result.enabled is True
    assert result.accepted is False
    assert result.fallback_reason == "unsupported_outputs"
```

- [ ] **Step 2: Run RED test**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_edge_force_compile_gate_rejects_unsupported_outputs -q
```

Expected: `edge_force_compile_gate()` rejects the extra keyword or returns `gate_not_run`.

- [ ] **Step 3: Implement unsupported-output check**

Change `edge_force_compile_gate` signature to:

```python
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
```

Add after the disabled branch:

```python
    if any((
        compute_virials,
        compute_stress,
        compute_displacement,
        compute_hessian,
        compute_edge_forces,
        compute_atomic_stresses,
    )):
        return EdgeForceCompileGateResult(
            enabled=True,
            accepted=False,
            fallback_reason="unsupported_outputs",
        )
```

- [ ] **Step 4: Run GREEN test**

Run the three edge force gate tests. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: reject unsupported edge force compile outputs"
```

## Task 3: Trace Metadata Adapter

**Files:**
- Modify: `mace/tools/training_compile.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_compile.py`:

```python
def test_edge_force_compile_result_from_trace_records_gate_metadata():
    from mace.tools.force_compile import trace_force_closure
    from mace.tools.training_compile import edge_force_compile_result_from_trace

    def fn(x):
        y = x + x.detach().detach()
        return y

    x = torch.tensor([1.0], requires_grad=True)
    trace_result = trace_force_closure(
        fn,
        (x,),
        tracing_mode="real",
        strip_detach=True,
    )
    comparison = {"ok": True, "failed_checks": []}

    result = edge_force_compile_result_from_trace(
        trace_result=trace_result,
        comparison=comparison,
        compile_kwargs={"backend": "inductor", "dynamic": True},
    )

    assert result.enabled is True
    assert result.accepted is True
    assert result.fallback_reason is None
    assert result.detach_nodes_before >= 2
    assert result.detach_nodes_after == 0
    assert result.node_count == len(list(trace_result.graph_module.graph.nodes))
    assert result.comparison == comparison
    assert result.compile_kwargs == {"backend": "inductor", "dynamic": True}
```

- [ ] **Step 2: Run RED test**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py::test_edge_force_compile_result_from_trace_records_gate_metadata -q
```

Expected: import failure for `edge_force_compile_result_from_trace`.

- [ ] **Step 3: Implement trace metadata adapter**

Add to `mace/tools/training_compile.py`:

```python
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
```

- [ ] **Step 4: Run GREEN test**

Run the new test. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: record edge force compile trace metadata"
```

## Task 4: Probe Metadata Plumbing

**Files:**
- Modify: `scripts/benchmarks/recio8k_accel/probe_edge_vector_force_equivalence.py`
- Modify: `tests/test_edge_vector_force_equivalence.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_edge_vector_force_equivalence.py`:

```python
def test_edge_force_compile_result_payload_contains_gate_metadata():
    from mace.tools.force_compile import trace_force_closure
    from mace.tools.training_compile import edge_force_compile_result_from_trace

    def fn(x):
        y = x + x.detach().detach()
        return y

    x = torch.tensor([1.0], requires_grad=True)
    trace_result = trace_force_closure(fn, (x,), tracing_mode="real", strip_detach=True)
    result = edge_force_compile_result_from_trace(
        trace_result=trace_result,
        comparison={"ok": True, "failed_checks": []},
        compile_kwargs=None,
    )

    payload = result.__dict__

    assert payload["enabled"] is True
    assert payload["accepted"] is True
    assert payload["fallback_reason"] is None
    assert payload["detach_nodes_after"] == 0
```

- [ ] **Step 2: Run RED test**

Run the new test. Expected: pass if Task 3 is complete; if it passes, continue to Step 3 because this test locks payload shape before modifying the probe.

- [ ] **Step 3: Add gate metadata to the probe payload**

In `_make_fx_edge_vector_snapshot()`, import `edge_force_compile_result_from_trace` from `mace.tools.training_compile`, then include this key in the returned dictionary:

```python
        "gate_result": edge_force_compile_result_from_trace(
            trace_result=trace_result,
            comparison={"ok": True, "failed_checks": []},
            compile_kwargs=compile_kwargs,
        ).__dict__,
```

This initial payload records trace/compile metadata. The existing outer `comparison` remains the authoritative equivalence result until the next integration slice moves the full comparison into the gate.

- [ ] **Step 4: Run related tests**

Run:

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_edge_vector_force_equivalence.py tests/test_force_compile.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Use `git add -u` because `scripts/` is ignored for new files but this probe is tracked:

```bash
git add -u scripts/benchmarks/recio8k_accel/probe_edge_vector_force_equivalence.py tests/test_edge_vector_force_equivalence.py
git commit -m "feat: expose edge force compile gate metadata"
```

## Task 5: Verification And Documentation

**Files:**
- Modify: `docs/acceleration/recio8k_training_acceleration.md`

- [ ] **Step 1: Run focused regression**

```bash
source /opt/envs/anaconda3.env && conda activate mace_env
pytest tests/test_compile.py tests/test_force_compile.py tests/test_edge_vector_force_equivalence.py tests/test_force_backward_compile_ops.py tests/test_training_compile_probe.py -q
```

Expected: pass, with CUDA cases skipped if no GPU is visible.

- [ ] **Step 2: Syntax check**

```bash
python -m py_compile mace/tools/training_compile.py mace/tools/force_compile.py scripts/benchmarks/recio8k_accel/probe_edge_vector_force_equivalence.py tests/test_compile.py tests/test_edge_vector_force_equivalence.py
```

Expected: exit code 0.

- [ ] **Step 3: Diff check**

```bash
git diff --check
```

Expected: exit code 0.

- [ ] **Step 4: Document the gate status**

Add this paragraph to `docs/acceleration/recio8k_training_acceleration.md` near the force-loss compile section:

```markdown
The training integration gate for the edge-vector force compile path is intentionally fail-closed. `EdgeForceCompileConfig` defaults to disabled, unsupported outputs such as virials and stress are rejected, and probe payloads now expose gate metadata alongside the existing force-loss equivalence comparison. This does not enable compiled force training by default; it makes the next training-loop wrapper auditable before any RECIO speed or accuracy claim.
```

- [ ] **Step 5: Commit docs**

```bash
git add docs/acceleration/recio8k_training_acceleration.md
git commit -m "docs: outline edge force compile training gate"
```
