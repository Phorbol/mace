# Force-Loss Subgraph Compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a correctness gate proving that future compiled subgraphs preserve MACE force-loss energy, forces, and parameter gradients.

**Architecture:** Reuse the existing RECIO compile probe utilities and add a small equivalence helper that runs two model variants on cloned batches. Tests exercise the helper first with identical eager candidates, then it can be reused for compiled subgraph candidates.

**Tech Stack:** Python, PyTorch autograd, MACE `ScaleShiftMACE`, pytest, existing RECIO benchmark scripts.

---

### Task 1: Equivalence Helper Tests

**Files:**
- Create: `tests/test_training_compile_equivalence.py`
- Modify: `scripts/benchmarks/recio8k_accel/probe_training_compile.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_training_compile_equivalence.py` with tests importing the probe module and asserting that `compare_force_loss_equivalence` exists, returns `ok=True`, includes `energy_max_abs_diff`, `forces_max_abs_diff`, `loss_abs_diff`, and a non-empty `param_grad_max_abs_diff` mapping for two identical eager candidates.

- [ ] **Step 2: Run test to verify failure**

Run: `source /opt/envs/anaconda3.env && conda activate mace_env && pytest tests/test_training_compile_equivalence.py -q`
Expected: FAIL because `compare_force_loss_equivalence` is not defined.

- [ ] **Step 3: Implement minimal helper**

Add `compare_force_loss_equivalence(base_model, candidate_model, batch, *, atol, rtol)` to `probe_training_compile.py`. It must run force-loss backward on cloned batch dictionaries for each model, compare energy, forces, loss, and gradients for matching named parameters, and return a JSON-serializable dictionary.

- [ ] **Step 4: Run test to verify pass**

Run: `source /opt/envs/anaconda3.env && conda activate mace_env && pytest tests/test_training_compile_equivalence.py -q`
Expected: PASS.

### Task 2: Regression Scope

**Files:**
- Test: `tests/test_training_compile_probe.py`
- Test: `tests/test_recio8k_parse_metrics.py`

- [ ] **Step 1: Run related focused tests**

Run: `source /opt/envs/anaconda3.env && conda activate mace_env && pytest tests/test_training_compile_equivalence.py tests/test_training_compile_probe.py tests/test_recio8k_parse_metrics.py -q`
Expected: all tests pass.

- [ ] **Step 2: Check formatting**

Run: `git diff --check`
Expected: no output and exit code 0.

- [ ] **Step 3: Commit**

Run: `git add docs/superpowers/specs/2026-06-29-force-loss-subgraph-compile-design.md docs/superpowers/plans/2026-06-29-force-loss-subgraph-compile.md tests/test_training_compile_equivalence.py scripts/benchmarks/recio8k_accel/probe_training_compile.py && git commit -m "test: gate force-loss compile equivalence"`
