# Muon Update-First Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-grade `cueq + HybridMuon` validation path with update-first reporting, explicit AdamW baselines, and a reliable 20k OC20NEB comparison matrix.

**Architecture:** Keep the existing MACE epoch loop as a compatibility shell and improve the benchmark/report layer so update count is the authoritative budget. The first implementation batch avoids compile changes and focuses on summary correctness, script defaults, and reproducible 20k validation commands.

**Tech Stack:** Python 3.11 in `/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126`, PyTorch 2.11/cu126 environment, MACE CLI, Slurm `sbatch`, `pytest`, shell `bash -n`.

## Global Constraints

- Do not add `runs/` outputs to git.
- Keep compile cases out of the default OC20NEB matrix for this phase.
- Primary cases are `adamw`, `hybrid_muon`, `cueq_adamw`, and `cueq_hybrid_muon`.
- 20k default budget: `max_num_updates=20000`, `start_stage_two_update=15000`, stage one `E:F=1:100`, stage two `E:F=100:1`.
- WSD should run per update with `lr_scheduler_interval=step` and resolve total steps from `max_num_updates`.
- Summaries must distinguish convergence-per-update from wall-clock speed.
- Use TDD for each code behavior change: failing test first, run it, implement, rerun.

---

## File Structure

- `scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py`
  - Owns OC20NEB matrix parsing and pairwise comparison rows.
  - Will gain final/best metric fields, update-based timing, manifest metadata fallback, and more stable case classification.

- `scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch`
  - Owns the submitted 20k benchmark matrix.
  - Will default to the four non-compile cases and explicitly use AdamW for Adam baselines.

- `tests/test_oc20neb_matrix_summary.py`
  - New focused tests for manifest/log/nvdmon parsing and pairwise comparisons.
  - Uses temporary fixture directories rather than real `runs/` outputs.

- `tests/test_update_based_training.py`, `tests/test_lr_scheduler.py`
  - Existing tests. Extend only if a task touches trainer/scheduler behavior.

- `docs/superpowers/specs/2026-07-12-muon-update-training-design.md`
  - Design source of truth. Do not alter in this plan unless implementation uncovers a contradiction.

---

### Task 1: Summary Parser Emits Update-Based Runtime and Manifest Metadata

**Files:**
- Modify: `scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py`
- Create: `tests/test_oc20neb_matrix_summary.py`

**Interfaces:**
- Consumes: `summarize_case(root: Path, case_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]`
- Produces: summary fields `seconds_per_update`, `updates_per_second`, `total_train_seconds_estimate`, `scheduler`, `lr_scheduler_interval`, `lr`, `weight_decay`, `hybrid_muon_*`, `best_mae_e_mev_atom`, `best_mae_f_mev_a`, `final_mae_e_mev_atom`, `final_mae_f_mev_a`.

- [ ] **Step 1: Write failing parser test for manifest metadata and update runtime**

Create `tests/test_oc20neb_matrix_summary.py` with this test skeleton:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import (
    summarize_case,
)


def _write_case(tmp_path: Path, case: str, log_text: str, nvdmon_text: str = ""):
    root = tmp_path / "matrix"
    case_dir = root / case
    log_dir = case_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / f"{case}.log").write_text(log_text)
    if nvdmon_text:
        (case_dir / f"nvdmon_job-1_{case}.log").write_text(nvdmon_text)
    return root, case_dir


def test_summarize_case_reports_update_runtime_and_manifest_metadata(tmp_path):
    manifest = {
        "target_steps": 20000,
        "max_num_updates": 20000,
        "train_size": 5000,
        "batch_size": 8,
        "scheduler": "WSD",
        "lr_scheduler_interval": "step",
        "lr_wsd_warmup_ratio": 0.03,
        "lr_wsd_stop_lr_ratio": 0.001,
        "lr_wsd_decay_phase_ratio": 0.1,
        "lr_wsd_decay_type": "inverse_linear",
        "lr": 0.001,
        "weight_decay": 0.001,
        "hybrid_muon_mode": "2d",
        "hybrid_muon_routing": "mace",
        "hybrid_muon_lr_factor": 0.1,
        "hybrid_muon_weight_decay": 0.0,
        "hybrid_muon_adam_variant": "adamw",
        "hybrid_muon_lr_scale_mode": "original",
        "stage_two_start_update": 15000,
    }
    log_text = """
Epoch 0: head: Default, loss=1.0, MAE_E_per_atom=20.00 meV, MAE_F=140.00 meV / A
Epoch 1: head: Default, loss=0.8, MAE_E_per_atom=18.00 meV, MAE_F=120.00 meV / A
Epoch 2: head: Default, loss=0.7, MAE_E_per_atom=19.00 meV, MAE_F=110.00 meV / A
Epoch 0 training: 10.0s
Epoch 1 training: 20.0s
Epoch 2 training: 30.0s
"""
    root, case_dir = _write_case(tmp_path, "cueq_hybrid_muon", log_text)

    row = summarize_case(root, case_dir, manifest)

    assert row["effective_updates"] == 20000
    assert row["steps_per_epoch"] == 625
    assert row["scheduler"] == "WSD"
    assert row["lr_scheduler_interval"] == "step"
    assert row["lr"] == 0.001
    assert row["weight_decay"] == 0.001
    assert row["hybrid_muon_adam_variant"] == "adamw"
    assert row["final_mae_e_mev_atom"] == pytest.approx(19.0)
    assert row["final_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["best_mae_e_mev_atom"] == pytest.approx(18.0)
    assert row["best_mae_f_mev_a"] == pytest.approx(110.0)
    assert row["mean_seconds_per_epoch"] == pytest.approx(20.0)
    assert row["seconds_per_update"] == pytest.approx(20.0 / 625)
    assert row["updates_per_second"] == pytest.approx(625 / 20.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_matrix_summary.py::test_summarize_case_reports_update_runtime_and_manifest_metadata -q
```

Expected: FAIL because new fields such as `final_mae_e_mev_atom`, `best_mae_e_mev_atom`, or `seconds_per_update` are missing.

- [ ] **Step 3: Implement summary fields**

In `summarize_fullcase200_ef20k_matrix.py`, extend `row` defaults with these exact keys:

```python
"seconds_per_update": None,
"updates_per_second": None,
"total_train_seconds_estimate": None,
"final_mae_e_mev_atom": None,
"final_mae_f_mev_a": None,
"final_rmse_e_mev_atom": None,
"final_rmse_f_mev_a": None,
"best_mae_e_mev_atom": None,
"best_mae_f_mev_a": None,
"best_rmse_e_mev_atom": None,
"best_rmse_f_mev_a": None,
```

After parsing log metrics, map legacy final fields to explicit final fields:

```python
row["final_mae_e_mev_atom"] = row["mae_e_mev_atom"]
row["final_mae_f_mev_a"] = row["mae_f_mev_a"]
row["final_rmse_e_mev_atom"] = row["rmse_e_mev_atom"]
row["final_rmse_f_mev_a"] = row["rmse_f_mev_a"]
```

Add a helper above `summarize_case`:

```python
def _best_metric_from_log(parsed: dict[str, Any], key: str) -> float | None:
    records = parsed.get("records") or []
    values = [record.get(key) for record in records if record.get(key) is not None]
    if values:
        return float(min(values))
    last = parsed.get("last") or {}
    value = last.get(key)
    return None if value is None else float(value)
```

Use it in `summarize_case`:

```python
row["best_mae_e_mev_atom"] = _best_metric_from_log(parsed, "mae_e_mev_atom")
row["best_mae_f_mev_a"] = _best_metric_from_log(parsed, "mae_f_mev_a")
row["best_rmse_e_mev_atom"] = _best_metric_from_log(parsed, "rmse_e_mev_atom")
row["best_rmse_f_mev_a"] = _best_metric_from_log(parsed, "rmse_f_mev_a")
```

Compute update timing after `effective_updates` is resolved and log timing is parsed:

```python
mean_epoch = row.get("mean_seconds_per_epoch")
steps_per_epoch = row.get("steps_per_epoch")
if mean_epoch is not None and steps_per_epoch:
    row["seconds_per_update"] = float(mean_epoch) / int(steps_per_epoch)
    row["updates_per_second"] = int(steps_per_epoch) / float(mean_epoch)
if row.get("seconds_per_update") is not None and row.get("effective_updates") is not None:
    row["total_train_seconds_estimate"] = (
        float(row["seconds_per_update"]) * int(row["effective_updates"])
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_matrix_summary.py::test_summarize_case_reports_update_runtime_and_manifest_metadata -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py tests/test_oc20neb_matrix_summary.py
git commit -m "Report update-based OC20NEB runtime metrics"
```

---

### Task 2: Pairwise Comparisons Use Final and Best Metrics

**Files:**
- Modify: `scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py`
- Modify: `tests/test_oc20neb_matrix_summary.py`

**Interfaces:**
- Consumes: `summarize_pairwise_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]`
- Produces: comparison fields for final and best E/F MAE plus `seconds_per_update_ratio` and `updates_per_second_ratio`.

- [ ] **Step 1: Write failing pairwise comparison test**

Append this test to `tests/test_oc20neb_matrix_summary.py`:

```python
from scripts.benchmarks.oc20neb_fps.summarize_fullcase200_ef20k_matrix import (
    summarize_pairwise_comparisons,
)


def test_pairwise_comparison_reports_final_best_and_update_speed():
    rows = [
        {
            "root": "run",
            "case": "cueq_adamw",
            "effective_updates": 20000,
            "stage_two_start_update": 15000,
            "scheduler": "WSD",
            "lr_scheduler_interval": "step",
            "final_mae_e_mev_atom": 20.0,
            "final_mae_f_mev_a": 130.0,
            "best_mae_e_mev_atom": 18.0,
            "best_mae_f_mev_a": 120.0,
            "seconds_per_update": 0.060,
            "updates_per_second": 16.6666667,
            "max_fb_memory_mb": 10000,
        },
        {
            "root": "run",
            "case": "cueq_hybrid_muon",
            "effective_updates": 20000,
            "stage_two_start_update": 15000,
            "scheduler": "WSD",
            "lr_scheduler_interval": "step",
            "hybrid_muon_mode": "2d",
            "hybrid_muon_routing": "mace",
            "hybrid_muon_lr_factor": 0.1,
            "final_mae_e_mev_atom": 17.0,
            "final_mae_f_mev_a": 105.0,
            "best_mae_e_mev_atom": 16.0,
            "best_mae_f_mev_a": 101.0,
            "seconds_per_update": 0.064,
            "updates_per_second": 15.625,
            "max_fb_memory_mb": 10150,
        },
    ]

    [comparison] = summarize_pairwise_comparisons(rows)

    assert comparison["baseline_case"] == "cueq_adamw"
    assert comparison["candidate_case"] == "cueq_hybrid_muon"
    assert comparison["final_mae_e_delta_mev_atom"] == -3.0
    assert comparison["final_mae_f_delta_mev_a"] == -25.0
    assert comparison["best_mae_e_delta_mev_atom"] == -2.0
    assert comparison["best_mae_f_delta_mev_a"] == -19.0
    assert comparison["seconds_per_update_ratio"] == pytest.approx(0.064 / 0.060)
    assert comparison["updates_per_second_ratio"] == pytest.approx(15.625 / 16.6666667)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_matrix_summary.py::test_pairwise_comparison_reports_final_best_and_update_speed -q
```

Expected: FAIL because comparison fields are absent and the pair mapping may not include `cueq_adamw` yet.

- [ ] **Step 3: Implement pair mappings and comparison fields**

Update `PAIRWISE_COMPARISONS`:

```python
PAIRWISE_COMPARISONS = (
    ("adamw", "hybrid_muon"),
    ("eager", "hybrid_muon"),
    ("cueq_adamw", "cueq_hybrid_muon"),
    ("cueq", "cueq_hybrid_muon"),
)
```

Extend `_comparison_row` with:

```python
"baseline_final_mae_e_mev_atom": baseline.get("final_mae_e_mev_atom"),
"candidate_final_mae_e_mev_atom": candidate.get("final_mae_e_mev_atom"),
"final_mae_e_delta_mev_atom": _numeric_delta(candidate.get("final_mae_e_mev_atom"), baseline.get("final_mae_e_mev_atom")),
"final_mae_e_ratio": _numeric_ratio(candidate.get("final_mae_e_mev_atom"), baseline.get("final_mae_e_mev_atom")),
"baseline_final_mae_f_mev_a": baseline.get("final_mae_f_mev_a"),
"candidate_final_mae_f_mev_a": candidate.get("final_mae_f_mev_a"),
"final_mae_f_delta_mev_a": _numeric_delta(candidate.get("final_mae_f_mev_a"), baseline.get("final_mae_f_mev_a")),
"final_mae_f_ratio": _numeric_ratio(candidate.get("final_mae_f_mev_a"), baseline.get("final_mae_f_mev_a")),
"baseline_best_mae_e_mev_atom": baseline.get("best_mae_e_mev_atom"),
"candidate_best_mae_e_mev_atom": candidate.get("best_mae_e_mev_atom"),
"best_mae_e_delta_mev_atom": _numeric_delta(candidate.get("best_mae_e_mev_atom"), baseline.get("best_mae_e_mev_atom")),
"best_mae_e_ratio": _numeric_ratio(candidate.get("best_mae_e_mev_atom"), baseline.get("best_mae_e_mev_atom")),
"baseline_best_mae_f_mev_a": baseline.get("best_mae_f_mev_a"),
"candidate_best_mae_f_mev_a": candidate.get("best_mae_f_mev_a"),
"best_mae_f_delta_mev_a": _numeric_delta(candidate.get("best_mae_f_mev_a"), baseline.get("best_mae_f_mev_a")),
"best_mae_f_ratio": _numeric_ratio(candidate.get("best_mae_f_mev_a"), baseline.get("best_mae_f_mev_a")),
"baseline_seconds_per_update": baseline.get("seconds_per_update"),
"candidate_seconds_per_update": candidate.get("seconds_per_update"),
"seconds_per_update_delta": _numeric_delta(candidate.get("seconds_per_update"), baseline.get("seconds_per_update")),
"seconds_per_update_ratio": _numeric_ratio(candidate.get("seconds_per_update"), baseline.get("seconds_per_update")),
"baseline_updates_per_second": baseline.get("updates_per_second"),
"candidate_updates_per_second": candidate.get("updates_per_second"),
"updates_per_second_delta": _numeric_delta(candidate.get("updates_per_second"), baseline.get("updates_per_second")),
"updates_per_second_ratio": _numeric_ratio(candidate.get("updates_per_second"), baseline.get("updates_per_second")),
```

Keep existing legacy MAE fields for backward compatibility.

- [ ] **Step 4: Run parser tests**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_matrix_summary.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py tests/test_oc20neb_matrix_summary.py
git commit -m "Compare OC20NEB final and best Muon metrics"
```

---

### Task 3: OC20NEB 20k Script Defaults to Four Non-Compile Cases and AdamW Baselines

**Files:**
- Modify: `scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch`
- Modify: existing script test file if present, otherwise create `tests/test_oc20neb_ef20k_script.py`

**Interfaces:**
- Consumes: shell env variables `MACE_OC20NEB_CASES`, `MACE_OC20NEB_BASE_OPTIMIZER`.
- Produces: default cases `adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon`; Adam baseline uses `--optimizer=adamw`.

- [ ] **Step 1: Write failing shell-script content test**

Create `tests/test_oc20neb_ef20k_script.py`:

```python
from pathlib import Path

SCRIPT = Path("scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch")


def test_oc20neb_20k_defaults_to_non_compile_adamw_muon_matrix():
    text = SCRIPT.read_text()

    assert "CASES=${MACE_OC20NEB_CASES:-adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon}" in text
    assert "BASE_OPTIMIZER=${MACE_OC20NEB_BASE_OPTIMIZER:-adamw}" in text
    assert "--optimizer="${BASE_OPTIMIZER}"" in text
    assert "run_selected_case adamw" in text
    assert "run_selected_case cueq_adamw" in text
    assert "run_selected_case compile" not in text.split("run_selected_case adamw", 1)[-1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_ef20k_script.py -q
```

Expected: FAIL because current defaults still include `eager`, `cueq`, and compile-capable case names.

- [ ] **Step 3: Modify sbatch defaults and case names**

In `fullcase200-ef-20k-demo.sbatch`:

Replace:

```bash
CASES=${MACE_OC20NEB_CASES:-eager,cueq,hybrid_muon,cueq_hybrid_muon}
```

with:

```bash
CASES=${MACE_OC20NEB_CASES:-adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon}
BASE_OPTIMIZER=${MACE_OC20NEB_BASE_OPTIMIZER:-adamw}
```

In `manifest.json`, add:

```json
  "base_optimizer": "${BASE_OPTIMIZER}",
```

In `common_args`, replace:

```bash
--optimizer=adam
```

with:

```bash
--optimizer="${BASE_OPTIMIZER}"
```

Replace bottom case calls with the four primary calls first:

```bash
run_selected_case adamw "${extra_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon "${extra_args[@]}" "${hybrid_muon_args[@]}" "${post_args[@]}"
run_selected_case cueq_adamw "${extra_args[@]}" "${cueq_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_args[@]}" "${post_args[@]}"
```

Keep compile case support below this block only as opt-in compatibility:

```bash
run_selected_case compile "${extra_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon_compile "${extra_args[@]}" "${hybrid_muon_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case cueq_compile "${extra_args[@]}" "${cueq_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon_compile "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_args[@]}" "${compile_args[@]}" "${post_args[@]}"
```

- [ ] **Step 4: Run tests and shell syntax check**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest tests/test_oc20neb_ef20k_script.py -q
bash -n scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch
```

Expected: PASS and no shell syntax output.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch tests/test_oc20neb_ef20k_script.py
git commit -m "Default OC20NEB 20k matrix to cueq Muon validation"
```

---

### Task 4: Validate Existing Artifacts With Improved Summary Parser

**Files:**
- Modify only if a parser bug is discovered: `scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py`
- No planned source file creation.

**Interfaces:**
- Consumes: existing untracked `runs/oc20neb_fullcase200_ef_20k/matrix_*_20260712` directories.
- Produces: command output showing rows for four primary cases with new metadata fields.

- [ ] **Step 1: Run parser on existing four-case artifacts**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py   runs/oc20neb_fullcase200_ef_20k/matrix_eager_a65aa53_20260712   runs/oc20neb_fullcase200_ef_20k/matrix_hybrid_muon_a65aa53_20260712   runs/oc20neb_fullcase200_ef_20k/matrix_cueq_a65aa53_20260712   runs/oc20neb_fullcase200_ef_20k/matrix_cueq_hybrid_muon_a65aa53_20260712
```

Expected: JSON contains `seconds_per_update`, `updates_per_second`, `final_mae_*`, and `best_mae_*` keys for every row. Older manifests may still have `null` scheduler/optimizer metadata; that is acceptable for old runs and not acceptable for new runs.

- [ ] **Step 2: If command fails, add a focused regression test**

If a specific missing-field or parse exception appears, add a fixture to `tests/test_oc20neb_matrix_summary.py` that reproduces the exact log or manifest pattern, then fix the parser.

The regression test must assert the exact recovered value. Example:

```python
def test_summarize_case_accepts_legacy_case_name_eager(tmp_path):
    root, case_dir = _write_case(
        tmp_path,
        "eager",
        "Epoch 1: head: Default, loss=0.8, MAE_E_per_atom=19.70 meV, MAE_F=132.56 meV / A
",
    )
    row = summarize_case(root, case_dir, {"target_steps": 20000, "train_size": 5000, "batch_size": 8})
    assert row["case"] == "eager"
    assert row["final_mae_f_mev_a"] == pytest.approx(132.56)
```

- [ ] **Step 3: Run full targeted tests**

Run:

```bash
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python -m pytest   tests/test_oc20neb_matrix_summary.py   tests/test_oc20neb_ef20k_script.py   tests/test_update_based_training.py   tests/test_lr_scheduler.py   -q
```

Expected: PASS.

- [ ] **Step 4: Commit only if source changed**

If Task 4 required parser fixes:

```bash
git add scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py tests/test_oc20neb_matrix_summary.py
git commit -m "Handle legacy OC20NEB summary artifacts"
```

If no source changed, do not create a commit.

---

### Task 5: Submit 20k Four-Case Validation Job

**Files:**
- No planned source modifications.
- Generated outputs stay under `runs/` and remain untracked unless explicitly requested.

**Interfaces:**
- Consumes: `scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch`
- Produces: Slurm job ID and later `matrix_summary.json`, `matrix_comparisons.json`, and logs under `runs/oc20neb_fullcase200_ef_20k/${JOB_ID}`.

- [ ] **Step 1: Confirm dataset exists**

Run:

```bash
test -f runs/oc20neb_fullcase200_fps_extxyz/train.extxyz
test -f runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
```

Expected: both commands exit 0.

- [ ] **Step 2: Submit four-case job on available V100 partition**

Run:

```bash
SUBMIT_OUTPUT=$(MACE_OC20NEB_CASES=adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon \
  MACE_OC20NEB_TARGET_STEPS=20000 \
  MACE_OC20NEB_STAGE_TWO=True \
  MACE_OC20NEB_STAGE_TWO_FRACTION=0.75 \
  MACE_OC20NEB_STAGE1_ENERGY_WEIGHT=1.0 \
  MACE_OC20NEB_STAGE1_FORCES_WEIGHT=100.0 \
  MACE_OC20NEB_STAGE2_ENERGY_WEIGHT=100.0 \
  MACE_OC20NEB_STAGE2_FORCES_WEIGHT=1.0 \
  sbatch --partition=16V100 --gpus-per-node=1 --qos=flood-1o2gpu \
    scripts/benchmarks/oc20neb_fps/fullcase200-ef-20k-demo.sbatch)
printf '%s\n' "${SUBMIT_OUTPUT}"
JOB_ID=$(printf '%s\n' "${SUBMIT_OUTPUT}" | awk '/Submitted batch job/ {print $4}')
test -n "${JOB_ID}"
RUN_ROOT="runs/oc20neb_fullcase200_ef_20k/${JOB_ID}"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
```

Expected: output matches `Submitted batch job [0-9]+`, `JOB_ID` is non-empty, and `RUN_ROOT` prints the default job run directory.

- [ ] **Step 3: Record job status**

Run:

```bash
squeue -u "$USER" -o '%.18i %.9P %.40j %.8T %.12M %.18R' | head -80
```

Expected: submitted job appears as `PENDING` or `RUNNING`, unless account GPU limits delay it.

- [ ] **Step 4: After completion, parse summaries**

Run after `RUN_ROOT` has been set from Step 2:

```bash
test -d "${RUN_ROOT}"
/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python \
  scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py \
  "${RUN_ROOT}" \
  --csv "${RUN_ROOT}/matrix_summary.csv" \
  --comparisons-csv "${RUN_ROOT}/matrix_comparisons.csv" \
  --comparisons-json "${RUN_ROOT}/matrix_comparisons.json" \
  > "${RUN_ROOT}/matrix_summary.json"
```

Expected: JSON and CSV files contain all four primary cases and two pairwise comparisons.

- [ ] **Step 5: Report results without committing large runs**

Summarize:

- final and best E/F MAE for each case.
- `seconds_per_update` and `updates_per_second` for each case.
- HybridMuon deltas vs AdamW.
- cueq HybridMuon deltas vs cueq AdamW.
- max GPU memory for each case.
- any nonfinite/fallback/resume flags.

Do not add `runs/` to git.

---

## Self-Review Notes

- Spec coverage: Task 1 and Task 2 cover reporting requirements; Task 3 covers the four-case non-compile default matrix; Task 4 validates parser behavior on existing artifacts; Task 5 submits the 20k validation run. Correctness gates already implemented in earlier commits are left in the targeted test suite.
- Placeholder scan: no deferred-work markers or unspecified error-handling steps remain.
- Type consistency: all referenced functions already exist except new test helpers local to `tests/test_oc20neb_matrix_summary.py`; fields produced by Task 1 are consumed by Task 2.
