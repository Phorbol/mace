# Muon Update-First Training and Validation Design

## Context

The current acceleration branch already contains several review-driven fixes and controls:

- HybridMuon no longer serializes Python `id(param)` metadata in optimizer checkpoints and raises on missing Muon matrix views instead of silently skipping routed parameters.
- Adam/Muon decay semantics are split through Adam/AdamW routing, and HybridMuon has module-declared routing, match-RMS scaling, Magma-lite controls, and name-based runtime metadata.
- MACE training now accepts `max_num_updates`, `start_stage_two_update`, `eval_interval_updates`, and `checkpoint_interval_updates`.
- The latest checkpoint fix separates update checkpoints from epoch checkpoints and resumes with distinct `start_epoch` and `start_update` values.

The user direction is to pause compile-line experiments and focus on proving `cueq + HybridMuon` correctness, convergence, and speed. DPA4/deepmd-kit is available locally as `deepmd-kit-dpa4-reference`; no local TACE source was found during the design pass. DeepMD training is step-first: learning-rate schedule, checkpointing, validation, and logging are driven by resolved training steps rather than epochs.

Existing OC20NEB fullcase-200 20k artifacts already show the main signal on the 5k FPS train / 10k validation split:

| Case | E MAE (meV/atom) | F MAE (meV/A) | seconds/epoch | FB memory MB |
| --- | ---: | ---: | ---: | ---: |
| eager | 19.70 | 132.56 | 58.59 | 9010 |
| hybrid_muon | 17.82 | 104.35 | 60.09 | 9006 |
| cueq | 19.70 | 132.56 | 38.07 | 10162 |
| cueq_hybrid_muon | 17.82 | 104.35 | 39.77 | 10158 |

This is enough to justify focusing on HybridMuon accuracy/convergence and cueq runtime as the next production path, but the current reports are incomplete: scheduler/LR/Muon metadata are missing for older runs, RMSE is not always parsed, and epoch-based timing obscures update-based budget semantics.

## Goals

1. Make update count the authoritative training budget for the new benchmark path while keeping existing epoch-based MACE workflows compatible.
2. Align the default OC20NEB demo with the intended two-stage schedule: 20k updates total, stage two at 15k updates, stage one `E:F = 1:100`, stage two `E:F = 100:1`.
3. Validate HybridMuon against AdamW at equal update budgets, both without cueq and with cueq.
4. Report accuracy and speed objectively enough to decide whether HybridMuon improves convergence per batch and whether cueq provides the preferred runtime backend.
5. Keep compile experiments out of the primary matrix until `cueq + HybridMuon` is characterized.
6. Preserve correctness gates for optimizer routing, resume behavior, and silent no-op prevention.

## Non-Goals

- Rewriting the whole trainer into a pure step loop in one patch.
- Reopening edge-force compile experiments as part of the primary validation matrix.
- Running the 200k-update expansion before the 20k report path is reliable.
- Implementing distributed/FSDP Muon semantics in this phase.
- Treating batch-size sweep as mandatory; it remains a secondary task after the main comparison is stable.

## Recommended Architecture

Use a conservative update-first layer inside the existing MACE trainer.

The epoch loop remains the outer compatibility shell because MACE data loaders, logging, SWA, and existing scripts expect epochs. New production validation controls should use `updates_completed` as the source of truth:

- `max_num_updates` stops training exactly at the requested optimizer update budget.
- `start_stage_two_update` switches loss weights based on global update, not epoch.
- WSD per-step scheduling resolves `num_steps` from `max_num_updates` when present.
- `eval_interval_updates` and `checkpoint_interval_updates` determine validation and checkpoint cadence in update-budget experiments.
- Update checkpoints are saved as update checkpoints and resume via `start_update`, not via epoch reconstruction.

This keeps old command lines working and makes the benchmark path DPA4-like where it matters.

## Optimizer Configuration

The primary optimizer comparison should be:

| Case | Backend | Optimizer | Purpose |
| --- | --- | --- | --- |
| `adamw` | eager/non-cueq | AdamW | accuracy baseline |
| `hybrid_muon` | eager/non-cueq | HybridMuon | optimizer effect without cueq |
| `cueq_adamw` | cueq | AdamW | runtime baseline |
| `cueq_hybrid_muon` | cueq | HybridMuon | intended production path |

Recommended defaults for the 20k pass:

- `batch_size = 8` initially, because existing 20k artifacts use it and it fits V100 memory.
- `max_num_updates = 20000`.
- `start_stage_two_update = 15000`.
- `scheduler = WSD`, `lr_scheduler_interval = step`.
- `lr_wsd_warmup_ratio = 0.03`.
- `lr_wsd_decay_phase_ratio = 0.1`.
- `lr_wsd_stop_lr_ratio = 0.001`.
- `lr_wsd_decay_type = inverse_linear`.
- Adam path should be AdamW for fair modern baseline.
- HybridMuon should record `mode`, `routing`, `lr_factor`, `adam_variant`, `lr_scale_mode`, `match_rms_coeff`, and Magma-lite controls in manifest and summary.

The first controlled comparison should keep the existing HybridMuon defaults unless the report shows instability or weak convergence. Tuning Muon LR scale and Magma-lite should be a follow-up matrix, not mixed into the baseline proof.

## Reporting Requirements

The benchmark report must include one row per case with:

- case name and enabled features: cueq, HybridMuon, compile disabled.
- commit SHA, Python path, CUDA/PyTorch/e3nn/cueq versions when available.
- dataset path, train size, valid size, batch size, target updates, effective updates.
- stage one and stage two loss weights and stage-two start update.
- scheduler settings and resolved WSD steps/warmup/decay.
- optimizer metadata, including HybridMuon route summary counts.
- final and best validation metrics: E MAE, F MAE, E RMSE, F RMSE when logs expose them.
- wall-clock: total runtime, seconds/update, updates/sec, and legacy seconds/epoch for continuity.
- GPU memory: max framebuffer memory and optionally mean SM/memory utilization from nvdmon.
- nonfinite status, training fallback status, and resume source if resumed.

Pairwise comparisons should be generated for:

- `adamw -> hybrid_muon`.
- `cueq_adamw -> cueq_hybrid_muon`.

Each comparison should report absolute and ratio deltas for E MAE, F MAE, wall-clock/update, and memory.

## Correctness Gates

Before accepting new 20k/200k conclusions:

1. Route coverage: every trainable parameter is routed exactly once to AdamW/Adam/Muon or explicitly frozen.
2. Muon no-op prevention: any Muon-routed parameter without a matrix spec or valid matrix view raises.
3. Resume equivalence: uninterrupted training and restart-from-update-checkpoint produce equivalent next-step optimizer state and parameter updates on a fixed batch.
4. Checkpoint metadata: update checkpoints preserve both epoch and update counters.
5. Scheduler state: WSD resume preserves the correct current LR and global update.
6. Log parse coverage: summary output must include optimizer, scheduler, stage, and timing metadata for new runs.

These gates are stronger evidence than final MAE alone, because they verify the training path that produces the measurements.

## Experiment Plan

### Phase 1: 20k controlled repeat

Run the four primary cases on 1 V100:

- `adamw`.
- `hybrid_muon`.
- `cueq_adamw`.
- `cueq_hybrid_muon`.

Use the OC20NEB fullcase-200 FPS 5k train / 10k valid split already prepared under `runs/oc20neb_fullcase200_fps_extxyz`, unless the dataset manifest indicates it is stale or inconsistent.

Compile cases are excluded from the default `CASES` list for this phase.

### Phase 2: 20k report and decision

Generate `matrix_summary.json/csv` and `matrix_comparisons.json/csv`. The report should explicitly state whether HybridMuon improves final and best E/F MAE at the same update budget, and whether the cueq HybridMuon overhead is acceptable relative to cueq AdamW.

### Phase 3: 200k expansion

Only after Phase 2 passes correctness gates, run the same primary cases at 200k updates. The purpose is to test whether the 20k trend persists or reverses over a larger budget.

### Phase 4: secondary sweeps

Optional follow-ups:

- batch size sweep for speed/accuracy tradeoff.
- HybridMuon `lr_scale_mode=match_rms` with a separately tuned LR.
- Magma-lite warmup/bypass settings.
- Gram NS / column-padding performance work.

## Implementation Boundaries

Implement this in small commits:

1. Reporting cleanup: make summary parse and emit update-based runtime, scheduler metadata, optimizer metadata, and best/final metrics.
2. Script cleanup: set the OC20NEB default matrix to the four non-compile cases and make AdamW baseline explicit.
3. Correctness tests: add resume/scheduler metadata tests if missing.
4. Experiment submission: submit the 20k matrix only after code/report tests pass.
5. Report commit: commit scripts and code, not large `runs/` outputs, unless the project policy explicitly calls for checked-in summary artifacts.

## Acceptance Criteria

The phase is ready for review when:

- The branch contains committed code/scripts/spec updates and no accidental `runs/` additions.
- Targeted tests for update-based training, checkpoint resume, scheduler, and summary parsing pass in the `torch211+cu126` environment.
- A 20k four-case run produces a summary with complete optimizer/scheduler/update metadata.
- Pairwise comparisons can answer: HybridMuon vs AdamW at equal updates, and cueq HybridMuon vs cueq AdamW at equal updates.
- The reported conclusion distinguishes convergence-per-update from wall-clock speed.

## Open Risks

- Older logs lack complete metadata; they can guide design but should not be the final evidence after parser improvements.
- Batch size 8 may not be optimal for V100 throughput; keep it fixed for baseline comparability, then sweep later.
- HybridMuon may improve MAE per update but lose wall-clock if NS overhead dominates; the report must show both.
- cueq raises memory by about 1.1 GB in existing artifacts; this is acceptable on V100 for batch 8 but must be tracked.
- The epoch compatibility shell can still leak into logs; update-based fields must be made explicit enough that reports do not rely on epoch inference.
