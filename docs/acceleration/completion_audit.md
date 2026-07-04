# Training Acceleration Completion Audit

This audit records the current evidence for the DPA4-inspired MACE training acceleration goal. It intentionally separates PR readiness from full objective completion.

## Current Branch State

- Branch: `dpa4-training-accel-20260703`
- Latest reviewed commit when this audit was written: `aff6dc7 docs: add training acceleration PR notes`
- Base alignment: `upstream/develop` behind count was `0`
- PR compare URL: `https://github.com/ACEsuit/mace/compare/develop...Phorbol:mace:dpa4-training-accel-20260703`

## Verification Evidence

- Post-merge test gate:

  ```bash
  /home/sjtu-caoxiaoming/gengjianrui/.conda/envs/mace_env/bin/python -m pytest \
    tests/test_recio8k_scaling_scripts.py \
    tests/test_oc20neb_fps_scripts.py \
    tests/test_lr_scheduler.py \
    tests/test_hybrid_muon.py \
    tests/test_foundations.py \
    tests/test_mdp_finetune.py -q
  ```

  Result: `68 passed, 5 skipped`.

- Documentation-only PR notes gate:

  ```bash
  git diff --check HEAD~1..HEAD
  ```

  Result: passed.

## Requirement Audit

### Adam + CUEQ + Full-Inductor Force Compile

Status: proved for the tested RECIO L1/C64, RECIO L2/C128, and OC20NEB FPS gates.

Evidence:

- RECIO L1/C64 200k WSD Adam full-CUEQ full-Inductor compile completed in `00:53:22`, with `fallbacks=0`, `runtime_recompiles=0`, hot-step mean `12.77 ms/batch`, and test `53.5 meV/atom / 158.8 meV/A`.
- Matched RECIO eager control completed in `01:40:51`, hot-step mean `26.41 ms/batch`, and test `51.3 meV/atom / 158.9 meV/A`.
- RECIO L2/C128 200k WSD Adam full-CUEQ full-Inductor compile completed across seeds `123` and `789`, with `fallbacks=0`, `runtime_recompiles=0`, hot-step means `26.28` and `22.86 ms/batch`, and test `50.3 / 118.5` and `49.8 / 115.4` meV/atom / meV/A.
- Matched RECIO L2/C128 eager controls completed in `02:07:07` and `02:08:12`, with test `50.3 / 117.5` and `47.9 / 114.8` meV/atom / meV/A. The compile runs completed in `01:50:06` and `01:36:06`, so the two-seed mean wall-clock speedup was about `1.24x`.
- OC20NEB FPS L1/C64 200k WSD Adam full-CUEQ full-Inductor compile completed in `01:08:02`, with `fallbacks=0`, `runtime_recompiles=0`, hot-step mean `14.69 ms/batch`, and valid/test `18.2 meV/atom / 43.3 meV/A`.
- Matched OC20NEB eager control completed in `01:40:34`, hot-step mean `24.72 ms/batch`, and valid/test `18.7 meV/atom / 42.9 meV/A`.

Boundary:

- This is a real end-to-end speedup, but not DPA4-level `~3x` on the current V100/MACE/CUEQ stack.
- Compile increases CPU MaxRSS in the tested jobs.

### CUEQ Compatibility

Status: proved for the tested full-CUEQ training paths.

Evidence:

- The accepted RECIO and OC20NEB compile runs used full CUEQ with full-Inductor edge-force compile and completed without fallbacks or runtime recompiles.
- The wrappers keep CUEQ independent from compile so CUEQ can be disabled or reduced for architecture-specific wheel issues.

Boundary:

- CUEQ wheel/GPU architecture compatibility is environment-specific. V100 kernel image errors remain a deployment issue for some wheel versions, not a MACE training-loop correctness result.

### Per-Step WSD Scheduling

Status: implemented and tested.

Evidence:

- WSD supports `lr_scheduler_interval=auto`, resolving to per-step updates for WSD and per-epoch updates for legacy schedulers.
- The training logs include `LR scheduler resolved config`, making per-batch scheduling auditable.
- Unit coverage is included in `tests/test_lr_scheduler.py`.

Boundary:

- WSD is a fairness control for Adam vs HybridMuon comparisons; it does not by itself make the current HybridMuon recipe superior.

### HybridMuon

Status: engineering-compatible, not a recommended default.

Evidence:

- Conservative `hybrid_muon_routing=mace`, `hybrid_muon_mode=2d` completed RECIO L1/C64 200k WSD with full CUEQ and full-Inductor compile, with `fallbacks=0`, `runtime_recompiles=0`, and test `49.4 meV/atom / 158.5 meV/A`.
- TACE-style `routing=tace`, `mode=slice` completed OC20NEB diagnostics with clean compile health and increased Muon-covered parameters.

Boundary:

- Conservative HybridMuon was slower than Adam on the current RECIO L1/C64 gate.
- OC20NEB HybridMuon diagnostics were weaker than Adam; TACE-style slice routing was also a negative result at the tested LR/decay settings.
- HybridMuon should remain an explicit ablation until routing, learning rate, schedule, and decay are revalidated across datasets/seeds.

### BF16 AMP

Status: experimental on the tested V100 environment.

Evidence:

- BF16 execution completed in the current environment.

Boundary:

- The measured speed gain was small and energy MAE regressed in the RECIO gate.
- BF16 should not be the default until validated on native-bf16 hardware with parity and validation/test metrics.

### Precision And Generalization

Status: partially proved for Adam compile on the tested datasets; not universally proved for every acceleration path.

Evidence:

- Adam full-CUEQ full-Inductor compile matched eager force accuracy on RECIO and OC20NEB FPS at displayed precision.
- The RECIO L2/C128 two-seed gate matched eager test force within about `1 meV/A` on both seeds, while compile test energy was identical for seed `123` and `1.9 meV/atom` higher for seed `789`.
- OC20NEB FPS provides a second dataset beyond RECIO and used a separate train/valid extxyz split.

Boundary:

- The broad objective requires all acceleration paths to work without harming MACE precision/generalization. Current evidence proves this for the recommended Adam compile path, but not for HybridMuon as a default optimizer and not for BF16.

## Completion Decision

The branch is PR-ready, but the full research objective is not complete.

Recommended PR default:

```bash
EDGE_FORCE_COMPILE=True
EDGE_FORCE_GRAPH=True
EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True
ENABLE_CUEQ=True
CUEQ_PROFILE=full
TRAIN_TF32=True
TRAIN_AMP_DTYPE=none
OPTIMIZER=adam
SCHEDULER=WSD
LR_SCHEDULER_INTERVAL=auto
```

Do not default-enable:

- `OPTIMIZER=hybrid_muon`
- `TRAIN_AMP_DTYPE=bf16`
- TACE-style slice Muon routing

Next evidence needed before claiming full objective completion:

- Native-bf16 GPU run with validation/test parity against FP32.
- A revised HybridMuon recipe that matches or beats Adam on both speed and validation/test metrics, or a documented decision to keep Muon experimental.
