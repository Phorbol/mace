# DPA4-Inspired Training Acceleration PR Notes

Suggested title:

```text
DPA4-inspired training acceleration for MACE
```

Suggested PR body:

```markdown
## Summary

- Add DPA4-inspired conservative force backward compile support for MACE training, with CUEQ-compatible Inductor graph lowering and guarded parity/fallback checks.
- Add training precision controls for TF32 and AMP dtype, plus per-step WSD scheduling and resolved scheduler logging.
- Add HybridMuon optimizer support with conservative MACE routing and TACE-style slice ablation support, while keeping Adam as the safe default.
- Add RECIO/OC20NEB benchmark wrappers and a portable best-practices guide for compile/CUEQ/Muon/mixed-precision training configs.

## Test Plan

- `/home/sjtu-caoxiaoming/gengjianrui/.conda/envs/mace_env/bin/python -m pytest tests/test_recio8k_scaling_scripts.py tests/test_oc20neb_fps_scripts.py tests/test_lr_scheduler.py tests/test_hybrid_muon.py tests/test_foundations.py tests/test_mdp_finetune.py -q`
  - `68 passed, 5 skipped`

## Notes

- Default training behavior remains conservative: compile is off, optimizer is Adam, and AMP dtype is none.
- `train_tf32` is enabled in benchmark wrappers as the recommended portable performance flag.
- HybridMuon is compatible with compile/CUEQ but is not recommended as the default recipe yet; current real-data evidence still favors Adam for the best throughput/accuracy tradeoff.
```

Compare URL:

```text
https://github.com/ACEsuit/mace/compare/develop...Phorbol:mace:dpa4-training-accel-20260703
```

Recommended stable benchmark knobs:

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

HybridMuon diagnostic knobs:

```bash
OPTIMIZER=hybrid_muon
HYBRID_MUON_MODE=2d
HYBRID_MUON_ROUTING=mace
HYBRID_MUON_LR_FACTOR=0.1
HYBRID_MUON_WEIGHT_DECAY=0.0
WEIGHT_DECAY=0.0
SCHEDULER=WSD
LR_SCHEDULER_INTERVAL=auto
```

Current real-data boundary:

- RECIO L1/C64 200k WSD Adam full-CUEQ full-Inductor compile: `00:53:22`, hot-step mean `12.77 ms/batch`, final test `53.5 meV/atom / 158.8 meV/A`.
- Matched RECIO eager control: `01:40:51`, hot-step mean `26.41 ms/batch`, final test `51.3 meV/atom / 158.9 meV/A`.
- OC20NEB FPS L1/C64 200k WSD Adam full-CUEQ full-Inductor compile: `01:08:02`, hot-step mean `14.69 ms/batch`, valid/test `18.2 meV/atom / 43.3 meV/A`.
- Matched OC20NEB eager control: `01:40:34`, hot-step mean `24.72 ms/batch`, valid/test `18.7 meV/atom / 42.9 meV/A`.
- Conservative HybridMuon is engineering-compatible with CUEQ plus full-Inductor force compile, but it is slower than Adam on the current L1/C64 RECIO gate and weaker than Adam on the current OC20NEB diagnostic gates.
- BF16 AMP remains experimental on V100 because the measured speed gain was small and energy MAE regressed in the current RECIO gate.
