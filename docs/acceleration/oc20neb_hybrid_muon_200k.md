# OC20NEB CUEQ HybridMuon 200k Notes

Date: 2026-07-16

## Dataset And Runtime

- Train set: OC20NEB fullcase-200 FPS subset, 5,000 structures.
- Validation/test set: OC20NEB fullcase-200 held-out subset, 10,000 structures.
- Code commit: `5d90fb4a1c9f7487f51494ecf96f37b71970deb2`.
- Environment: `torch211+cu126` MACE DPA4 conda env.
- Hardware: single V100 per run on SAI `16V100`.
- CUEQ: enabled.
- Compile: disabled.
- Batch size: train 8, valid 16.
- Target: 200,000 optimizer updates.
- Stage 1 loss: energy:forces = 1:100.
- Stage 2 starts at 150,000 updates.
- Stage 2 loss for this run: energy:forces = 100:1.
- HybridMuon config: `hybrid_muon_lr_factor=3.0`, `hybrid_muon_stage_two_route=adamw`, `hybrid_muon_adam_variant=adamw`.

## Completed Jobs

| Case | Job | State | Final E MAE (meV/atom) | Final F MAE (meV/A) | Train s/update | Peak FB mem (MB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 673484 | COMPLETED 0:0 | 2.35 | 40.57 | 0.05323 | 10162 |
| CUEQ + HybridMuon | 673485 | COMPLETED 0:0 | 2.31 | 37.73 | 0.05439 | 10162 |

Final delta, HybridMuon minus AdamW:

- Energy MAE: -0.04 meV/atom.
- Force MAE: -2.84 meV/A.
- Step time: +0.00116 s/update, about 2.2% slower.
- Peak memory: no measured difference.

## Evaluation History

| Update | AdamW E | HybridMuon E | Delta E | AdamW F | HybridMuon F | Delta F |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | 162.02 | 126.34 | -35.68 | 39.29 | 41.17 | +1.88 |
| 40k | 131.05 | 88.59 | -42.46 | 34.91 | 37.11 | +2.20 |
| 60k | 105.17 | 66.78 | -38.39 | 33.41 | 35.57 | +2.16 |
| 80k | 87.36 | 54.66 | -32.70 | 32.01 | 33.37 | +1.36 |
| 100k | 73.85 | 46.06 | -27.79 | 31.44 | 32.50 | +1.06 |
| 120k | 64.18 | 39.28 | -24.90 | 30.98 | 31.56 | +0.58 |
| 140k | 56.24 | 36.15 | -20.09 | 29.88 | 30.20 | +0.32 |
| 160k | 4.74 | 5.31 | +0.57 | 45.94 | 41.50 | -4.44 |
| 180k | 3.78 | 6.97 | +3.19 | 40.60 | 37.51 | -3.09 |
| 200k | 2.35 | 2.31 | -0.04 | 40.57 | 37.73 | -2.84 |

Lower is better. Delta is HybridMuon minus AdamW.

## Interpretation

Before stage 2, HybridMuon strongly reduces energy MAE but force MAE remains slightly worse. At 140k, immediately before the stage-2 switch, HybridMuon is much better on energy but still 0.32 meV/A worse on force.

The 100:1 stage-2 loss sharply reduces energy MAE for both optimizers but increases force MAE. HybridMuon, after switching its Muon group to AdamW for stage 2, shows a smaller force regression and finishes with better force MAE and similar energy MAE.

This supports continuing the CUEQ + HybridMuon line. It does not yet justify deeper optimizer kernel work by itself, because the measured speed penalty is small but nonzero and the result is from one seed and one 5k/10k subset.

## Follow-Up Sweep

The next sweep tests whether less aggressive stage-2 energy weighting preserves final force while keeping energy close:

| Stage 2 E:F | Job | Cases | Commit | Status at submission |
| --- | --- | --- | --- | --- |
| 50:1 | 673660 | `cueq_adamw,cueq_hybrid_muon` | `5d90fb4a1c9f7487f51494ecf96f37b71970deb2` | RUNNING |
| 20:1 | 673662 | `cueq_adamw,cueq_hybrid_muon` | `5d90fb4a1c9f7487f51494ecf96f37b71970deb2` | RUNNING |

Both jobs explicitly set `MACE_OC20NEB_REPO_ROOT=/home/gengjianrui/worktrees/mace-update-boundary` and use the existing OC20NEB FPS data directory from the migrated data checkout.
