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
- Stage 2 loss sweep: energy:forces = 20:1, 50:1, 100:1.
- HybridMuon config: `hybrid_muon_lr_factor=3.0`, `hybrid_muon_stage_two_route=adamw`, `hybrid_muon_adam_variant=adamw`.

## Completed Jobs

| Stage 2 E:F | Case | Job | State | Final E MAE (meV/atom) | Final F MAE (meV/A) | Train s/update | Peak FB mem (MB) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 20:1 | CUEQ + AdamW | 673670 | COMPLETED before timeout wrapper | 3.92 | 37.96 | 0.05107 | 10162 |
| 20:1 | CUEQ + HybridMuon | 674544 | COMPLETED 0:0 | 3.22 | 33.79 | 0.05433 | 10162 |
| 50:1 | CUEQ + AdamW | 673671 | COMPLETED before timeout wrapper | 2.85 | 39.17 | 0.05220 | 10162 |
| 50:1 | CUEQ + HybridMuon | 674543 | COMPLETED 0:0 | 2.63 | 35.44 | 0.05270 | 10162 |
| 100:1 | CUEQ + AdamW | 673484 | COMPLETED 0:0 | 2.35 | 40.57 | 0.05323 | 10162 |
| 100:1 | CUEQ + HybridMuon | 673485 | COMPLETED 0:0 | 2.31 | 37.73 | 0.05439 | 10162 |

Final delta, HybridMuon minus AdamW:

| Stage 2 E:F | Delta E MAE (meV/atom) | Delta F MAE (meV/A) | Delta train s/update | Relative train speed |
| --- | ---: | ---: | ---: | ---: |
| 20:1 | -0.70 | -4.17 | +0.00326 | 6.4% slower |
| 50:1 | -0.22 | -3.73 | +0.00050 | 1.0% slower |
| 100:1 | -0.04 | -2.84 | +0.00116 | 2.2% slower |

Peak memory was unchanged in these runs at 10162 MB.

## Evaluation History

100:1 stage-2 loss:

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

HybridMuon-only completed follow-up histories:

| Stage 2 E:F | Update | E MAE (meV/atom) | F MAE (meV/A) |
| --- | ---: | ---: | ---: |
| 20:1 | 20k | 126.34 | 41.17 |
| 20:1 | 40k | 88.59 | 37.11 |
| 20:1 | 60k | 66.76 | 35.35 |
| 20:1 | 80k | 54.86 | 33.42 |
| 20:1 | 100k | 46.22 | 32.41 |
| 20:1 | 120k | 39.53 | 31.47 |
| 20:1 | 140k | 35.96 | 30.18 |
| 20:1 | 160k | 7.81 | 36.00 |
| 20:1 | 180k | 5.56 | 33.90 |
| 20:1 | 200k | 3.22 | 33.79 |
| 50:1 | 20k | 126.34 | 41.17 |
| 50:1 | 40k | 88.59 | 37.11 |
| 50:1 | 60k | 66.43 | 35.41 |
| 50:1 | 80k | 54.62 | 33.42 |
| 50:1 | 100k | 45.95 | 32.43 |
| 50:1 | 120k | 40.43 | 31.33 |
| 50:1 | 140k | 35.61 | 30.21 |
| 50:1 | 160k | 5.96 | 38.21 |
| 50:1 | 180k | 6.50 | 35.33 |
| 50:1 | 200k | 2.63 | 35.44 |

## Interpretation

Before stage 2, HybridMuon strongly reduces energy MAE but force MAE remains slightly worse in the 100:1 paired run. In the HybridMuon-only 20:1 and 50:1 completions, the best force MAE is reached immediately before the stage-2 switch, around 30.2 meV/A at 140k.

The stage-2 switch reduces energy MAE and increases force MAE. The larger the stage-2 energy weight, the better the final energy and the worse the final force. Among the completed HybridMuon runs, 20:1 gives the best final force, 100:1 gives the best final energy, and 50:1 is the middle point.

Across all three stage-2 weights, HybridMuon improves final energy and force MAE over AdamW at the same batch budget. The speed cost is small but nonzero: about 1.0% to 6.4% slower by train-metrics seconds/update in these single-V100 runs.

This supports continuing the CUEQ + HybridMuon line and moving the next engineering effort toward update-based training control, LR warmup/WSD, loss-prefactor scheduling, and HybridMuon defaults. It does not yet justify compile work, because compile was disabled in this matrix and the current priority is to prove Muon convergence benefits under stable CUEQ training.

## Next Experiments

Recommended next matrix:

- Keep 20:1 as the force-oriented default and 50:1 as the balanced default.
- Re-run a 20k quick matrix after adding warmup/WSD and smooth loss-prefactor scheduling.
- If the 20k matrix preserves the same ordering, run 200k for AdamW vs HybridMuon under the new schedule.
- Add stress once the selected dataset has reliable stress labels and matching loss terms.

All jobs should explicitly set `MACE_OC20NEB_REPO_ROOT=/home/gengjianrui/worktrees/mace-update-boundary` and use the existing OC20NEB FPS data directory from the migrated data checkout.
