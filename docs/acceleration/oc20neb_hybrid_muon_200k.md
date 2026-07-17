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

## 2026-07-17 Smooth 20:1 WSD 20k Check

SAI job `675964` reran the 20k OC20NEB FPS quick matrix after adding smooth
loss-prefactor scheduling. The job used the migrated extxyz split under
`/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz`,
single V100, CUEQ, WSD with per-step scheduling, batch size `8`, `20,000`
updates, stage two starting at update `15,000`, and a linear prefactor ramp from
`1:100` to `20:1` between updates `15,000` and `20,000`. HybridMuon used
`hybrid_muon_lr_factor=3.0`, `lr_scale_mode=match_rms`, Magma-lite, and switched
Muon-routed parameters to AdamW in stage two.

The job completed successfully in `01:00:00` with Slurm MaxRSS `5,672,984K`.
Both cases reported `20,006` observed updates and no NaNs. The structured output
is in `runs/oc20neb_fullcase200_ef_20k/675964/matrix_summary.csv` and
`matrix_comparisons.csv`.

| Case | Final E MAE (meV/atom) | Final F MAE (meV/A) | Best E MAE | Best F MAE | Seconds/update | Max FB memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 96.10 | 45.90 | 96.10 | 39.63 | 0.05488 | 10160 MB |
| CUEQ + HybridMuon | 60.42 | 46.27 | 60.42 | 42.11 | 0.05620 | 10160 MB |

HybridMuon improved final energy MAE by `35.68 meV/atom` (`0.63x` of AdamW),
but final force MAE was essentially unchanged and slightly worse by
`0.37 meV/A`; best force over the run was also worse by `2.48 meV/A`. It was
about `2.4%` slower by wrapper seconds/update and about `3.5%` slower by the
training-loop timing. The extra optimizer-step cost is visible:
`0.00324 s/update` for HybridMuon versus `0.00086 s/update` for AdamW.

This is a useful negative/neutral force result for the current 20k recipe. The
smooth `20:1` stage-two schedule avoids a hard loss jump and improves energy,
but it does not prove HybridMuon accelerates force convergence at this short
budget. The next optimizer test should not simply scale this exact 20k recipe to
200k. Better candidates are a 50:1 smooth schedule, a no-stage-two force-focused
control, or a revised Muon routing/LR recipe before spending another long
OC20NEB budget.

## 2026-07-17 Smooth 50:1 WSD 20k Check

SAI job `676044` repeated the same 20k quick matrix with the smooth stage-two
prefactor ramp ending at `50:1`. The job used the same migrated OC20NEB
fullcase-200 FPS split, single V100, CUEQ, WSD per-step scheduling, batch size
`8`, `20,000` updates, and stage two from update `15,000` to `20,000`.
HybridMuon used the same `hybrid_muon_lr_factor=3.0`,
`lr_scale_mode=match_rms`, Magma-lite, and stage-two route-to-AdamW settings as
the 20:1 check.

The job completed successfully in `01:00:33` with Slurm MaxRSS `5,707,848K`.
Both cases reported `20,006` observed updates and no NaNs. The structured output
is in `runs/oc20neb_fullcase200_ef_20k/676044/matrix_summary.csv` and
`matrix_comparisons.csv`.

| Case | Final E MAE (meV/atom) | Final F MAE (meV/A) | Best E MAE | Best F MAE | Seconds/update | Max FB memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 54.78 | 52.02 | 54.78 | 40.63 | 0.05490 | 10160 MB |
| CUEQ + HybridMuon | 36.51 | 47.84 | 36.51 | 43.65 | 0.05748 | 10162 MB |

HybridMuon improved final energy MAE by `18.27 meV/atom` and final force MAE by
`4.18 meV/A` at the end of the energy-heavy ramp. However, AdamW still reached
the better best force during the force-heavy portion of the run: `40.63 meV/A`
for AdamW versus `43.65 meV/A` for HybridMuon. HybridMuon was `4.7%` slower by
wrapper seconds/update and `4.9%` slower by training-loop timing; its optimizer
step was `0.00328 s/update` versus `0.00092 s/update` for AdamW.

The 50:1 smooth schedule is better than the 20:1 smooth schedule for final
energy and final force at 20k, but it still does not prove that HybridMuon
accelerates force convergence throughout training. The improvement appears
concentrated after the energy-weight ramp, while force-focused stage-one
behavior remains worse than AdamW. A 200k follow-up should therefore use this as
a balanced energy-oriented candidate, not as proof that the current Muon routing
already solves force convergence. A no-stage-two or delayed-stage-two
force-oriented control remains necessary.

## 2026-07-17 No-Stage Force-Focused WSD 20k Check

SAI job `676083` ran the force-focused control requested by the 20:1 and 50:1
smooth checks. It used the same OC20NEB fullcase-200 FPS split, single V100,
CUEQ, WSD per-step scheduling, batch size `8`, and `20,000` updates, but set
`MACE_OC20NEB_STAGE_TWO=False` and `MACE_OC20NEB_LOSS_PREFACTOR_SCHEDULE=off`.
The loss therefore stayed at `energy:forces = 1:100` for the whole run.
HybridMuon used the same `hybrid_muon_lr_factor=3.0`, `lr_scale_mode=match_rms`,
Magma-lite, and MACE routing settings as the smooth-schedule checks.

The job completed successfully in `00:54:06` with Slurm MaxRSS `5,646,404K`.
Both cases reported `20,006` observed updates and no NaNs. The structured output
is in `runs/oc20neb_fullcase200_ef_20k/676083/matrix_summary.csv` and
`matrix_comparisons.csv`.

| Case | Final E MAE (meV/atom) | Final F MAE (meV/A) | Best E MAE | Best F MAE | Seconds/update | Max FB memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 159.00 | 38.09 | 158.30 | 38.09 | 0.05473 | 10158 MB |
| CUEQ + HybridMuon | 128.11 | 40.44 | 128.11 | 40.44 | 0.05814 | 10156 MB |

HybridMuon again improved energy MAE, by `30.89 meV/atom`, but force MAE was
worse by `2.35 meV/A` for both final and best force because the no-stage run
kept improving force to the final checkpoint. HybridMuon was `6.2%` slower by
wrapper seconds/update and `5.7%` slower by training-loop timing. The optimizer
step cost was `0.00412 s/update` versus `0.00086 s/update` for AdamW.

This control separates the optimizer effect from the energy-heavy stage-two
ramp. Under a pure force-heavy `1:100` budget, the current HybridMuon routing and
LR recipe does not improve force convergence at 20k; it only improves energy.
The smooth 50:1 run therefore should be interpreted as an energy-ramp-assisted
final-force improvement, not proof that HybridMuon itself is better for the
force objective. Before any 200k force-oriented run, the next engineering work
should target HybridMuon routing/LR/defaults, or test a delayed-stage schedule
that preserves force-heavy training longer while using the 50:1 ramp only near
the end.

## 2026-07-17 No-Stage HybridMuon LR-Factor 1.0 Check

SAI job `676131` tested whether the no-stage force regression was caused by an
overly aggressive Muon learning-rate multiplier. It reused the same no-stage
force-focused setup as job `676083`: OC20NEB fullcase-200 FPS, single V100,
CUEQ, WSD per-step scheduling, batch size `8`, `20,000` updates,
`MACE_OC20NEB_STAGE_TWO=False`, and `MACE_OC20NEB_LOSS_PREFACTOR_SCHEDULE=off`.
Only `cueq_hybrid_muon` was run, with `hybrid_muon_lr_factor=1.0` instead of
`3.0`; all other HybridMuon settings stayed the same.

The job completed successfully in `00:27:36` with Slurm MaxRSS `5,589,404K` and
`20,006` observed updates. The structured output is in
`runs/oc20neb_fullcase200_ef_20k/676131/matrix_summary.csv`.

| Case | Muon LR factor | Final E MAE (meV/atom) | Final F MAE (meV/A) | Best F MAE | Seconds/update | Max FB memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW no-stage (`676083`) | n/a | 159.00 | 38.09 | 38.09 | 0.05473 | 10158 MB |
| CUEQ + HybridMuon no-stage (`676083`) | 3.0 | 128.11 | 40.44 | 40.44 | 0.05814 | 10156 MB |
| CUEQ + HybridMuon no-stage (`676131`) | 1.0 | 128.24 | 42.26 | 42.26 | 0.05791 | 10156 MB |

Lowering the Muon LR factor from `3.0` to `1.0` did not recover force accuracy.
Energy MAE stayed essentially unchanged, while final/best force worsened by
`1.82 meV/A` relative to the `3.0` run and by `4.17 meV/A` relative to AdamW.
The runtime cost also remained around `5.8%` slower than AdamW, dominated by the
HybridMuon optimizer step.

This makes LR factor alone an unlikely root cause for the no-stage force gap.
The next useful ablation should change which parameters are routed through Muon,
not merely lower the Muon LR. In particular, the current MACE routing only sends
8 radial tensor-product MLP matrices to Muon while keeping the large equivariant,
symmetric-contraction, skip, and readout weights on AdamW. The next candidate is
a routing experiment, such as `hybrid_muon_routing=module` or a narrower radial
subset, with the same no-stage force-focused setup.


## 2026-07-17 No-Stage CUEQ Broad-Routing Check

The first attempt to run `cueq_hybrid_muon_tace` exposed an important routing
bug rather than a valid optimizer ablation. Job `676152` started from commit
`07ab587` and was cancelled after the startup log showed that CUEQ+tace still
routed only the 8 radial tensor-product MLP tensors to Muon, the same effective
coverage as the conservative MACE route. The CUEQ conversion reshaped e3nn flat
weights to singleton-leading tensors such as `(1, numel)` and did not preserve
the original e3nn instruction metadata needed to recover MatrixSpec slices.

Commit `728d6c4` fixes this measurement bug by preserving HybridMuon slice specs
across e3nn-to-CUEQ conversion and by allowing `routing=tace` to consume those
module-declared specs. The focused validation was:

- `tests/test_hybrid_muon.py` plus the CUEQ converter tests: `44 passed`.
- Tiny CUEQ routing probe: CUEQ+tace now routes flat `linear_up`, `linear`,
  `skip_tp`, and product linear weights through module-declared MatrixSpecs.
- Corrected OC20NEB startup: `Muon tensors: 16 (467,968 parameters)`, including
  module-declared `linear_up`, `linear`, `skip_tp`, and `products.*.linear`
  weights. This is the first valid CUEQ broad-routing measurement.

Job `676174` was cancelled because it was scheduled on `16v100n03`, where Slurm
step statistics timed out and no training log/GPU progress occurred. The same
command was resubmitted with `--exclude=16v100n03` as job `676176`, which ran on
`16v100n04` and completed normally.

Job `676176` reused the no-stage force-focused setup from job `676083`: OC20NEB
fullcase-200 FPS, single V100, CUEQ, WSD per-step scheduling, batch size `8`,
`20,000` updates, `MACE_OC20NEB_STAGE_TWO=False`,
`MACE_OC20NEB_LOSS_PREFACTOR_SCHEDULE=off`, and constant `energy:forces =
1:100`. HybridMuon used `hybrid_muon_lr_factor=3.0`, `lr_scale_mode=match_rms`,
Magma-lite, and `routing=tace` after the CUEQ metadata fix.

The job completed successfully in `00:28:24` with Slurm MaxRSS `5,647,728K` and
`20,006` observed updates. The structured output is in
`runs/oc20neb_fullcase200_ef_20k/676176/matrix_summary.csv`.

| Case | Routing | Final E MAE (meV/atom) | Final F MAE (meV/A) | Best F MAE | Seconds/update | Optimizer s/update | Max FB memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW no-stage (`676083`) | n/a | 159.00 | 38.09 | 38.09 | 0.05473 | 0.00086 | 10158 MB |
| CUEQ + HybridMuon no-stage (`676083`) | mace/radial | 128.11 | 40.44 | 40.44 | 0.05814 | 0.00412 | 10156 MB |
| CUEQ + HybridMuon no-stage (`676176`) | tace/broad | 112.74 | 39.00 | 39.00 | 0.06064 | 0.00859 | 10154 MB |

Broad routing improved HybridMuon over radial-only routing by `15.37 meV/atom`
in final energy and `1.44 meV/A` in final force. It therefore fixes a real
coverage gap and partially closes the no-stage force deficit. However, it still
did not beat AdamW on the force objective at 20k: final/best force remained
`0.91 meV/A` worse than AdamW, while seconds/update was `10.8%` slower than
AdamW and `4.3%` slower than radial-only HybridMuon. The optimizer step cost
roughly doubled relative to radial-only HybridMuon because many more flat blocks
were orthogonalized.

The practical conclusion is that the previous `cueq_hybrid_muon_tace` case was
not measuring broad Muon under CUEQ at all; that is now fixed. The corrected
broad route is better than radial-only HybridMuon for both energy and force, but
it is not yet a sufficient force-convergence win over AdamW. The next ablation
should narrow the broad route rather than simply scale to 200k: likely compare
`linear/skip_tp` without product linear, skip species blocks separately, and a
lower LR specifically for module-declared flat specs while keeping radial TP MLP
on the current settings.


## 2026-07-17 CUEQ TACE Route-Subset 20k Checks

After CUEQ conversion was fixed to preserve module-declared HybridMuon slice
specs, job `676176` showed that full broad `routing=tace` used 16 Muon tensors
(`467,968` parameters) and improved energy over radial-only HybridMuon, but
force was still slightly worse than AdamW. To isolate whether all CUEQ flat
specs should be Muon-routed, commit `4e017b2` added
`--hybrid_muon_tace_module_include`, a comma-separated fnmatch filter that only
applies to module-declared specs under `routing=tace`. Filtered specs fall back
to AdamW and are logged as `module-spec-filtered`.

Two no-stage, force-focused `1:100` 20k checks were then run on the same
OC20NEB fullcase-200 FPS split, single V100, CUEQ, WSD per-step schedule,
`hybrid_muon_lr_factor=3.0`, `match_rms`, and Magma-lite:

| Case | Job | TACE module include | Muon tensors / params | Final E MAE | Final F MAE | Seconds/update | Optimizer s/update | Peak FB memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 676083 | n/a | n/a | 159.00 | 38.09 | 0.05473 | 0.00086 | 10158 MB |
| CUEQ + HybridMuon radial-only | 676083 | n/a | 8 / 74,752 | 128.11 | 40.44 | 0.05814 | 0.00412 | 10156 MB |
| CUEQ + HybridMuon full TACE | 676176 | `*` | 16 / 467,968 | 112.74 | 39.00 | 0.06064 | 0.00859 | 10154 MB |
| CUEQ + HybridMuon interactions | 676207 | `interactions.*` | 14 / 455,680 | 117.56 | 41.47 | 0.06049 | 0.00816 | 10154 MB |
| CUEQ + HybridMuon linear-only | 676208 | `interactions.*.linear.weight,interactions.*.linear_up.weight` | 12 / 144,384 | 97.79 | 40.05 | 0.05910 | 0.00745 | 10156 MB |

The route summaries confirmed the intended coverage:

- `676207` routed `linear_up`, `linear`, `skip_tp`, and radial
  `conv_tp_weights` through Muon, while `products.*.linear.weight` fell back to
  AdamW as `module-spec-filtered`.
- `676208` additionally filtered both `interactions.*.skip_tp.weight` tensors,
  leaving only `linear_up`, `linear`, and radial `conv_tp_weights` on Muon.

Interpretation:

- Narrowing module-declared routing to interaction `linear/linear_up` improved
  energy substantially versus full TACE (`97.79` vs `112.74 meV/atom`) but did
  not improve force versus full TACE (`40.05` vs `39.00 meV/A`).
- Full TACE remains the best HybridMuon force result among these no-stage 20k
  checks, but it is still worse than AdamW by `0.91 meV/A` and about `10.8%`
  slower by wrapper seconds/update.
- Excluding products alone was negative for both energy and force relative to
  full TACE. Excluding skip_tp as well helped energy but still failed to beat
  AdamW force.
- These results do not justify a 200k force-oriented run with the current TACE
  routing recipe. The next optimizer changes should target LR/routing defaults
  or a delayed energy ramp, not simply broader Muon coverage.

Structured outputs used here:

- `runs/oc20neb_fullcase200_ef_20k/676176/matrix_summary.csv`
- `runs/oc20neb_fullcase200_ef_20k/676207/matrix_summary.csv`
- `runs/oc20neb_fullcase200_ef_20k/676208/matrix_summary.csv`


## 2026-07-17 CUEQ TACE Module-LR-Scale 20k Checks

The route-subset checks left one plausible explanation for the broad TACE force
shortfall: the module-declared CUEQ flat specs might need a smaller Muon LR than
the radial tensor-product MLP matrices. Commit `2231efe` added
`--hybrid_muon_tace_module_lr_scale`, an extra positive per-parameter LR
multiplier that applies only to module-declared Muon specs under
`routing=tace`. Radial `tace-matrix-muon` dense matrices keep the base
`hybrid_muon_lr_factor=3.0` setting. Commit `610b900` then made non-default
per-parameter LR scales visible in the HybridMuon route summary for future runs.

Jobs `676250` and `676251` reran the no-stage, force-focused `1:100` 20k setup
with CUEQ, WSD per-step scheduling, `routing=tace`, full module include `*`,
`match_rms`, and Magma-lite. The only changed parameter was
`MACE_OC20NEB_HYBRID_MUON_TACE_MODULE_LR_SCALE`.

| Case | Job | TACE module LR scale | Final E MAE | Final F MAE | Best F MAE | Seconds/update | Optimizer s/update | Peak FB memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW | 676083 | n/a | 159.00 | 38.09 | 38.09 | 0.05473 | 0.00086 | 10158 MB |
| CUEQ + HybridMuon full TACE | 676176 | 1.0 | 112.74 | 39.00 | 39.00 | 0.06064 | 0.00859 | 10154 MB |
| CUEQ + HybridMuon linear-only | 676208 | n/a | 97.79 | 40.05 | 40.05 | 0.05910 | 0.00745 | 10156 MB |
| CUEQ + HybridMuon full TACE, scaled modules | 676250 | 0.5 | 123.35 | 41.45 | 41.45 | 0.06192 | 0.00881 | 9420 MB |
| CUEQ + HybridMuon full TACE, scaled modules | 676251 | 0.25 | 136.86 | 43.75 | 42.67 | 0.06204 | 0.00881 | 9418 MB |

Matched-update validation showed the same ordering by mid-run:

| Job | Module LR scale | F MAE @4k | F MAE @8k | F MAE @12k | F MAE @16k | F MAE @20k |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 676176 | 1.0 | 45.11 | 46.73 | 43.54 | 41.09 | 39.00 |
| 676250 | 0.5 | 43.73 | 47.83 | 46.27 | 43.36 | 41.45 |
| 676251 | 0.25 | 42.67 | 47.79 | 47.97 | 45.82 | 43.75 |

Interpretation:

- Lowering the LR only for module-declared flat specs did not fix the force gap.
  The `0.5` run briefly looked better at 4k, but by 12k/16k it lagged full TACE
  and finished `2.45 meV/A` worse in force. The `0.25` run was worse still.
- Energy also degraded as the module LR scale decreased: final energy went from
  `112.74` for full TACE to `123.35` at scale `0.5` and `136.86` at scale
  `0.25`.
- Runtime did not improve. Both scaled runs were slightly slower than full TACE
  by wrapper seconds/update and had essentially the same optimizer-step cost.
- This makes per-module LR damping an unlikely standalone fix. The remaining
  HybridMuon path should change routing semantics or scheduling, not just lower
  the CUEQ flat-spec LR. In this no-stage 20k matrix, AdamW is still the best
  force baseline, full TACE is the best broad-HybridMuon force result, and
  linear-only is the best energy result.

Structured outputs used here:

- `runs/oc20neb_fullcase200_ef_20k/676250/matrix_summary.csv`
- `runs/oc20neb_fullcase200_ef_20k/676251/matrix_summary.csv`
- `runs/oc20neb_fullcase200_ef_20k/676250/cueq_hybrid_muon_tace/logs/oc20neb_fullcase200_ef20k_cueq_hybrid_muon_tace_run-456.log`
- `runs/oc20neb_fullcase200_ef_20k/676251/cueq_hybrid_muon_tace/logs/oc20neb_fullcase200_ef20k_cueq_hybrid_muon_tace_run-456.log`


## 2026-07-17 Smooth 50:1 Full-TACE Stage-Two Check

Job `676273` tested whether the broad CUEQ `routing=tace` HybridMuon recipe
benefits from the same delayed energy ramp used in job `676044`. It reused the
OC20NEB fullcase-200 FPS split, single V100, CUEQ, batch size `8`, WSD
per-step scheduling, `hybrid_muon_lr_factor=3.0`, `match_rms`, Magma-lite, full
module include `*`, and module LR scale `1.0`. The loss was `energy:forces =
1:100` until update `15000`, then used the `step_linear` schedule to ramp toward
`50:1` by update `20000`. During stage two,
`MACE_OC20NEB_HYBRID_MUON_STAGE_TWO_ROUTE=adamw` moved the Muon-routed
parameters to AdamW for the high-energy-weight tail.

| Case | Job | Routing / schedule | Final E MAE | Final F MAE | Best F MAE | Seconds/update | Optimizer s/update | Peak FB memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUEQ + AdamW 50:1 ramp | 676044 | AdamW, step-linear | 54.78 | 52.02 | 40.63 | 0.04990 | 0.00085 | n/a |
| CUEQ + HybridMuon radial 50:1 ramp | 676044 | mace/radial, step-linear | 36.51 | 47.84 | 43.65 | 0.05008 | 0.00061 | n/a |
| CUEQ + HybridMuon full TACE no-stage | 676176 | tace/broad, 1:100 | 112.74 | 39.00 | 39.00 | 0.06064 | 0.00859 | 10154 MB |
| CUEQ + HybridMuon full TACE 50:1 ramp | 676273 | tace/broad -> AdamW tail | 25.64 | 45.40 | 43.22 | 0.05786 | 0.00667 | 8894 MB |

Matched-update validation confirms that the run is identical to no-stage full
TACE before the ramp, then trades force for energy after update `15000`:

| Job | Schedule | E/F @4k | E/F @8k | E/F @12k | E/F @16k | E/F @20k |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 676176 | off, 1:100 | 150.43 / 45.11 | 147.33 / 46.73 | 134.14 / 43.54 | 119.67 / 41.09 | 112.74 / 39.00 |
| 676273 | step-linear 1:100 -> 50:1 | 150.43 / 45.11 | 147.33 / 46.73 | 134.14 / 43.54 | 78.77 / 43.22 | 25.64 / 45.40 |

Interpretation:

- The delayed 50:1 ramp is very effective for energy. Full-TACE HybridMuon
  reached the best 20k energy MAE in this matrix (`25.64 meV/atom`), beating the
  radial HybridMuon ramp (`36.51`) and AdamW ramp (`54.78`).
- The same ramp worsens force relative to no-stage full TACE. Final force moved
  from `39.00` to `45.40 meV/A`, and best force moved from `39.00` to
  `43.22 meV/A`.
- Switching Muon-routed parameters to AdamW in stage two reduces optimizer-step
  cost in the tail: the last-1000-step optimizer time was about
  `0.00055 s/update`, close to AdamW and far below no-stage full TACE
  (`0.00858 s/update`). The whole-run average still includes the first 15k Muon
  steps.
- The current smooth stage-two recipe is useful when energy MAE is the primary
  objective, but it is not the force-accuracy answer. For force-focused OC20NEB
  20k, the best measured endpoint remains no-stage AdamW (`38.09 meV/A`), with
  no-stage full TACE HybridMuon close behind (`39.00 meV/A`) but slower.

Structured outputs used here:

- `runs/oc20neb_fullcase200_ef_20k/676273/matrix_summary.csv`
- `runs/oc20neb_fullcase200_ef_20k/676273/cueq_hybrid_muon_tace/logs/oc20neb_fullcase200_ef20k_cueq_hybrid_muon_tace_run-456.log`
