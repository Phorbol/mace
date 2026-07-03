# MACE Training Acceleration Best Practices

This note is the portable usage guide for the DPA4-inspired training acceleration work in this branch. It focuses on the new training-time knobs added here: conservative force backward compile, CUEQ compatibility profiles, TF32/AMP precision controls, HybridMuon routing, and per-step WSD scheduling.

The short version is: start with the conservative recipe, verify parity and validation accuracy on the target machine, then turn on the higher-risk performance knobs one at a time. Do not enable every experimental switch at once.

## What Is Stable Enough To Try First

Recommended first configuration for ordinary small or medium MACE training on a CUDA GPU:

```bash
python -m mace.cli.run_train \
  --name=recio8k_accel_baseline \
  --train_file=/path/to/train.xyz \
  --valid_fraction=0.05 \
  --test_file=/path/to/test.xyz \
  --E0s=average \
  --energy_key=energy \
  --forces_key=forces \
  --model=ScaleShiftMACE \
  --num_interactions=2 \
  --num_channels=64 \
  --max_L=1 \
  --correlation=3 \
  --r_max=5.0 \
  --batch_size=16 \
  --valid_batch_size=32 \
  --max_num_epochs=43 \
  --patience=999 \
  --eval_interval=14 \
  --error_table=PerAtomMAE \
  --default_dtype=float32 \
  --device=cuda \
  --seed=456 \
  --shuffle=False \
  --train_tf32 \
  --train_amp_dtype=none \
  --optimizer=adam \
  --scheduler=ReduceLROnPlateau \
  --lr=0.01 \
  --weight_decay=5e-7 \
  --amsgrad \
  --loss=weighted \
  --energy_weight=1.0 \
  --forces_weight=100.0 \
  --swa \
  --start_swa=32 \
  --swa_energy_weight=1000.0 \
  --swa_forces_weight=100.0 \
  --swa_lr=0.001 \
  --clip_grad=10.0
```

This is not the fastest configuration. It is the reference configuration for checking that the dataset, model size, CUEQ installation, and validation metrics are sane before adding compile or HybridMuon.

## Precision Controls

### `--train_tf32`

Use `--train_tf32` on CUDA unless you have a reason to debug strict FP32. It wraps the training closure with `torch.set_float32_matmul_precision("high")`. On V100 this does not provide TF32 tensor-core acceleration, but it is still harmless as a portable config flag. On Ampere/Hopper-class GPUs it can improve matmul throughput. Disable with `--no-train_tf32` for strict precision diagnostics.

### `--train_amp_dtype`

Allowed values:

- `none`: default and recommended on V100.
- `bf16`: experimental on this branch. It can run on the current V100 stack, but the RECIO/8k 20k-step check showed only a small speed gain and a measurable energy-accuracy regression, so keep it seed-gated rather than default.
- `fp16`: diagnostic only for this branch; not recommended for conservative force training unless separately validated.

Important boundary: the compiled force-loss path runs outside autocast. This is intentional. Geometry-sensitive operations, coordinate/edge derivatives, force construction, and the conservative force-loss graph stay FP32 unless a future segmented AMP path explicitly proves safety. Ordinary eager model forward can use autocast when AMP is enabled.

V100 bf16 note: job `620592` proved that bf16 autocast can execute in `mace_develop`, but job `620602` showed this is not a free default: elapsed improved only from `09:25` to `09:06` versus fp32 full Inductor, while final test energy MAE regressed from `30.6` to `37.8` meV/atom and force MAE moved from `202.6` to `204.7` meV/A.

## Conservative Force Backward Compile

The main new path is `--edge_force_compile`. It compiles a conservative energy-to-force training closure, not a direct-force head. The production endpoint is the edge-gradient path:

```text
edge vectors -> MACE energy -> dE/d(edge vectors) -> atomic forces -> energy/force loss
```

The fuller positions path also exists for diagnostics:

```text
positions -> edge vectors/shifts -> MACE energy -> dE/dpositions -> force loss
```

but current RECIO/8k tests show it is slower, so do not use it as the default training acceleration path.

### Recommended Compile Config: Accuracy-First Smoke

Use this when moving to a new machine, new CUDA/PyTorch version, or new CUEQ version:

```bash
  --edge_force_compile \
  --edge_force_compile_tracing_mode=symbolic \
  --no-edge_force_compile_graph \
  --edge_force_compile_dynamic \
  --edge_force_compile_shape_padding \
  --edge_force_compile_cache_policy=dynamic \
  --no-edge_force_compile_cache_hit_gate \
  --edge_force_compile_min_repeats=2 \
  --edge_force_compile_spherical_harmonics=polynomial \
  --edge_force_compile_force_gradient_mode=edge \
  --edge_force_compile_setup_gate=strict \
  --edge_force_compile_atol=2e-2 \
  --edge_force_compile_rtol=2e-4 \
  --edge_force_compile_parity_check_interval=200 \
  --edge_force_compile_parity_check_gradients \
  --no-edge_force_compile_allow_fallback
```

This FX-only mode is a correctness and cache smoke. It verifies the trace, detach repair, dynamic cache, and same-weight parity without asking Inductor to fuse the graph. It may not be faster than eager.

### Recommended Compile Config: Performance Candidate

After the smoke passes on the target machine, enable Inductor graph lowering:

```bash
  --edge_force_compile \
  --edge_force_compile_tracing_mode=symbolic \
  --edge_force_compile_graph \
  --edge_force_compile_dynamic \
  --edge_force_compile_shape_padding \
  --edge_force_compile_cache_policy=dynamic \
  --no-edge_force_compile_cache_hit_gate \
  --edge_force_compile_min_repeats=2 \
  --edge_force_compile_spherical_harmonics=polynomial \
  --edge_force_compile_force_gradient_mode=edge \
  --edge_force_compile_setup_gate=strict \
  --edge_force_compile_max_fusion_size=8 \
  --edge_force_compile_atol=2e-2 \
  --edge_force_compile_rtol=2e-4 \
  --edge_force_compile_parity_check_interval=1000 \
  --edge_force_compile_parity_check_gradients \
  --edge_force_compile_fixed_probe_interval=1000 \
  --no-edge_force_compile_fixed_probe_gradients \
  --no-edge_force_compile_allow_fallback
```

This is the configuration that should be benchmarked against eager for wall-clock speed and validation accuracy. Keep `--edge_force_compile_setup_gate=strict` for any run used to make an accuracy claim, but keep cache-hit gate disabled for production timing. If setup cost dominates a very short smoke, that is expected. Compile speedup should be judged on a long enough run for the cache to amortize setup.

### Diagnostic-Only Compile Options

Use these only to isolate problems:

- `--edge_force_compile_force_gradient_mode=positions`: fuller positions-to-force graph. It currently works but is slower on RECIO/8k.
- `--edge_force_compile_setup_gate=none`: skips the expensive first-compile reference/candidate gate. Use only after strict evidence exists for the same machine and config.
- `--edge_force_compile_cache_hit_gate`: checks every cache hit. Good for one-off debugging, too expensive for production, and should not be used as a long-run benchmark hot path because repeated full-gradient gates can perturb higher-order autograd/compiler state in a live process.
- `--edge_force_compile_fixed_probe_gradients`: compares fixed-batch parameter gradients. Good for trajectory drift diagnosis, expensive.
- lower `--edge_force_compile_max_fusion_size` such as `4` or `2`: useful when investigating Inductor over-fusion or trajectory drift.

## CUEQ Profiles

CUEQ is controlled independently from compile:

```bash
  --enable_cueq=True \
  --cueq_optimize_all \
  --cueq_optimize_linear \
  --cueq_optimize_channelwise \
  --cueq_optimize_symmetric \
  --cueq_optimize_fctp \
  --cueq_conv_fusion
```

Use full CUEQ only if the installed `cuequivariance` and ops wheels support the GPU architecture. If you see `cudaErrorNoKernelImageForDevice`, the wheel likely does not include kernels for that GPU. That happened with newer CUEQ wheels on V100. In that case, either use a compatible CUEQ version/build, move to a supported GPU, or disable CUEQ for that benchmark.

For force-compile debugging, a safer CUEQ-minus-linear profile is useful:

```bash
  --enable_cueq=True \
  --no-cueq_optimize_all \
  --no-cueq_optimize_linear \
  --cueq_optimize_channelwise \
  --cueq_optimize_symmetric \
  --cueq_optimize_fctp \
  --cueq_conv_fusion
```

Earlier gates isolated optimized CUEQ Linear as a possible source of zero-force compiled graph failures in some configurations. Current full-CUEQ training has also passed in tested environments, so treat this profile as a debugging fallback, not a universal default.

## HybridMuon

HybridMuon is available as:

```bash
  --optimizer=hybrid_muon \
  --hybrid_muon_mode=2d \
  --hybrid_muon_routing=mace \
  --hybrid_muon_lr_factor=0.1 \
  --hybrid_muon_weight_decay=0.0
```

Recommended stable route:

- `--hybrid_muon_routing=mace`
- `--hybrid_muon_mode=2d`
- `--hybrid_muon_lr_factor=0.1`

This routes the radial tensor-product MLP matrices to Muon and keeps sensitive equivariant product/symmetric-contraction parameters on Adam. It is conservative and physically safer.

TACE-style experimental route:

```bash
  --optimizer=hybrid_muon \
  --hybrid_muon_mode=slice \
  --hybrid_muon_routing=tace \
  --hybrid_muon_lr_factor=0.03
```

This lets more rank-3 parameters use per-slice Muon, closer to TACE/DPA4 routing. In the current RECIO/8k full-Inductor test it increased Muon coverage from 8 tensors / 74,752 parameters to 10 tensors / 111,232 parameters by moving the two `symmetric_contractions.weight` tensors onto per-slice Muon. It is compatible with CUEQ and full Inductor, but SAI job `620634` was less accurate than both Adam and conservative MACE routing, so keep it as an ablation rather than a default.

Optional `--hybrid_muon_magma_lite` enables momentum-gradient alignment damping. It is a stability experiment; keep the optimizer, compile, and scheduler fixed when testing it.

## WSD And Per-Step Scheduling

This branch adds per-step scheduler support and WSD knobs. A single-stage WSD recipe is useful for comparing Adam and HybridMuon without mixing in Stage Two/SWA effects:

```bash
  --scheduler=WSD \
  --lr_scheduler_interval=step \
  --lr_wsd_warmup_steps=0 \
  --lr_wsd_warmup_ratio=0.03 \
  --lr_wsd_warmup_start_factor=0.1 \
  --lr_wsd_stop_lr_ratio=1e-3 \
  --lr_wsd_decay_phase_ratio=0.1 \
  --lr_wsd_decay_type=inverse_linear
```

Use this when the goal is an optimizer comparison. Use the standard Stage Two recipe when the goal is reproducing normal MACE training accuracy. Do not compare single-stage WSD HybridMuon against two-stage Adam and attribute the difference only to the optimizer.

## SAI Wrapper Usage

For SAI runs, the branch provides:

```bash
scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

Typical conservative smoke:

```bash
env MACE_CONDA_ENV=mace_develop \
  RUN_ROOT=runs/recio8k_compile_smoke \
  NAME=recio8k_compile_smoke \
  TRAIN_FILE=/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz \
  MAX_NUM_EPOCHS=1 \
  BATCH_SIZE=16 \
  EVAL_INTERVAL=1 \
  ENABLE_CUEQ=False \
  EDGE_FORCE_COMPILE=True \
  EDGE_FORCE_GRAPH=False \
  EDGE_FORCE_GRADIENT_MODE=edge \
  TRAIN_TF32=True \
  TRAIN_AMP_DTYPE=none \
  sbatch scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

Performance candidate:

```bash
env MACE_CONDA_ENV=mace_develop \
  RUN_ROOT=runs/recio8k_compile_perf \
  NAME=recio8k_compile_perf \
  MAX_NUM_EPOCHS=43 \
  BATCH_SIZE=16 \
  EVAL_INTERVAL=14 \
  OPTIMIZER=adam \
  ENABLE_CUEQ=True \
  CUEQ_PROFILE=full \
  EDGE_FORCE_COMPILE=True \
  EDGE_FORCE_GRAPH=True \
  EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True \
  EDGE_FORCE_GRADIENT_MODE=edge \
  EDGE_FORCE_PARITY_CHECK_INTERVAL=1000 \
  EDGE_FORCE_FIXED_PROBE_INTERVAL=1000 \
  TRAIN_TF32=True \
  TRAIN_AMP_DTYPE=none \
  sbatch scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

`EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True` is deliberately required for `EDGE_FORCE_GRAPH=True` in the wrapper so that accidental Inductor benchmark runs do not happen silently.

## Recommended Ablation Order

Use this order when moving to another machine:

1. Adam, no compile, CUEQ off or known-good CUEQ. Check baseline validation/test metrics.
2. Adam, CUEQ on. Confirm speed and accuracy are sane.
3. Adam, CUEQ on, `edge_force_compile` FX-only. Confirm compile gate/cache/parity.
4. Adam, CUEQ on, `edge_force_compile_graph=True`. Check speed, MaxRSS, validation/test metrics.
5. Conservative HybridMuon: `routing=mace`, `mode=2d`, same compile/CUEQ setting as the accepted Adam run.
6. TACE-style HybridMuon: `routing=tace`, `mode=slice`, lower `hybrid_muon_lr_factor` such as `0.03` or `0.05`.
7. Native-bf16 AMP only on a bf16-capable GPU, with parity and validation checks.

Keep one variable changed per run. For speed claims, report hot-step timing and end-to-end wall time separately. For accuracy claims, report both validation and held-out test energy/force errors, not just training loss.

## How To Interpret Logs

A healthy compile epoch summary looks like:

```text
Edge-force compile epoch ... summary: steps=..., compiled=..., cache_hits=...,
new_compiles=1, fallbacks=0, runtime_recompiles=0, ... parity=..., fixed_probe=...
```

Red flags:

- `fallbacks > 0` when `--no-edge_force_compile_allow_fallback` was expected.
- repeated `new_compiles` under `cache_policy=dynamic`; this means some input shape is still concrete in the cache key.
- parity failures with large force or parameter-gradient differences.
- MaxRSS increases that exceed available GPU memory at the target model size.
- validation force improves but test force degrades systematically across seeds; treat this as generalization risk, not success.

## Current Evidence Boundary

Current RECIO/8k evidence supports these cautious claims:

- Dynamic edge-force compile can cache across real RECIO batches.
- SAI job `620035` showed that repeated live full-gradient setup gates can fail on the second real batch even with `state_dict`, module training flags, CPU RNG, and CUDA RNG unchanged; treat this as a gate-harness non-reentrancy diagnostic, not as the production hot training path.
- SAI job `620059` completed a real RECIO/8k one-epoch smoke with HybridMuon, full CUEQ, dynamic FX edge-force compile, and cache-hit gate disabled: `compiled=475`, `cache_hits=474`, `new_compiles=1`, `fallbacks=0`, MaxRSS `3620308K`.
- SAI jobs `620074`-`620081` completed the real RECIO/8k 20k-update ablation at batch size 16. End-to-end elapsed times were Adam/no-CUEQ eager `13:35`, Adam/no-CUEQ compile `14:16`, Adam/CUEQ eager `15:19`, Adam/CUEQ compile `11:39`, Muon/no-CUEQ eager `14:42`, Muon/no-CUEQ compile `13:56`, Muon/CUEQ eager `11:57`, and Muon/CUEQ compile `11:46`. Treat this as evidence that the FX-only compile path is production-runnable, but not yet DPA4-level 3x faster.
- Full Inductor graph lowering is now the stronger CUEQ performance candidate: SAI jobs `620575` and `620576` completed the same 20k RECIO/8k CUEQ runs with full Inductor. Adam/CUEQ full Inductor finished in `09:25` with test `30.6` meV/atom and `202.6` meV/A, versus FX-only `11:39` and eager `15:19`; Muon/CUEQ full Inductor finished in `11:17` with test `36.1` meV/atom and `223.2` meV/A, versus FX-only `11:46` and eager `11:57`.
- Model-size smoke checks also passed with full Inductor and full CUEQ: SAI jobs `620760`, `620761`, and `620762` completed short RECIO/8k runs for `max_L=0`, `num_channels=128`, and `max_L=2`, respectively, with `fallbacks=0`. Hot epochs were about `3.6-4.0s`, `6.0-6.5s`, and `10.9-12.0s`.
- Longer model-size checks support the same direction: SAI job `620775` (`num_channels=128`) completed 20k updates in `09:29`, MaxRSS `5207212K`, test `25.9` meV/atom / `181.6` meV/A; SAI job `620776` (`max_L=2`) completed in `15:44`, MaxRSS `6168444K`, test `27.3` meV/atom / `188.2` meV/A. Both used Adam, full CUEQ, full Inductor, TF32, and AMP none, with no fallbacks.
- Matching CUEQ-eager controls finished slower with comparable accuracy: SAI job `620827` (`num_channels=128`) finished in `11:47`, MaxRSS `3652688K`, test `27.4` meV/atom / `180.9` meV/A; SAI job `620828` (`max_L=2`) finished in `18:01`, MaxRSS `4316052K`, test `34.9` meV/atom / `189.5` meV/A. This supports full Inductor as a speed path for these model sizes, while also showing that Inductor increases peak memory on the current V100 stack.
- The model-size 20k numbers above use the RECIO/8k training source as the training/test-file source, so they are training-throughput plus in-distribution holdout evidence rather than a clean cross-sampling generalization claim. As an independent labeled RECIO check, the saved full-Inductor+CUEQ Adam models were evaluated on `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/random_8k.xyz` with CUEQ inference; a dedicated extxyz scorer was needed because the current ASE high-level `read()` path drops this file's `energy/forces` fields. The resulting MAE was `21.38` meV/atom / `131.75` meV/A for `num_channels=128` and `22.30` meV/atom / `132.80` meV/A for `max_L=2`, with relative force MAE about `13.94%` and `14.05%`, respectively.
- The longer 200k-step RECIO/8k matrix is the best amortization evidence for the current V100/CUEQ stack. With full CUEQ `optimize_all`, TF32, AMP none, and Stage Two enabled, Adam compile improved wall time from `01:27:34` to `01:23:49` while keeping stage-two test force essentially unchanged (`137.2` -> `136.7` meV/A); radial-only HybridMuon compile improved wall time from `01:35:17` to `01:26:07`, but its stage-two test force stayed worse than Adam (`147.4` meV/A). Treat this as stable compatibility and modest end-to-end speedup, not as a DPA4-level 3x result.
- Per-step WSD and `--single_stage` are implemented and should be used for fair optimizer ablations. The 20k per-step WSD gate showed that removing Stage Two improved radial-only HybridMuon force (`232.5` -> `222.9` meV/A), but single-stage Adam was still better (`211.4` meV/A). WSD is therefore a necessary fairness control, not a sufficient HybridMuon fix.
- Fixed-probe and three-seed trajectory gates show the current full-Inductor edge-force path has stable cache behavior and real hot-step speedup, but accuracy should remain seed-gated. Across seeds `456`, `123`, and `789`, eager full-CUEQ Adam had median hot steps `23.86 +/- 2.34 ms`; compile had `12.90 +/- 0.48 ms` (`1.85x` median-step speedup), no fallbacks, and no runtime recompiles. Test energy means were close, while test force was modestly worse for compile in the small sample; do not enable compile as a universal default until the target recipe has a larger repeat/seed table.
- The fuller `force_gradient_mode=positions` path now works with RECIO/8k, CUEQ, TF32, and HybridMuon, but it is slower than the edge-gradient production path. Use positions mode as a correctness/cache diagnostic, not as the recommended training acceleration mode.
- BF16 remains experimental on V100: SAI job `620602` completed Adam/CUEQ full Inductor bf16 in `09:06`, but test energy MAE regressed to `37.8` meV/atom and force MAE to `204.7` meV/A.
- Inductor edge-force compile can accelerate hot training steps, especially when the graph lowering path is accepted.
- Compile plus CUEQ does not automatically give DPA4-level 3x end-to-end speedup because MACE still has more work outside the compiled closure and CUEQ custom kernels hide tensor-product internals from Inductor.
- Precision preservation is promising but not proven as a universal default; use multi-seed or longer validation before making production claims.
- HybridMuon is compatible with the compile path, but the current RECIO/8k evidence favors Adam for this model size. Conservative MACE-routed HybridMuon is stable but slower and less accurate than Adam in the full-Inductor run. TACE-style slice routing is compatible and faster than conservative Muon in full Inductor, but job `620634` finished in `09:46` with MaxRSS `5152156K` and test `40.4` meV/atom / `261.2` meV/A, worse than Adam full Inductor (`09:25`, `30.6` / `202.6`) and worse in force than conservative MACE-Muon (`11:17`, `36.1` / `223.2`). Keep TACE-style routing as an explicit ablation; do not make it the default without a changed scheduler/lr recipe and multi-seed evidence.

- Larger-model and cross-dataset gates sharpen the current boundary. RECIO C128/L2 200k WSD showed a clear end-to-end compile gain (`02:25:08` eager -> `01:34:47` compile) with matching or slightly better test E/F (`49.9/114.8` -> `47.7/114.4`), but higher MaxRSS (`3.78 GB` -> `6.39 GB`). OC20NEB FPS L1/C64 showed compatibility and no displayed precision loss (`31.0` meV/atom / `54.1` meV/A for both eager and compile), but only `1.13x` wall-clock speedup and higher memory (`5.80 GB` -> `7.72 GB`).
- The RECIO L1/C64 200k WSD compile-only gate, SAI job `621462`, completed in `00:53:22` with MaxRSS `4.95 GB`. It used Adam, full CUEQ, full Inductor edge-force compile, TF32 enabled, and AMP none. Compile health was clean over all `211` epochs: one initial compile, `fallbacks=0`, `runtime_recompiles=0`, hot-step mean `12.77 ms/batch`, max energy/force parity difference `2.289e-5`, and max parameter-gradient difference `3.967e-4`. Final metrics were train `53.7` meV/atom / `155.2` meV/A, valid `48.9` / `226.6`, and test `53.5` / `158.8`. `nvdmon` showed only about `22%` average SM utilization and about `2.51 GB` framebuffer use, so the current bottleneck is not compile cache churn. This supports the interpretation that L1/C64 is too small to expose DPA4-like large-graph speedups on V100; use larger channels/L or larger batches when judging force-backward compile as a throughput feature.

### OC20NEB FPS Benchmark Wrapper

For the DeepMD DPA4 OC20NEB FPS split, first convert the DeepMD mixed dataset to MACE extxyz:

```bash
python scripts/benchmarks/oc20neb_fps/convert_deepmd_mixed_to_extxyz.py \
  --outdir runs/oc20neb_fps_extxyz \
  --overwrite
```

The converter reads `real_atom_types.npy`; do not use `type.raw` for this benchmark because it is only a placeholder. The wrapper `scripts/benchmarks/oc20neb_fps/run_mace_oc20neb_fps_sai.sh` defaults to `L_max=1`, `num_channels=64`, Adam, WSD, full CUEQ, TF32, AMP none, `r_max=6.0`, and separate `train.extxyz`/`valid.extxyz` files. On SAI, prefer environment-prefix submission rather than explicit `sbatch --export=ALL,...`, which cancelled immediately in the observed environment:

```bash
RUN_ROOT=runs/oc20neb_fps_l1c64_wsd_20k/adam_cueq_compile \
NAME=oc20neb_fps_l1c64_wsd_adam_cueq_compile \
EDGE_FORCE_COMPILE=True EDGE_FORCE_GRAPH=True EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True \
ENABLE_CUEQ=True CUEQ_PROFILE=full TRAIN_TF32=True TRAIN_AMP_DTYPE=none \
OPTIMIZER=adam SCHEDULER=WSD MAX_NUM_EPOCHS=32 BATCH_SIZE=8 VALID_BATCH_SIZE=8 \
sbatch scripts/benchmarks/oc20neb_fps/run_mace_oc20neb_fps_sai.sh
```
