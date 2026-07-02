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
- `bf16`: use only on native-bf16 GPUs after parity and validation checks.
- `fp16`: diagnostic only for this branch; not recommended for conservative force training unless separately validated.

Important boundary: the compiled force-loss path runs outside autocast. This is intentional. Geometry-sensitive operations, coordinate/edge derivatives, force construction, and the conservative force-loss graph stay FP32 unless a future segmented AMP path explicitly proves safety. Ordinary eager model forward can use autocast when AMP is enabled.

On V100, `bf16` is expected to fail closed through `torch.cuda.is_bf16_supported()`. Do not interpret that as a MACE bug.

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

This is the configuration that should be benchmarked against eager for wall-clock speed and validation accuracy. Keep `--edge_force_compile_setup_gate=strict` for any run used to make an accuracy claim. If setup cost dominates a very short smoke, that is expected. Compile speedup should be judged on a long enough run for the cache to amortize setup.

### Diagnostic-Only Compile Options

Use these only to isolate problems:

- `--edge_force_compile_force_gradient_mode=positions`: fuller positions-to-force graph. It currently works but is slower on RECIO/8k.
- `--edge_force_compile_setup_gate=none`: skips the expensive first-compile reference/candidate gate. Use only after strict evidence exists for the same machine and config.
- `--edge_force_compile_cache_hit_gate`: checks every cache hit. Good for debugging, too expensive for production.
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

This lets more rank-3 parameters use per-slice Muon, closer to TACE/DPA4 routing. It may improve optimization but must be treated as an ablation. Check route logs at startup and compare validation/test energy and force metrics against Adam and conservative MACE routing. Do not use it as the first run on a new dataset.

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
- Full positions-gradient compile now works as a diagnostic but is slower.
- Inductor edge-force compile can accelerate hot training steps, especially when the graph lowering path is accepted.
- Compile plus CUEQ does not automatically give DPA4-level 3x end-to-end speedup because MACE still has more work outside the compiled closure and CUEQ custom kernels hide tensor-product internals from Inductor.
- Precision preservation is promising but not proven as a universal default; use multi-seed or longer validation before making production claims.
- HybridMuon is compatible with the compile path, but broader TACE-style routing still needs dataset-level ablation before being considered a default.
