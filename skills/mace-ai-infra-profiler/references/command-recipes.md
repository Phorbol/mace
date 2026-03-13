# Command Recipes

## 1. End-To-End Infra Pipeline

Use for baseline vs optimized comparison with gate, parity, and diagnostics in one run.

```powershell
py -3 scripts/infra_speed_pipeline.py `
  --structures D:\path\TiO2.cif D:\path\AlN.cif `
  --sizes 2 3 4 `
  --device cuda `
  --compile-mode default `
  --optimized-amp bf16 `
  --optimized-tf32 `
  --compile-bucket-atoms 512 `
  --compile-bucket-edges 0 `
  --threshold-pct 5 `
  --gate-modes compile `
  --run-id run_20260313 `
  --out-dir scripts/profile_outputs/infra_pipeline
```

Primary outputs:
- `<run-id>_baseline.csv/.json`
- `<run-id>_optimized.csv/.json`
- `<run-id>_summary.json/.md`
- optional parity and diagnostics artifacts

## 2. Baseline Or Candidate Benchmark Matrix

Use for controlled micro-benchmarking across structures, sizes, and eager or compile modes.

Baseline style:
```powershell
py -3 scripts/profile_no_cueq_real_structures.py `
  --structures D:\path\TiO2.cif D:\path\AlN.cif `
  --sizes 2 3 4 `
  --device cuda `
  --amp none `
  --out-json scripts/profile_outputs/base.json `
  --out-csv scripts/profile_outputs/base.csv
```

Optimized style:
```powershell
py -3 scripts/profile_no_cueq_real_structures.py `
  --structures D:\path\TiO2.cif D:\path\AlN.cif `
  --sizes 2 3 4 `
  --device cuda `
  --amp bf16 `
  --tf32 `
  --compile-bucket-atoms 512 `
  --compile-bucket-edges 0 `
  --out-json scripts/profile_outputs/opt.json `
  --out-csv scripts/profile_outputs/opt.csv
```

## 3. Perf Regression Gate

Use when baseline and candidate CSV already exist.

```powershell
py -3 scripts/perf_gate.py `
  --baseline-csv scripts/profile_outputs/base.csv `
  --candidate-csv scripts/profile_outputs/opt.csv `
  --threshold-pct 5 `
  --modes compile
```

## 4. Numeric Parity Check

Use before accepting aggressive optimization knobs.

```powershell
py -3 scripts/check_numeric_parity.py `
  --structures D:\path\TiO2.cif D:\path\AlN.cif `
  --sizes 2 3 4 `
  --device cuda `
  --amp bf16 `
  --tf32 `
  --energy-abs-tol 2e-1 `
  --energy-rel-tol 3e-2 `
  --force-abs-tol 5e-3 `
  --force-rel-tol 5e-2 `
  --out-json scripts/profile_outputs/parity.json `
  --out-csv scripts/profile_outputs/parity.csv
```

## 5. Torch Operator Diagnostics

Use to rank operator hotspots and classify bottleneck tendency.

```powershell
py -3 scripts/profile_torch_ops_diagnostics.py `
  --structure D:\path\TiO2.cif `
  --size 4 `
  --mode compile `
  --task infer `
  --compute-force `
  --device cuda `
  --amp bf16 `
  --tf32 `
  --top-k 30 `
  --out-json scripts/profile_outputs/ops_tio2_compile_infer.json `
  --out-csv scripts/profile_outputs/ops_tio2_compile_infer.csv
```

Read `intensity_hint` in output (`io_bound_likely`, `compute_bound_likely`, `mixed`) to choose optimization direction.

## 6. CUDA Timing + NVTX Diagnostic

Use to generate CUDA-event metrics and NVTX ranges for timeline tools.

```powershell
py -3 scripts/profile_cuda_diagnostics.py `
  --structure D:\path\TiO2.cif `
  --size 4 `
  --mode compile `
  --task train `
  --iters 8 `
  --warmup 2 `
  --out-json scripts/profile_outputs/diag_tio2_compile_train.json
```

## 7. Nsight Systems Capture (Optional)

Use when kernel-level timeline analysis is required and `nsys` is installed.

```powershell
nsys profile -o scripts/profile_outputs/nsys/tio2_compile_train `
  --force-overwrite=true `
  -t cuda,nvtx,osrt,cublas,cudnn `
  --sample=none `
  py -3 scripts/profile_cuda_diagnostics.py `
    --structure D:\path\TiO2.cif `
    --size 4 `
    --mode compile `
    --task train `
    --iters 8 `
    --warmup 2 `
    --out-json scripts/profile_outputs/diag_tio2_compile_train.json
```

Optional stats extraction:

```powershell
nsys stats --report cuda_gpu_kern_sum --format csv `
  --output scripts/profile_outputs/nsys/tio2_compile_train_kern `
  scripts/profile_outputs/nsys/tio2_compile_train.nsys-rep
```

## 8. Fast Triage Heuristics

1. Start with `infra_speed_pipeline.py` to avoid manual orchestration mistakes.
2. If gate fails, inspect rows with highest regression, then run operator diagnostics on those cases.
3. If operator mix is compute-heavy, test AMP or TF32 first.
4. If operator mix is IO-heavy, test compile bucket settings and force-computation scope first.
5. Re-run parity after any change that alters math precision or force path.
