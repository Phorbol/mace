# RECIO/8k Training Acceleration Benchmark

Generate isolated short-run cases:

```bash
python scripts/benchmarks/recio8k_accel/generate_cases.py \
  --output /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke \
  --epochs 20
```

Submit one case from a clean login-node shell:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke/fp32_hybrid_muon_cueq
sbatch mace-recio8k.sbatch
```

Parse completed or running logs:

```bash
python scripts/benchmarks/recio8k_accel/parse_metrics.py \
  /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-smoke
```

The parser reports validation MAE, timestamp-derived seconds per epoch, and `nvidia-smi dmon` summaries when a case-local `nvdmon_job-*.log` file is present.

The default template uses the existing RECIO `4V100` partition and exports this repository on `PYTHONPATH` before calling `python -m mace.cli.run_train`, so the Slurm job uses the checked-out develop code while preserving case-local relative paths. Generated cases set `distributed: false` because the template is a single-rank, single-GPU smoke run. V100 does not provide native bf16 tensor cores, so `bf16_*` cases should be submitted only on a bf16-capable GPU partition. On `4V100`, use `baseline_fp32_adam_cueq`, `fp32_hybrid_muon_cueq`, and `compile_fp32_adam_cueq` as the first smoke comparison.

The historical full baseline final stage-two validation is about `22.1 meV/atom` energy and `260.2 meV/A` force. Smoke runs are for finite-loss, speed, memory, and trend checks; long runs are required before claiming no accuracy or generalization regression.

## Current RECIO/8k Smoke Evidence

Successful 20 epoch SAI smoke runs on `4V100`, single GPU, `mace_env`:

| Case | Slurm job | Wall time | Stage-two valid MAE E | Stage-two valid MAE F | Late epoch mean step time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_fp32_adam_cueq` | `576258` | `8:10` | `49.4 meV/atom` | `307.9 meV/A` | `0.0676 s` |
| `fp32_hybrid_muon_cueq` | `576259` | `8:13` | `44.3 meV/atom` | `289.9 meV/A` | `0.0676 s` |
| `compile_fp32_adam_cueq` | `576338` | `9:03` | `45.8 meV/atom` | `300.6 meV/A` | `0.0713 s` |

HybridMuon is numerically promising in this short run, but this is not enough to claim final accuracy or generalization. It needs a full RECIO/8k schedule before acceptance.

`train_compile` is correctness-safe but not a speedup for the current force-loss MACELES path on this PyTorch/V100 stack. The energy-only compile wrapper starts correctly, then the first force-loss backward hits PyTorch AOTAutograd's double-backward limitation and falls back to eager with this expected warning:

```text
training torch.compile failed during backward; disabling compiled training model and retrying eager: torch.compile with aot_autograd does not currently support double backward
```

This fallback is intentional: it preserves MACE's conservative-force training instead of hiding a broken compiled gradient path. Future compile acceleration should target lower-level force-safe kernels or submodules with proven higher-order-gradient support, not the whole MACE force-training graph.

## Full RECIO/8k Validation Runs

Full 800 epoch single-GPU validation jobs were submitted on SAI `4V100` with the same random seed and split:

| Case | Slurm job | Status at submit check | Early validation |
| --- | ---: | --- | --- |
| `baseline_fp32_adam_cueq` | `576362` | running on `4v100n35` | epoch 0: `302.25 meV/atom`, `600.83 meV/A` |
| `fp32_hybrid_muon_cueq` | `576363` | running on `4v100n33` | epoch 0: `261.64 meV/atom`, `567.72 meV/A` |

The generated case root is:

```text
/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-full-20260629-040425
```

The visible SAI GPU partitions at submission time were V100-only (`4V100`, `4V100PX`, `8V100V0`), so no bf16 validation job was submitted. The bf16 path should be validated on A100/H100 or another native bf16 GPU; running it on V100 would not test the DPA4-style bf16 acceleration path.

## GPU D3 Backend Status

`mace_mp(..., dispersion=True, dispersion_backend="nvalchemi")` now routes D3 dispersion through an optional `NvalchemiDFTD3Calculator` ASE adapter. The adapter keeps the dispersion correction outside the MACE neural model and sums it at the calculator level, matching the existing `torch_dftd` architecture and preserving MACE model semantics.

The first mapped backend is deliberately narrow: PBE-D3(BJ), using the explicit Grimme parameters `a1=0.4289`, `a2=4.4407` Bohr, and `s8=0.7875`, because nvalchemi's `DFTD3ModelWrapper` takes explicit BJ damping parameters rather than an XC string. Other XC/damping combinations still require `dispersion_backend="torch_dftd"` until their parameters are mapped and verified against a reference.

`mace_env` has been live-tested with `nvalchemi-toolkit==0.1.0`, `nvalchemi-toolkit-ops==0.3.1`, `warp-lang==1.14.0`, and PyTorch `2.8.0+cu128`.

A SAI `4V100` CUDA smoke run completed successfully:

| Check | Value |
| --- | ---: |
| Slurm job | `576417` |
| Node | `4v100n31` |
| Exit state | `COMPLETED`, `0:0` |
| CPU D3 energy | `-0.009781921282 eV` |
| CUDA D3 energy | `-0.009781923145 eV` |
| CPU/CUDA energy absolute difference | `1.862645e-09 eV` |
| CPU/CUDA max force absolute difference | `4.001777e-11 eV/A` |
| CUDA availability in job | `True` |

The smoke script is `scripts/benchmarks/nvalchemi_d3_smoke/run_nvalchemi_d3_smoke.py`; submit it with `scripts/benchmarks/nvalchemi_d3_smoke/nvalchemi-d3-smoke.sbatch`. This validates the optional CUDA backend on a small ASE molecule. A larger periodic RECIO-like timing comparison against `torch_dftd` is still the next gate before claiming production speedup from GPU D3.

