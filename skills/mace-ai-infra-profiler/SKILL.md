---
name: mace-ai-infra-profiler
description: Profile and optimize MACE training and inference AI infra in this repository. Use when requests mention profiling, operator diagnosis, latency or memory regressions, baseline versus optimized comparisons, perf gating, numeric parity checks, CUDA or NVTX timelines, Nsight Systems capture, AMP or TF32 tuning, torch.compile tuning, or compile bucket tuning for structure-based workloads.
---

# MACE AI Infra Profiler

## Overview

Use repository scripts in `scripts/` to benchmark train and infer performance, locate operator hotspots, validate numerical safety, and gate regressions before and after infra changes.
Use [references/command-recipes.md](references/command-recipes.md) for exact command templates.

## Preconditions

1. Run from repo root: `D:\Download\trae-research-code\mace`.
2. Ensure Python environment can import local `mace` package and dependencies (`torch`, `ase`, `e3nn`).
3. Pass absolute structure paths to scripts (`--structure` or `--structures`), because this repo does not ship CIF samples.
4. Prefer CUDA device for meaningful infra profiling; many scripts raise when CUDA is unavailable.

## Workflow Selection

1. Use `scripts/infra_speed_pipeline.py` for end-to-end baseline vs optimized run, perf gate, optional parity, and optional operator diagnostics.
2. Use `scripts/profile_no_cueq_real_structures.py` for pure latency or memory benchmarking matrix without gate orchestration.
3. Use `scripts/profile_torch_ops_diagnostics.py` for top-k torch operator diagnosis and intensity hint (`io_bound_likely`, `compute_bound_likely`, `mixed`).
4. Use `scripts/profile_cuda_diagnostics.py` when CUDA event timing and NVTX ranges are needed for Nsight Systems timelines.
5. Use `scripts/check_numeric_parity.py` to validate AMP or TF32 numerical risk on energy and force metrics.
6. Use `scripts/perf_gate.py` when baseline and candidate CSV already exist and only regression gating is required.

## Standard Optimization Loop

1. Run baseline profiling with conservative settings (`amp=none`, `tf32=false`).
2. Run candidate profiling with infra knobs (`amp`, `tf32`, `compile_bucket_atoms`, `compile_bucket_edges`).
3. Run regression gate with project threshold.
4. Run parity check before accepting speedups.
5. Run operator diagnostics on representative or full case set to identify bottlenecks.
6. Propose or implement code changes based on hotspot type.
7. Re-run the same matrix and compare median speedups plus gate status.

## Hotspot-To-Action Mapping

1. Treat `io_bound_likely` (scatter, index, copy heavy) as memory traffic pressure:
   reduce edge or atom pressure, use compile bucket reuse, avoid unnecessary `compute_force=True`, and check data movement patterns.
2. Treat `compute_bound_likely` (matmul or gemm heavy) as arithmetic pressure:
   test `--amp bf16` or `--amp fp16`, `--tf32`, and `--mode compile`; verify parity afterward.
3. Treat `mixed` as pipeline bottleneck:
   split diagnosis between operator mix (`profile_torch_ops_diagnostics.py`) and timeline (`profile_cuda_diagnostics.py` + NSYS).
4. If compile infer with force fails, follow script fallback behavior (energy-only path) and record the fallback note in report.

## Reporting Contract

1. Report exact commands and key arguments used.
2. Report generated artifact paths (`*.json`, `*.csv`, optional `*.md`, optional `*.nsys-rep`).
3. Report baseline vs candidate latency deltas for train and infer.
4. Report perf gate result and threshold.
5. Report parity result and tolerance set.
6. Report top operator mix shares (`io`, `compute`, `mixed`) for each diagnosed case.
7. If code changes are made, report modified files and expected effect on bottlenecks.

## References

1. Use [references/command-recipes.md](references/command-recipes.md) for command templates and argument presets.
