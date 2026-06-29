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

The parser reports validation MAE, timestamp-derived seconds per epoch, `nvidia-smi dmon` summaries when a case-local `nvdmon_job-*.log` file is present, and `train_compile_fallback` fields when a compiled training run disables compile and retries eager.

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

`train_compile` is correctness-safe but not a speedup for the current force-loss MACELES path on this PyTorch/V100 stack. The wrapper now uses the compiled model for `compute_force=False` energy-only training, where ordinary first-order parameter backpropagation does not require differentiating forces. For the RECIO/MACELES force-loss path, the energy-only forward compile starts correctly, then the first force-loss backward hits PyTorch AOTAutograd's double-backward limitation and falls back to eager with this expected warning:

```text
training torch.compile failed during backward; disabling compiled training model and retrying eager: torch.compile with aot_autograd does not currently support double backward
```

This fallback is intentional: it preserves MACE's conservative-force training instead of hiding a broken compiled gradient path. Current accepted compile boundaries are: compiled energy-only training for `compute_force=False`, eager fallback for conservative force training when compiled higher-order autograd fails, and future lower-level force-safe kernels or submodules with proven higher-order-gradient support.

## DeepMD DPA4 Reference Boundaries

The local DeepMD-kit reference checkout is `/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/reference_repos/deepmd-kit` at commit `a9bcbc50`. The relevant DPA4/SeZM pieces support the current MACE design choices:

- `doc/model/dpa4.md` documents DPA4 as a conservative energy model where forces come from differentiating energy. This matches MACE's conservative-force principle and rules out direct-force shortcuts unless they are explicitly separate heads with separate validation.
- `deepmd/pt/model/descriptor/sezm.py` applies optional CUDA bfloat16 autocast only in training forward regions when `use_amp=True`, and recommends it only on GPUs with native bf16 support. MACE's `--train_amp_dtype bf16` follows the same fail-closed rule on non-bf16 hardware.
- `deepmd/pt/utils/compile_compat.py` shows that DPA4 compile support relies on dedicated make_fx/AOTInductor plumbing, second-order-autograd graph repair, trace-shape control, and Inductor workarounds. It is not equivalent to wrapping the whole training model with `torch.compile`.
- DPA4's `.pt2`/AOTInductor path is primarily an export/inference artifact; its Triton SO(2) kernels keep full float32 accumulation. For MACE, the analogous production path should be kernel/submodule level acceleration for known hot spots, not replacing equivariant tensor-product training semantics.

Current MACE implementation therefore treats DPA4-inspired bf16 and compile as conservative opt-in paths: AMP is allowed only where hardware supports it, while training compile preserves correctness through eager fallback until force-safe compiled subgraphs are proven.

## Force-Loss Compile Design

MACE force training must keep forces as `-dE/dR` and must let the force loss backpropagate through that gradient to model parameters. In code this is the `get_outputs(..., training=True)` path, where `torch.autograd.grad(..., create_graph=True)` is required. A compiled path that drops this second-order gradient would train a different objective, even if the forward forces look numerically close for one batch.

The accepted compile roadmap is therefore staged:

1. Keep whole-model training compile as an opt-in wrapper with explicit eager fallback for conservative force losses. The RECIO parser now reports `train_compile_fallback` so benchmark summaries cannot accidentally count fallback-eager runs as compile speedups.
2. Add a dedicated force-loss micro-benchmark/profiler that runs the same RECIO batch in eager, energy-only compile, and force-loss compile modes, then records compile status, timing, memory, and the fallback reason. This should be the next code step before changing model internals.
3. Only compile subgraphs whose outputs remain differentiable through the force-loss second-order path. Candidate regions are pure tensor compute blocks such as radial/readout MLPs or future cueq-backed tensor-product kernels, not the `autograd.grad` force construction itself unless an FX/AOT path proves second-order correctness.
4. Gate any subgraph compile change with unit tests comparing energy, forces, and parameter gradients against eager on a small batch, followed by a RECIO/8k sbatch smoke with unchanged validation trend. A speedup without these equivalence checks is not acceptable for MACE because it may silently violate the conservative-force training objective.

This differs from copying DeepMD DPA4 directly: DeepMD relies on model-specific make_fx/AOTInductor plumbing, shape control, detach repair, and Inductor patches for its second-order graph. MACE should borrow the principle, not the implementation, and should keep compatibility with cueq by placing compile/kernel boundaries around established equivariant operations rather than replacing MACE architecture.


The current probe implementation is `scripts/benchmarks/recio8k_accel/probe_training_compile.py`. A minimal CPU smoke is:

```bash
python scripts/benchmarks/recio8k_accel/probe_training_compile.py \
  --device cpu --indices 0 --modes eager_force_loss \
  --hidden-channels 8 --max-ell 1 --num-interactions 1 --correlation 1 \
  --warmup 0 --repeats 1
```

On SAI, submit the CUDA/cueq probe directly:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/compile-probe
sbatch /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/training-compile-probe.sbatch
```

The sbatch writes `training_compile_probe_${SLURM_JOB_ID}.json` and `nvdmon_job-${SLURM_JOB_ID}.log`. It defaults to RECIO `train.xyz` index `0`, `mace_env`, `4V100`, `improper-gpu`, cueq enabled, and the force-loss equivalence gate enabled; override with environment variables such as `RECIO_INDICES=0,4,8`, `MACE_PROBE_REPEATS=10`, `MACE_PROBE_ENABLE_CUEQ=0`, or `MACE_PROBE_EQUIVALENCE_GATE=0`.

Initial SAI probe results:

| Job | RECIO indices | Atoms | Mode | Compile disabled | Mean seconds/step | Max CUDA memory |
| ---: | --- | ---: | --- | --- | ---: | ---: |
| `576776` | `0` | 9 | `eager_force_loss` | no | `2.750e-02` | `21.3 MB` |
| `576776` | `0` | 9 | `compile_force_loss` | yes | `1.149e-02` | `24.3 MB` |
| `576776` | `0` | 9 | `compile_energy_only` | no | `3.388e-03` | `22.7 MB` |
| `576782` | `0,4,8` | 22 | `eager_force_loss` | no | `1.968e-02` | `24.2 MB` |
| `576782` | `0,4,8` | 22 | `compile_force_loss` | yes | `1.169e-02` | `25.3 MB` |
| `576782` | `0,4,8` | 22 | `compile_energy_only` | no | `3.215e-03` | `22.8 MB` |

Both probe jobs completed with Slurm `COMPLETED` / `ExitCode 0:0` on `4V100` with cueq enabled. The force-loss compile mode logged the expected AOTAutograd double-backward fallback and then timed the eager fallback path; it must not be interpreted as compiled force-loss acceleration. The energy-only mode kept compile enabled, which confirms the current compile boundary works when training does not need conservative-force second derivatives.

A local CPU smoke of `--equivalence-gate` on RECIO index `0` returned `ok: true` with zero energy, force, loss, and selected parameter-gradient differences for eager-vs-eager candidates. SAI job `576838` repeated the gate on `4V100` with cueq enabled and completed with Slurm `COMPLETED` / `ExitCode 0:0`; the gate returned `ok: true`, no failed checks, energy max diff `2.384e-07`, force max diff `4.470e-08`, and loss diff `2.980e-07`. This is now the required precondition for future compiled subgraph candidates: any candidate must pass the same force-loss equivalence gate before it can be wired into training.

The first real subgraph candidate, `compile_readouts`, compiles only the readout modules on the equivalence candidate model. A CPU RECIO index `0` smoke passed after canonicalizing `torch.compile` wrapper parameter names from `_orig_mod.*` back to their logical MACE names. On SAI `4V100` with cueq enabled, job `576896` completed with Slurm `COMPLETED` / `ExitCode 0:0`, but the equivalence gate returned `status=error`, `ok=false`, and `RuntimeError('torch.compile with aot_autograd does not currently support double backward')`. Therefore readout-level `torch.compile` is not accepted for conservative force-loss training on this stack; the gate correctly prevents wiring it into `run_train`.

The second subgraph candidate, `compile_radial_embedding`, compiles only `model.radial_embedding` on the equivalence candidate model. A local CPU RECIO index `0` smoke and SAI `4V100` cueq job `577068` both completed the probe but failed the force-loss equivalence gate with the same `torch.compile with aot_autograd does not currently support double backward` error. Job `577068` exited `COMPLETED` / `ExitCode 0:0`, and its JSON recorded `status=error`, `ok=false`, `candidate=compile_radial_embedding`. Therefore radial-embedding-level `torch.compile` is also rejected for conservative force-loss training on this PyTorch/V100/cueq stack. The next viable compile work should either use kernels/subgraphs with explicit higher-order-gradient support or follow a deeper DeepMD-style make_fx/AOT path, rather than stacking more ordinary `torch.compile` wrappers around differentiable force-loss regions.

## Training Infrastructure Reference

The local TACE reference checkout is `/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/reference_repos/tace` at commit `c669bee`. The useful lessons for a larger MACE training refactor are infrastructure-level rather than model-copying:

- `docs/source/guide/training.rst` uses Hydra-style component configuration for dataset, trainer, callbacks, optimizer, scheduler, loss, logger, model, resume, and finetune. MACE's current CLI/YAML flow could gradually move toward typed component configs without changing model physics.
- `tace/lightning/trainer.py` centralizes trainer creation, callbacks, checkpoint policy, resume, logging, and scheduler behavior. MACE currently spreads these concerns across `run_train.py` and `tools/train.py`; a future refactor should isolate orchestration from model/loss code before adding heavier distributed or precision features.
- `tace/dataset/datamodule.py` supports rank-aware graph preprocessing and LMDB lazy loading/sharding. For MACE medium and large datasets this is a more promising memory and startup optimization than changing the equivariant model itself.
- TACE exposes Lightning precision modes such as `bf16-mixed`, but adopting Lightning wholesale would be a large API and checkpointing change. For this branch, the safer step is the narrower MACE-native `--train_amp_dtype` path plus explicit hardware gates.

These references suggest a two-track roadmap: keep small acceleration knobs compatible with existing MACE scripts now, and separately design a training-infra refactor around data caching, typed configs, callback isolation, and distributed strategy.

## Full RECIO/8k Validation Runs

Full 800 epoch single-GPU validation jobs were submitted on SAI `4V100` with the same random seed and split. The later `576509` run is a shorter 200 epoch stability gate for the corrected HybridMuon routing, not an equal-budget accuracy comparison against the 800 epoch baseline:

| Case | Slurm job | Status at submit check | Early validation |
| --- | ---: | --- | --- |
| `baseline_fp32_adam_cueq` | `576362` | completed, 800 epoch full run | no NaN through epoch 795; stage-two valid `21.2 meV/atom`, `267.5 meV/A`; mean `16.77 s/epoch`, max FB `4582 MB` |
| `fp32_hybrid_muon_cueq` | `576363` | cancelled after NaN | first NaN at epoch 125 with original Muon lr equal to base lr `0.04` |
| `fp32_hybrid_muon_cueq`, `hybrid_muon_lr_factor=0.1` | `576454` | cancelled after NaN | first NaN at epoch 80; lr reduction alone was not the root fix |
| `fp32_hybrid_muon_cueq`, effective-rank routing before Adam parity fix | `576498` | cancelled | superseded after fixing the Adam-routed branch to match `torch.optim.Adam(amsgrad=True)` |
| `fp32_hybrid_muon_cueq`, effective-rank routing plus Adam parity fix | `576509` | completed, 200 epoch stability gate | no NaN through epoch 195; stage-two valid `21.7 meV/atom`, `227.5 meV/A`; route summary has `Muon tensors: 0`; mean `17.01 s/epoch`, max FB `5452 MB` |

The generated case root is:

```text
/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-full-20260629-040425
/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/accel-hybrid-muon-adamfix-20260629
```

The visible SAI GPU partitions at submission time were V100-only (`4V100`, `4V100PX`, `8V100V0`), so no bf16 validation job was submitted. The bf16 path should be validated on A100/H100 or another native bf16 GPU; running it on V100 would not test the DPA4-style bf16 acceleration path.

The first full HybridMuon attempt exposed an important stability constraint: applying Muon's orthogonalized update at the same `lr=0.04` as Adam produced NaNs from epoch 125. A second run with `--hybrid_muon_lr_factor=0.1` still produced NaNs from epoch 80, so lr scale alone was not the root fix. The common route summary showed that all RECIO/MACELES Muon tensors were singleton matrix views such as `(1, 128)` and `(1, 2048)`. Following the effective-shape principle used by the TACE/DeepMD HybridMuon reference, singleton dimensions are now removed before routing; effective rank `<2` parameters route to Adam even if their names contain `readout`. The Adam-routed branch now also matches `torch.optim.Adam` semantics, including coupled weight decay and AMSGrad, so a zero-Muon route is a stability-preserving Adam fallback rather than a separate optimizer. The corrected `576509` gate crossed both old failure points, epoch 80 and 125, and completed with Slurm `COMPLETED` / `ExitCode 0:0`. For this RECIO/MACELES case, conservative routing means HybridMuon is not currently an acceleration source; Muon should only be re-enabled for genuine matrix-like dense parameters after a separate stability and accuracy gate.

## GPU D3 Backend Status

`mace_mp(..., dispersion=True, dispersion_backend="nvalchemi")` now routes D3 dispersion through an optional `NvalchemiDFTD3Calculator` ASE adapter. The adapter keeps the dispersion correction outside the MACE neural model and sums it at the calculator level, matching the existing `torch_dftd` architecture and preserving MACE model semantics.

The first mapped backend is deliberately narrow: PBE-D3(BJ), using the explicit Grimme parameters `a1=0.4289`, `a2=4.4407` Bohr, and `s8=0.7875`, because nvalchemi's `DFTD3ModelWrapper` takes explicit BJ damping parameters rather than an XC string. Other XC/damping combinations still require `dispersion_backend="torch_dftd"` until their parameters are mapped and verified against a reference.

`mace_env` has been live-tested with `nvalchemi-toolkit==0.1.0`, `nvalchemi-toolkit-ops==0.3.1`, `warp-lang==1.14.0`, `torch-dftd==0.5.3`, and PyTorch `2.8.0+cu128`.

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

The smoke script is `scripts/benchmarks/nvalchemi_d3_smoke/run_nvalchemi_d3_smoke.py`; submit it with `scripts/benchmarks/nvalchemi_d3_smoke/nvalchemi-d3-smoke.sbatch`. It defaults to a small ASE molecule and also accepts `--xyz`/`--index` for dataset structures. On SAI, a case-local sbatch with explicit script arguments was more reliable than passing long paths through `sbatch --export`.

A RECIO/8k periodic structure smoke also completed successfully on `4V100`:

| Check | Value |
| --- | ---: |
| Slurm job | `576442` |
| Structure | `train.xyz` index `0`, `Ag4Pd5`, 9 atoms, PBC |
| Exit state | `COMPLETED`, `0:0` |
| CPU D3 energy | `-5.519044399261 eV` |
| CUDA D3 energy | `-5.519046306610 eV` |
| CPU/CUDA energy absolute difference | `1.907349e-06 eV` |
| CPU/CUDA max force absolute difference | `5.081296e-06 eV/A` |
| CPU seconds per eval | `5.594959e-05` |
| CUDA seconds per eval | `5.523749e-05` |

This validates the optional CUDA backend on both a small molecule and a RECIO periodic cell. The original smoke timing rows were generated before the benchmark scripts cleared ASE calculator caches inside repeat loops, so they are retained only as correctness smoke evidence; use the cache-fixed benchmark below for timing claims.

A multi-structure benchmark harness is available at `scripts/benchmarks/nvalchemi_d3_smoke/run_nvalchemi_d3_benchmark.py`, with SAI submission template `scripts/benchmarks/nvalchemi_d3_smoke/nvalchemi-d3-benchmark.sbatch`. Example RECIO run:

```bash
cd /home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/d3-benchmark-nvalchemi
D3_XYZ=/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz \
D3_INDICES=0:32:4 D3_REPEAT=200 D3_MAX_STRUCTURES=8 D3_SUPERCELL=3,3,3 \
sbatch /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/nvalchemi_d3_smoke/nvalchemi-d3-benchmark.sbatch
```

The benchmark writes a JSON payload containing per-structure CPU/CUDA energy and force agreement, seconds per evaluation, CUDA speedup versus nvalchemi CPU, optional supercell expansion, and an explicit `torch_dftd_comparison` status. If `D3_COMPARE_TORCH_DFTD=1` is set but `torch_dftd` is not installed, the run still reports `torch_dftd_comparison=unavailable` instead of pretending a reference comparison was performed.

A cache-fixed RECIO supercell timing gate completed on SAI `4V100` after installing `torch-dftd==0.5.3` in `mace_env`:

| Check | Value |
| --- | ---: |
| Slurm job | `576712` |
| Exit state | `COMPLETED`, `0:0` |
| Structures | `train.xyz` indices `0,4,8`, each repeated `3 x 3 x 3` |
| Total atoms benchmarked | `594` |
| Repeat count | `10` true evaluations per calculator after clearing ASE cache |
| nvalchemi CPU mean seconds/eval | `4.294e-02` |
| nvalchemi CUDA mean seconds/eval | `1.354e-02` |
| torch_dftd CUDA mean seconds/eval | `3.265e-02` |
| nvalchemi CUDA speedup vs nvalchemi CPU | `3.17x` |
| nvalchemi CUDA speedup vs torch_dftd CUDA | `2.41x` |
| nvalchemi CPU/CUDA max energy difference | `7.629e-05 eV` |
| nvalchemi CPU/CUDA max force difference | `8.308e-06 eV/A` |
| nvalchemi CUDA vs torch_dftd max energy difference | `9.515e-02 eV` total, about `0.39 meV/atom` on the largest tested cell |
| nvalchemi CUDA vs torch_dftd max force difference | `9.085e-05 eV/A` |

This is the first production-like GPU D3 speed evidence: the nvalchemi CUDA backend is faster than both nvalchemi CPU and `torch_dftd` CUDA on RECIO periodic supercells while staying close in forces. The nonzero total D3 energy offset versus `torch_dftd` is small per atom but should be tracked across more chemistries before changing any default backend.

