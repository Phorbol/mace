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

A closer source-code reading of DeepMD commit `a9bcbc50` narrows the compile lesson further. DPA4/SeZM does not solve force-loss training by wrapping model submodules with ordinary `torch.compile`. In `deepmd/pt/model/model/sezm_model.py`, the conservative path first makes the per-edge displacement tensor the force autograd endpoint via `edge_vec.detach().requires_grad_(True)`, keeps neighbor-list construction and coordinate gathers outside that differentiated region, computes energy from this edge-vector leaf, and then calls a single `autograd.grad(energy, edge_vec, create_graph=self.training)` inside `edge_energy_deriv`. `trace_and_compile()` wraps that whole tensor-only `core_compute` closure and traces it with `make_fx(tracing_mode="symbolic", _allow_non_fake_inputs=True)`, so the inner force gradient is materialized into ordinary FX ops before Inductor sees the graph.

The rest of DeepMD's compile path is also structural rather than cosmetic: `compile_compat.py` strips make_fx-inserted saved-tensor detach chains so second-order parameter gradients are not severed, rebuilds the FX graph after erase/rewrite, promotes task-local buffers to explicit graph inputs, chooses pairwise-distinct prime trace shapes to avoid false symbolic shape equalities, decomposes `silu_backward`, disables DDPOptimizer graph splitting, forces int64 Inductor indexing, and locks down Triton/Inductor options for dynamic edge counts. This explains why MACE's current `compile_readouts`, `compile_radial_embedding`, and `compile_symmetric_contractions` candidates fail: they leave the nested `autograd.grad(create_graph=True)` inside an ordinary compiled module boundary, which is exactly the unsupported AOTAutograd pattern DPA4 avoids by tracing after the first derivative has been materialized.

For MACE, the DPA4-like prototype should therefore start from an edge-vector force endpoint. MACE already computes `vectors = positions[receiver] - positions[sender] + shifts`, and the LAMMPS/`prepare_graph` path already has a `vectors.requires_grad_(True)` mode, but ordinary training still computes forces by differentiating total energy with respect to atom positions. The next correct experiment is an edge-vector force equivalence probe: run the same MACE energy graph from detached `vectors`, compute per-edge gradients, scatter them back to atoms with sender/receiver signs, and compare energy, forces, force loss, and parameter gradients against the current position-gradient path on RECIO. Only if that eager equivalence passes should a make_fx/AOT prototype be attempted around the edge-vector closure, with DPA4-style detach repair and donated-buffer/Inductor workarounds guarded by tests.

That precondition now passes for ordinary `ScaleShiftMACE`. `probe_edge_vector_force_equivalence.py` reconstructs the MACE energy graph from detached edge vectors, computes `dE/d(edge_vec)`, scatters it back to atoms as `sender += grad`, `receiver -= grad`, and compares the result with the current position-gradient path after a force-loss backward. A CPU RECIO index `0` smoke with `hidden_channels=8`, `max_ell=1`, `num_interactions=1`, and `correlation=1` returned exact equality for energy, forces, loss, and all parameter gradients. SAI job `579346` repeated that small gate on `4V100` with cueq enabled and returned `ok=true`, energy diff `0.0`, force diff `3.73e-09`, loss diff `0.0`, and max parameter-gradient diff `2.38e-07`.

A larger SAI gate, job `579360`, used RECIO indices `0:32` (`286` atoms), cueq, `hidden_channels=32`, `max_ell=3`, `num_interactions=2`, and `correlation=2`. It also passed with `ok=true`, energy diff `2.38e-07`, force diff `1.49e-07`, loss diff `1.19e-07`, max parameter-gradient diff `5.96e-08`, and peak CUDA allocation `104.9 MB`. This is the first DPA4-style compile-relevant success for MACE: the force endpoint can be moved from atom positions to edge vectors without changing conservative-force training gradients. The next implementation step is to wrap this edge-vector energy/force closure in a `make_fx` trace, apply DPA4-style detach-chain repair, and only then try Inductor/AOT lowering.

That `make_fx` precondition is now also tested. On CPU, tracing the edge-vector force-loss closure without detach repair preserved energy, forces, and scalar loss, but severed the second-order parameter-gradient path: all checked parameter gradients failed the gate, and `interactions.0.skip_tp.weight` had no traced gradient. Enabling the DPA4-style saved-tensor detach repair removed `18` detach nodes, kept energy/forces/loss identical, and restored every checked parameter gradient exactly. SAI job `579410` repeated the repaired trace on `4V100` with cueq enabled; it removed `16` detach nodes, returned `make_fx_edge_vector.status=ok`, force diff `3.73e-09`, loss diff `0.0`, max parameter-gradient diff `9.54e-07`, and peak CUDA allocation `18.6 MB`. This makes the next compile target precise: trace the tensor-only edge-vector closure, repair saved-tensor detach chains, then attempt an Inductor/AOT lowering under the same equivalence gate. A no-repair trace is a useful negative control, not an acceptable training path.

The reusable tracing pieces behind this probe now live in `mace.tools.force_compile`: edge-gradient-to-atomic-force scattering, make_fx saved-tensor detach-chain repair, FX graph rebuilding, generic force-closure tracing, and optional Inductor compilation. The benchmark script still owns the MACE-specific edge-vector energy closure, but the graph surgery and compile wrapper are now production helpers so future training integration will not duplicate benchmark-only logic. A post-refactor CPU RECIO index `0` smoke with `hidden_channels=8`, `max_ell=1`, `num_interactions=1`, `correlation=1`, repaired make_fx, and dynamic Inductor compile passed the gate with no failed checks; make_fx detach nodes were reduced from `18` to `0`, force diff was `1.16e-08`, loss diff was `4.47e-07`, and max checked parameter-gradient diff was `1.67e-06`.

The first Inductor lowering gate is more nuanced. The probe can now optionally run `torch.compile(backend="inductor")` on the repaired FX graph. CPU RECIO index `0` passed with both `dynamic=False` and `dynamic=True`: the compiled graph preserved energy, forces, loss, and all checked second-order parameter gradients within the existing tolerance. On SAI `4V100`, job `579487` showed that CUDA Inductor also works for the no-cueq MACE/e3nn graph (`compile_graph=true`, `dynamic=true`, force diff `5.49e-08`, no failed checks). However, cueq compatibility is not yet solved. Jobs `579480` (`dynamic=true`) and `579485` (`dynamic=false`) both compiled and ran with cueq enabled, but the compiled force tensor had norm `0.0`; the gate rejected both with force diff `7.97e-02`, loss diff `9.46e-04`, and `11` failed checks. Therefore this path is not ready for training with cueq. The next compile work must isolate the cueq custom-op derivative/lowering interaction or place the compile boundary around no-cueq/e3nn-safe regions only; wiring this compiled graph into cueq training would silently train the wrong force objective.

The cueq failure has now been narrowed. SAI job `579543` showed that disabling only cueq `conv_fusion` does not help: the compiled force norm was still `0.0`. Job `579548`, which disabled `optimize_all` without enabling the required lower-level replacements, failed before the compile gate because the eager baseline hit a symmetric-contraction shape mismatch, so it is not a valid force-compile comparison. Granular controls then isolated the issue: disabling only `optimize_channelwise` (`579589`) or only `optimize_fctp` (`579591`) still produced zero compiled forces, and disabling both while keeping cueq linear/symmetric (`579600`) still failed. In contrast, job `579610` kept only cueq symmetric and disabled cueq linear; the compiled force graph passed with force diff `1.04e-07`. The most useful configuration, job `579631`, kept cueq conv fusion, channelwise, fctp, and symmetric enabled, but disabled cueq linear; it passed with force diff `1.06e-07`, no failed checks, and peak CUDA allocation `24.1 MB`. This points to cueq optimized Linear lowering as the current zero-force root cause, not cueq tensor products or symmetric contraction in general. The next implementation path should add a force-compile-specific cueq profile that leaves TP/symmetric acceleration enabled but routes Linear through the non-cueq implementation inside the compiled edge-vector force graph, then validate on the larger RECIO `0:32` gate before any training integration.

That larger gate now passes. SAI job `579742` used RECIO indices `0:32` (`286` atoms), `hidden_channels=32`, `max_ell=3`, `num_interactions=2`, `correlation=2`, CUDA dynamic Inductor, and the cueq-minus-linear profile (`conv_fusion=true`, `optimize_channelwise=true`, `optimize_fctp=true`, `optimize_symmetric=true`, `optimize_linear=false`). The eager edge-vector path still matched the position-gradient baseline (`ok=true`, force diff `7.45e-08`). The compiled repaired FX graph had `1365` nodes, removed `269` detach nodes to `0`, and passed the same force-loss parameter-gradient gate with force diff `4.17e-07`, loss diff `2.38e-07`, no failed checks, and peak CUDA allocation `127.6 MB`. This is the first larger RECIO evidence that the DPA4-style edge-vector force trace can be Inductor-compiled while retaining most cueq tensor-product/symmetric acceleration, provided cueq Linear is excluded from the compiled force graph.

The training integration gate for the edge-vector force compile path is intentionally fail-closed. `EdgeForceCompileConfig` defaults to disabled, unsupported outputs such as virials and stress are rejected, and probe payloads now expose `gate_result` metadata alongside the existing force-loss equivalence comparison. This does not enable compiled force training by default and is not a RECIO training speed claim; it makes the next training-loop wrapper auditable before any real RECIO/8k Adam, HybridMuon, cueq, or cueq-minus-linear benchmark is accepted.

The first real-batch training-step timing gate is now available through `profile_edge_force_training_steps.py` and `edge-force-training-step-profile.sbatch`. It uses the same DPA4-style edge-vector force endpoint, force-loss backward, optimizer step, and cueq-minus-linear profile that passed the equivalence gates above. The profiler now runs each optimizer/mode case in a fresh Python process when multiple cases are requested, because a same-process sweep repeatedly hit PyTorch's "backward through the graph a second time" failure after moving from eager cases into compiled cases. Process isolation is a harness fix, not a model change: it prevents compiler/autograd state from leaking across benchmark cases and lets the merged JSON preserve per-case outputs.

SAI job `580285` is the current authoritative step-level timing result. It ran on `4V100PX` with one `Tesla V100-SXM2-16GB`, RECIO `train.xyz` indices `0:32` (`286` atoms), ordinary `ScaleShiftMACE` with `hidden_channels=64`, `max_ell=2`, `num_interactions=2`, `correlation=3`, cueq conv fusion/channelwise/FCTP/symmetric enabled, cueq linear disabled, warmup `5`, repeats `20`, and both Adam and HybridMuon. All required force-loss gates were accepted.

| Optimizer | Mode | Mean step time | Median step time | Setup/compile time | Speedup vs position eager |
| --- | --- | ---: | ---: | ---: | ---: |
| Adam | `position_eager` | `13.88 ms` | `13.88 ms` | n/a | `1.00x` |
| Adam | `edge_eager` | `13.60 ms` | `13.60 ms` | `15.81 s` | `1.02x` |
| Adam | `edge_compile` | `5.48 ms` | `5.45 ms` | `36.26 s` | `2.53x` |
| HybridMuon | `position_eager` | `14.28 ms` | `14.24 ms` | n/a | `1.00x` |
| HybridMuon | `edge_eager` | `14.10 ms` | `14.12 ms` | `15.81 s` | `1.01x` |
| HybridMuon | `edge_compile` | `5.97 ms` | `5.99 ms` | `25.28 s` | `2.39x` |

This is the first non-toy evidence that the force-backward compile route can accelerate a real RECIO training step while keeping HybridMuon, force-loss gradients, and most cueq acceleration active. It is still not an end-to-end training-speed or final-accuracy claim. The compile/setup cost is tens of seconds, the benchmark repeats one fixed batch, and real RECIO training sees changing graph sizes and dataloader behavior. The next acceptance gate must wire this path into the training loop with shape/cache policy, run real epochs with validation and guard logging, and show that setup cost is amortized without degrading the energy/force validation trend.

A follow-up epoch-like multi-batch gate now quantifies that setup issue. The first attempt, job `580506`, failed after the Adam `position_eager` case when the same Python process moved into `edge_compile`, reproducing the same PyTorch saved-tensor/backward reuse failure seen in the single-step sweeps. The epoch profiler now uses isolated child processes for each optimizer/mode case, matching the fixed step profiler. Job `580633` then completed on `4V100PX` with RECIO `0:64` split into two real batches, two epochs, Adam and HybridMuon, `position_eager` and `edge_compile`, `hidden_channels=64`, `max_ell=2`, `num_interactions=2`, `correlation=3`, and cueq-minus-linear. All edge-compile gates were accepted.

| Optimizer | Mode | Cache hits | Compile setups | Median step excluding setup | Median step including setup | Setup total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adam | `position_eager` | `0/4` | `0` | `148.57 ms` | `148.57 ms` | `0.00 s` |
| Adam | `edge_compile` | `2/4` | `2` | `6.39 ms` | `6696.45 ms` | `49.38 s` |
| HybridMuon | `position_eager` | `0/4` | `0` | `148.90 ms` | `148.90 ms` | `0.00 s` |
| HybridMuon | `edge_compile` | `2/4` | `2` | `6.85 ms` | `1695.24 ms` | `28.70 s` |

This is the first real multi-batch evidence that the compiled edge-force path can train over multiple RECIO batches with HybridMuon once process isolation is used. It also exposes the next blocker: the current compiled closure captures batch tensors and labels, so the safe cache key is batch identity rather than shape. That is enough to study repeated fixed batches, but not enough for shuffled epoch training. The next implementation step is to promote non-vector batch tensors to explicit FX graph inputs, then gate shape-based reuse with energy, force, loss, and parameter-gradient equivalence before wiring the path into `take_step`.

That cache-safety blocker is now fixed in the epoch profiler. The compiled closure takes the edge vectors plus non-label batch tensors (`positions`, `edge_index`, `node_attrs`, `batch`, `ptr`, and `head`) as explicit FX inputs, computes energy and forces inside the repaired graph, and computes the supervised force/energy loss outside the compiled graph so labels are not captured. The profiler now supports `--edge-cache-scope shape`; shape-cache hits can be gated by re-running the current batch through both the position-gradient baseline and the cached compiled executable, comparing energy, forces, loss, and checked parameter gradients before the actual optimizer step. Two CPU RECIO smokes validated this path: repeated index `0` after one optimizer update, and distinct same-shape structures `16 -> 26`; both returned zero differences for energy, forces, loss, and checked gradients.

SAI job `581156` then validated the same shape-cache path on `4V100PX` with one `Tesla V100-SXM2-16GB`, RECIO indices `16,26`, `hidden_channels=64`, `max_ell=2`, `num_interactions=2`, `correlation=3`, cueq-minus-linear, Adam and HybridMuon, and cache-hit gate enabled. Both optimizers hit the same shape cache on the second distinct structure and accepted the gate. Adam's cache-hit gate reported energy diff `1.49e-08`, force diff `1.79e-07`, and loss diff `3.05e-05` within `rtol=1e-4`; HybridMuon reported energy diff `2.98e-08`, force diff `2.38e-07`, and loss diff `5.34e-05`. This proves that the cached executable uses current batch tensors and current model parameters rather than stale trace-time values.

A larger epoch-like throughput run, SAI job `581162`, used RECIO `0:64` split into two real batches, three epochs, Adam and HybridMuon, `position_eager` and `edge_compile`, the same `hidden_channels=64/max_ell=2/num_interactions=2/correlation=3` model, and cueq-minus-linear. Cache-hit gating was disabled in this timing run because job `581156` had already validated shape-cache correctness and the gate itself adds a baseline backward. All compile setup gates were accepted.

| Optimizer | Mode | Steps | Cache hits | Compile setups | Setup total | Median step excluding setup | Loss first -> last | Steady speedup vs position eager |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam | `position_eager` | `6` | `0` | `0` | `0.00 s` | `149.66 ms` | `2.724 -> 0.556` | `1.00x` |
| Adam | `edge_compile` | `6` | `4` | `2` | `45.70 s` | `6.09 ms` | `2.724 -> 1.267` | `24.59x` |
| HybridMuon | `position_eager` | `6` | `0` | `0` | `0.00 s` | `149.07 ms` | `2.724 -> 0.987` | `1.00x` |
| HybridMuon | `edge_compile` | `6` | `4` | `2` | `25.75 s` | `6.44 ms` | `2.724 -> 1.294` | `23.16x` |

This is now real multi-step, multi-batch, multi-epoch RECIO evidence that force-backward compile is compatible with HybridMuon and most cueq acceleration when cueq Linear is excluded from the compiled force graph. It is still not the final acceptance benchmark: the path is only integrated in the profiler, not `take_step`; setup cost is still large; the timing run used only `64` structures and did not run validation; and V100 cannot validate native bf16 acceleration. The next implementation step is to move the guarded edge-force compiled step into the training loop with separate train/eval cache slots and then run an 8k-scale multi-epoch validation comparison against ordinary MACE.

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

The first low-risk data-movement knob is `--non_blocking_transfer`, which forwards `non_blocking=True` to every batch `.to(device)` call in training, validation, and LBFGS closures. It is opt-in and does not change MACE energies, conservative forces, losses, optimizers, or cueq execution. It can only help when host tensors come from pinned memory, so use it together with the existing default `--pin_memory True`.

A 1 epoch RECIO/8k CUDA/cueq entry smoke with `--non_blocking_transfer True` completed as SAI job `578509` on `4V100` with Slurm `COMPLETED` / `ExitCode 0:0` in `00:02:20`. The run directory is `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/nonblocking-transfer-smoke-20260629/ordinary_adam_cueq_nonblocking`; it reached epoch `0`, wrote the final train/valid error table, and ended with `Done`.

A paired ordinary `ScaleShiftMACE` 20 epoch serial timing gate then ran on the same SAI `4V100` node (`4v100n26`) with cueq, Adam, batch size `32`, seed `123`, and only `non_blocking_transfer` changed. Baseline job `578667` and nonblocking job `578669` both completed with Slurm `COMPLETED` / `ExitCode 0:0` and elapsed `00:03:59`. Parsed mean logged epoch time was `6.4777 s/epoch` for baseline and `6.4661 s/epoch` for nonblocking, a negligible `0.18%` difference; max FB memory was unchanged at `4160 MB`. Last logged validation was also comparable (`49.71 meV/atom`, `309.99 meV/A` baseline; `48.08 meV/atom`, `308.29 meV/A` nonblocking). This validates the wiring and confirms no obvious accuracy or memory regression in the short gate, but it is not a meaningful speed lever for ordinary MACE on this V100/cueq case. Future training-infra work should prioritize data caching/lazy loading for larger datasets or return to the force/backward kernel/AOT hot path.

## Training Stability Guards

DPA4 and modern trainer stacks also treat unstable force-loss steps as an infrastructure problem rather than a model-architecture change. MACE now has opt-in guard flags for this layer:

```bash
--loss_skip True \
--loss_skip_nan True \
--loss_skip_large True \
--loss_skip_ema_window 100 \
--loss_skip_multiplier 3.0 \
--loss_skip_start_step 1000 \
--stable_grad_clip True \
--nonfinite_grad_guard True
```

`loss_skip` checks the scalar loss before `loss.backward()`, so skipped NaN or unusually large losses do not write gradients, do not call `optimizer.step()`, and do not update EMA. `stable_grad_clip` uses an overflow-resistant norm computation for very large gradients while matching PyTorch clipping on ordinary gradients. `nonfinite_grad_guard` records non-finite gradient norms and raises before checkpoint writes, preventing corrupted checkpoints from being accepted as a valid training state. All flags default to disabled, so existing MACE training behavior is unchanged unless a smoke or benchmark case explicitly enables them.

These guards are not a speedup by themselves. They are intended to make force-backward compile, AMP, and HybridMuon experiments fail closed during RECIO/8k sbatch runs, especially when testing DPA4-style compiled force-backward paths or bf16-capable GPU partitions. The local focused regression for the guard integration passed with `71 passed, 21 skipped` across training guards, `take_step`, training precision, HybridMuon, edge-vector force equivalence, force-backward compile ops, and training compile probes. A RECIO smoke case should enable the guards first with ordinary eager/cueq training, then repeat with any new compile path so skip counts, fallback status, validation trend, and checkpoint behavior can be compared.

The first guard-enabled SAI smoke, job `580031`, used ordinary `ScaleShiftMACE` on RECIO/8k with `num_channels=64`, `max_L=1`, `correlation=3`, cueq enabled, `mace_env`, `4V100`, and `rush-1o2gpu`. It completed with Slurm `COMPLETED` / `ExitCode 0:0` in `00:02:02`. The run reached epoch `4`, saved both stage-one and stage-two checkpoints/models, and ended with final stage-two validation `71.7 meV/atom` energy MAE and `384.2 meV/A` force MAE. The run directory is `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/stability-guard-smoke-20260629/ordinary_mace_adam_cueq_guards`.

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

For ordinary `ScaleShiftMACE`, the genuine dense MLP-like parameters are the radial tensor-product weight networks `interactions.*.conv_tp_weights.layer*.weight`, defined as `FullyConnectedNet([edge_feats] + radial_MLP + [conv_tp.weight_numel])`. These scalar radial MLP weights do not encode equivariant contraction coefficients themselves; they generate tensor-product weights from radial edge features, so they are the closest MACE analogue to dense hidden-layer matrices in the DeepMD/TACE HybridMuon references. The HybridMuon router now sends only 2D `*.conv_tp_weights.*.weight` tensors with effective rank at least 2 to Muon with reason `radial-tp-weight-mlp`; symmetric contractions, product tensors, e3nn/cueq flattened equivariant linear weights, readout singleton matrix views, scales, shifts, embeddings, and biases remain Adam-routed. A route probe on ordinary RECIO `ScaleShiftMACE` showed 8 Muon tensors before and after cueq conversion.

A 20 epoch ordinary `ScaleShiftMACE` smoke pair was run on SAI `4V100` with cueq enabled from root `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/ordinary-mace-hybrid-muon-radial-tp-20260629`:

| Case | Slurm job | Exit state | Muon tensors | Last valid MAE E | Last valid MAE F | Mean seconds/epoch | Max FB memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `ordinary_mace_adam_cueq` | `577291` | `COMPLETED`, `0:0` | `0` | `51.45 meV/atom` | `317.55 meV/A` | `6.309 s` | `4160 MB` |
| `ordinary_mace_hybrid_muon_cueq` | `577292` | `COMPLETED`, `0:0` | `8` (`132096` params) | `36.77 meV/atom` | `265.43 meV/A` | `6.556 s` | `4160 MB` |

This is useful positive evidence for ordinary MACE: the new Muon route is active, finite, checkpoint/eval compatible, and the early validation trend improved versus Adam in this matched short smoke. It is not yet a speedup claim: these two jobs ran concurrently on the same node and HybridMuon was about `3.9%` slower by logged epoch intervals. The next gate should run a longer ordinary-MACE comparison, preferably one job at a time or under identical occupancy, before accepting the route as an accuracy/generalization improvement or tuning Muon lr/weight decay for speed.

A cleaner 60 epoch serial gate then ran the same ordinary `ScaleShiftMACE` Adam case first, followed by HybridMuon with a Slurm `afterok` dependency, so the two jobs did not share the GPU concurrently. Both jobs used `4V100`, cueq, `mace_env`, `max_num_epochs=60`, `start_swa=45`, and `eval_interval=10`; the root was `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/ordinary-mace-hybrid-muon-radial-tp-serial60-20260629`.

| Case | Slurm job | Exit state | Last logged valid MAE E | Last logged valid MAE F | Final table valid MAE E | Final table valid MAE F | Mean seconds/epoch | Max FB memory |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ordinary_mace_adam_cueq` | `577382` | `COMPLETED`, `0:0` | `27.34 meV/atom` | `241.39 meV/A` | `27.3 meV/atom` | `241.4 meV/A` | `6.301 s` | `4162 MB` |
| `ordinary_mace_hybrid_muon_cueq` | `577387` | `COMPLETED`, `0:0` | `24.24 meV/atom` | `228.77 meV/A` | `24.2 meV/atom` | `228.8 meV/A` | `6.535 s` | `4160 MB` |

The serial gate strengthens the optimizer conclusion: the radial-TP-MLP Muon route remains active and stable for ordinary MACE, improves the 60 epoch validation trend versus Adam, and does not increase peak GPU memory. It is still not a throughput win on this V100/PyTorch stack: logged epoch intervals are about `3.7%` slower than Adam. The rational next optimizer step is not to broaden routing into the current flat e3nn/cueq equivariant or symmetric-contraction tensors; it is to tune Muon hyperparameters or make the Muon update cheaper while preserving the current conservative route and the force/energy validation gains.

The review notes in `review/` refine this point rather than reversing it. A future `EquivariantHybridMuon` should be Schur-aware / irrep-aware / path-aware: Muon may act on legal multiplicity/channel mixing matrices, but never by flattening across degree, parity, SO(2) frequency, species, correlation order, CG path, or other semantic axes. For current MACE parameters, `conv_tp_weights.*.weight` remains the only accepted Muon route. Any extension into equivariant modules first needs explicit module-provided block metadata plus equivariance, force-gradient, and block-health tests; without that, Adam/AdamW remains the physically safer route.

A dedicated optimizer-step micro-profile then isolated the remaining HybridMuon overhead using ordinary-MACE-like parameter shapes on a single SAI `4V100` GPU. Before switching the Adam-routed branch to PyTorch functional Adam, job `577496` measured `torch_adam_all=170.59 us/step`, `hybrid_muon_all=1401.22 us/step`, `hybrid_muon_group_only=816.96 us/step`, and `hybrid_adam_group_only=531.49 us/step`. After using `torch.optim._functional.adam(..., foreach=True)` for the Adam-routed branch, job `577517` measured `torch_adam_all=170.66 us/step`, `hybrid_muon_all=985.66 us/step`, `hybrid_muon_group_only=848.52 us/step`, and `hybrid_adam_group_only=154.72 us/step`. This removes most of the Python-loop Adam fallback overhead while preserving parity with `torch.optim.Adam(amsgrad=True)`. The remaining optimizer bottleneck is now the Muon group itself, so future speed work should target the Newton-Schulz update implementation or Muon hyperparameters, not broader parameter routing.

A follow-up 60 epoch ordinary-MACE training rerun with the functional Adam fallback, job `577530`, completed on SAI `4V100` with Slurm `COMPLETED` / `ExitCode 0:0` after switching the pending single-GPU short test to `rush-1o2gpu` QOS. It used the same RECIO/8k config, seed, cueq setting, and radial-TP-MLP Muon route as job `577387`. The final logged validation was `25.10 meV/atom` energy and `230.20 meV/A` force, with final table `25.1 meV/atom` and `230.2 meV/A`; mean logged epoch time was `6.361 s/epoch`, max FB memory was `4160 MB`, and no NaNs were reported. Compared with the previous HybridMuon serial gate (`6.535 s/epoch`), the functional Adam fallback improves real training throughput by about `2.7%`. It still remains about `1.0%` slower than the Adam baseline job `577382` (`6.301 s/epoch`), so HybridMuon is now close to throughput neutral on this case but not yet a speedup.

The next Muon-side optimization batches same-shape Newton-Schulz updates inside the Muon-routed group without changing the route, number of Newton-Schulz iterations, Muon learning rate, weight decay, or MACE model/loss semantics. Unit tests compare the batched helper against the existing per-tensor helper on square, wide, and tall updates, and verify that `_step_muon_group` batches repeated shapes. SAI optimizer-step profile job `577684` completed on `4V100` with Slurm `COMPLETED` / `ExitCode 0:0` and measured `torch_adam_all=171.04 us/step`, `hybrid_muon_all=677.57 us/step`, `hybrid_muon_group_only=502.72 us/step`, and `hybrid_adam_group_only=154.61 us/step`. Relative to job `577517`, this lowers the Muon group micro-kernel cost from `848.52` to `502.72 us/step` and the whole HybridMuon optimizer step from `985.66` to `677.57 us/step`.

A matched 60 epoch ordinary-MACE training gate, job `577710`, completed on SAI `4V100` with `rush-1o2gpu`, Slurm `COMPLETED` / `ExitCode 0:0`, the same RECIO/8k config, seed, cueq setting, and radial-TP-MLP Muon route. It reported no NaNs, final logged validation `24.72 meV/atom` energy and `229.49 meV/A` force, final table `24.7 meV/atom` and `229.5 meV/A`, mean logged epoch time `6.536 s/epoch`, and max FB memory `4160 MB`. This preserves the validation trend and memory profile, but it does not prove an end-to-end training speedup: the epoch timing is effectively the same as the pre-functional-Adam HybridMuon serial gate (`6.535 s/epoch`) and slower than the functional-Adam-only rerun (`6.361 s/epoch`). The conservative interpretation is that shape batching is a real optimizer-substep win, but ordinary-MACE RECIO training remains dominated by model/force work and run-to-run/node variation; acceptance as a training throughput optimization needs another paired serial gate or a lower-overhead Muon implementation.

To quantify that bottleneck directly, `scripts/benchmarks/recio8k_accel/profile_training_step_phases.py` profiles one realistic force-loss training step by phase: `zero_grad`, `forward_force_loss`, `backward_clip`, and `optimizer_step`. The companion SAI template is `scripts/benchmarks/recio8k_accel/training-step-phase-profile.sbatch`; it defaults to RECIO `train.xyz` indices `0:32`, `ScaleShiftMACE`, cueq enabled, `max_ell=3`, warmup/repeat controls, `mace_env`, `4V100`, and `rush-1o2gpu`. A CPU smoke is:

```bash
python scripts/benchmarks/recio8k_accel/profile_training_step_phases.py \
  --device cpu --no-enable-cueq --indices 0 --hidden-channels 8 \
  --max-ell 1 --num-interactions 1 --correlation 1 \
  --warmup 0 --repeats 1 --optimizers adam \
  --output /tmp/mace_step_phase_smoke.json
```

SAI job `577924` is the authoritative steady profile after fixing the harness to match the real training model (`max_ell=3`; route summary `Muon tensors: 8 (132096 parameters)`). It used warmup `10`, repeats `30`, batch `32`, `286` atoms, cueq, and completed with Slurm `COMPLETED` / `ExitCode 0:0`. Median phase times were: Adam total `20.34 ms`, forward-force-loss `10.17 ms` (`50.0%`), backward+clip `9.84 ms` (`48.4%`), optimizer step `0.28 ms` (`1.39%`); HybridMuon total `20.24 ms`, forward-force-loss `9.81 ms` (`48.4%`), backward+clip `9.54 ms` (`47.1%`), optimizer step `0.85 ms` (`4.20%`). This explains why optimizer micro-kernel improvements do not reliably move epoch time: even HybridMuon's larger optimizer step is a small single-digit percentage of the full force-loss step, while conservative force construction and second-order backprop dominate. The next high-leverage speed path for ordinary MACE should focus on the model/force/backward region, cueq/e3nn kernels, data batching, or force-safe compiled subgraphs, not more optimizer-only micro-optimizations.

`scripts/benchmarks/recio8k_accel/profile_force_energy_modes.py` then separates the same force-training bottleneck into energy-only and force-training modes. The SAI template is `scripts/benchmarks/recio8k_accel/force-energy-mode-profile.sbatch`. A CPU smoke is:

```bash
python scripts/benchmarks/recio8k_accel/profile_force_energy_modes.py \
  --device cpu --no-enable-cueq --indices 0 --hidden-channels 8 \
  --max-ell 1 --num-interactions 1 --correlation 1 \
  --warmup 0 --repeats 1 --output /tmp/mace_force_energy_smoke.json
```

SAI job `577998` completed on `4V100` with cueq, RECIO indices `0:32`, warmup `10`, repeats `30`, and Slurm `COMPLETED` / `ExitCode 0:0`. Median timings were `energy_forward=4.59 ms`, `energy_loss_backward=9.18 ms`, `force_forward=9.55 ms`, and `force_loss_backward=19.25 ms`. The derived conservative-force increments are therefore about `+4.96 ms` for force construction (`force_forward - energy_forward`) and `+10.07 ms` for the force-loss second-order backward path (`force_loss_backward - energy_loss_backward`).

A matched no-cueq run, job `578064`, used the same SAI `4V100` node class, RECIO indices `0:32`, warmup `10`, repeats `30`, and only added `--no-enable-cueq` to the profiler command. It completed with Slurm `COMPLETED` / `ExitCode 0:0` and produced this median comparison:

| Mode | cueq job `577998` | no-cueq job `578064` | no-cueq / cueq |
| --- | ---: | ---: | ---: |
| `energy_forward` | `4.59 ms` | `6.07 ms` | `1.32x` |
| `energy_loss_backward` | `9.18 ms` | `19.71 ms` | `2.15x` |
| `force_forward` | `9.55 ms` | `17.18 ms` | `1.80x` |
| `force_loss_backward` | `19.25 ms` | `49.49 ms` | `2.57x` |

With cueq disabled, the force-construction increment rises from `+4.96 ms` to `+11.11 ms`, and the force-loss second-order backward increment rises from `+10.07 ms` to `+29.77 ms`. cueq is therefore already a major part of the answer for ordinary MACE on this V100 stack and should remain compatible with any DPA4-inspired compile or training-infra work.

The force/energy profiler now also accepts the same conservative training compile wrapper as `run_train`: `--train-compile`, `--train-compile-mode`, `--train-compile-fullgraph`, and `--train-compile-allow-fallback`. The SAI sbatch template keeps eager profiling as the default and enables these flags only when `MACE_FORCE_PROFILE_TRAIN_COMPILE=1` is set. A short CUDA/cueq compile-profile job, `578121`, used RECIO indices `0:32`, warmup `2`, repeats `5`, `train_compile=true`, and completed with Slurm `COMPLETED` / `ExitCode 0:0`. Median timings were `energy_forward=3.40 ms`, `energy_loss_backward=6.19 ms`, `force_forward=10.87 ms`, and `force_loss_backward=19.74 ms`; the payload recorded `compile_disabled=false` for both energy modes and `compile_disabled=true` after entering the force modes. This is the useful boundary: ordinary `torch.compile` is beneficial for energy-only work on this stack, but conservative force construction/backward still falls back to eager once second-order force training is involved.

The remaining high-leverage region is therefore still conservative force/backward. Future work should prototype force-safe lower-level kernels, custom autograd boundaries, or DeepMD-style make_fx/AOT handling with explicit energy/force/parameter-gradient equivalence gates instead of replacing MACE architecture.

`scripts/benchmarks/recio8k_accel/profile_force_ops.py` adds a torch-profiler view of the same RECIO batch and shares the force/energy mode runner, so it profiles the same conservative-force path. Its SAI template is `scripts/benchmarks/recio8k_accel/force-op-profile.sbatch`. Job `578243` completed on `4V100` with cueq, RECIO indices `0:32`, warmup `2`, modes `energy_forward,force_loss_backward`, and Slurm `COMPLETED` / `ExitCode 0:0`. The profiler uses PyTorch 2.8 `device_time_total` fields and filters out the outer `mace_*` record-function rows; the device-time values are profiler aggregation data, not a replacement for the wall-clock medians above. It now also supports `--group-by-input-shape`, which enables `record_shapes` and groups profiler rows by input shape so repeated contraction/backward patterns can be identified before attempting any force-safe compile or kernel work.

Top device-time operators from job `578243` point to the next kernel/AOT boundaries:

| Mode | Top device-time operators | Interpretation |
| --- | --- | --- |
| `energy_forward` | `aten::einsum` (`0.329 ms`), `aten::bmm` (`0.237 ms`), `cuequivariance_ops::tensor_product_uniform_1d_jit` (`0.159 ms`), `aten::mm` (`0.154 ms`) | Forward cost is split between tensor contractions, cueq tensor products, and dense matrix multiplies; cueq compatibility must stay first-class. |
| `force_loss_backward` | `aten::bmm` (`1.412 ms`), `SliceBackward0` (`1.322 ms`), `BmmBackward0` (`1.237 ms`), cueq tensor-product backward/evaluate (`~1.08 ms`), `aten::mm` (`1.059 ms`), `aten::mul` (`1.038 ms`) | The second-order force-loss path is dominated by many autograd contraction/backward nodes, not a single optimizer or readout MLP. Lower-level work should target bmm/einsum/cueq tensor-product backward and graph-level AOT treatment of these force-safe regions. |

A shape-grouped follow-up, job `578770`, completed on `4V100` with cueq, RECIO indices `0:32`, mode `force_loss_backward`, warmup `2`, and Slurm `COMPLETED` / `ExitCode 0:0`. It recorded `286` atoms and `186.5 MB` peak allocated CUDA memory. The dominant shape-aware rows include `BmmBackward0` over `[1,286,128]`, `[1,858,128]`, and `[1,1430,128]` batches, `MmBackward0` over `[4632,64]` and `[4632,1280]`, cueq tensor-product backward with shapes like `[4632,1280]`, `[286,512]`, `[4632,16]`, and many small elementwise/fill/copy kernels. This confirms that the compile target is not a single large FFN-style matrix but a cluster of fixed-shape contraction/backward patterns plus cueq tensor-product backward. The next force-backward compile prototype should therefore be a force-equivalent custom autograd/AOT boundary around one repeated contraction family, gated by energy/force/loss/parameter-gradient equivalence, not another whole-module `torch.compile` wrapper.

This narrows the DPA4-inspired compile lesson for MACE: a useful next prototype should not wrap more Python modules with ordinary `torch.compile`; it should either improve cueq-backed tensor-product backward, fuse repeated contraction/backward patterns, or trace an explicitly force-equivalent AOT subgraph whose parameter gradients pass the existing energy/force/loss/gradient equivalence gate.

`probe_force_backward_compile_ops.py` is the first synthetic check for that narrower path. It does not instantiate MACE; instead it reproduces the repeated `mm`/`bmm` contraction shapes observed in job `578770`, builds a conservative-force-style objective `loss(||-dE/dx||)`, and compares eager against compiled second-order gradients for both inputs and weights. This is deliberately a lower-level AOT probe: it asks whether force-loss backward over the contraction family can compile at all before any MACE training code is changed.

Two SAI `4V100` jobs define the current result. Job `578960` showed that `mm_4632x1280_1280x64` compiled and matched eager gradients with about `2.0x` mean-step speedup, but both `bmm` cases failed with PyTorch's donated-buffer restriction for compiled backward under `create_graph=True`. Job `579022` reran the same probe with `torch._functorch.config.donated_buffer=False`; all three cases then compiled and passed the gradient equivalence gate. On that run, `mm_4632x1280_1280x64` improved from `1.194 ms` eager to `0.549 ms` compiled (`2.17x`), `bmm_1x286x128_128x128` from `0.424 ms` to `0.360 ms` (`1.18x`), and `bmm_1x858x128_128x128` from `0.420 ms` to `0.328 ms` (`1.28x`). The maximum absolute input/weight-gradient differences stayed below `6e-10` for the `bmm` cases and below `1e-11` for the `mm` case.

This is useful but intentionally limited evidence. It says fixed-shape contraction/backward families can be compiled through a force-loss second-order objective on the current PyTorch/V100 stack if the donated-buffer workaround is applied. It does not yet prove that a full MACE subgraph is compile-safe, because MACE adds equivariant tensor-product semantics, cueq kernels, indexing/slicing, atom-graph aggregation, and trainable physical paths. The next compile prototype should therefore wrap one real repeated contraction family or custom-autograd boundary inside MACE, with `donated_buffer=False` only around that experiment, and must pass the existing energy, force, loss, and parameter-gradient equivalence gate before it is wired into training.

The first real-module follow-up is the `compile_symmetric_contractions` equivalence candidate in `probe_training_compile.py`, which wraps only `products[*].symmetric_contractions` and then runs the existing conservative-force gate. This candidate is intentionally diagnostic, not accepted training code. A CPU RECIO index `0` smoke with `hidden_channels=8`, `max_ell=1`, `num_interactions=1`, and `correlation=1` returned `status=error`, `ok=false`, and the same AOTAutograd double-backward error. SAI job `579253` repeated the candidate on `4V100` with cueq enabled and also completed at Slurm level while the equivalence gate rejected the candidate with `RuntimeError('torch.compile with aot_autograd does not currently support double backward')`. The run also showed whole force-loss compile fallback remained active (`compile_disabled=true` for `compile_force_loss`), while energy-only compile still ran without disabling compile.

This rejects the simplest real MACE submodule wrapper. The next attempt should not be another ordinary `torch.compile(module)` boundary. It should either use the synthetic probe's `donated_buffer=False` workaround inside a lower-level AOT/custom-autograd experiment, or move closer to the actual repeated `mm`/`bmm` contraction kernels extracted from symmetric contraction/cueq internals, while preserving the same force-loss parameter-gradient gate.

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



## Edge-Force Cache Policy Smoke on RECIO/8k

A real RECIO/8k strict smoke was run after adding the edge-force cache policy layer and CLI controls. This was not a toy task: the run used the full `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz` split, `ScaleShiftMACE`, `num_interactions=2`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size 32, `default_dtype=float32`, cueq-minus-linear (`optimize_linear=False`, channelwise/symmetric/FCTP/fusion enabled), and `HybridMuon`.

| Check | Value |
| --- | ---: |
| Slurm job | `581318` |
| Node | `4v100pxn10` |
| Final state | `CANCELLED` after sufficient negative-speed evidence |
| Elapsed | `00:07:52` |
| Edge-force policy | `repeat_only`, `min_repeats=2` |
| Gate tolerance | `atol=5e-5`, `rtol=2e-4` |
| Epoch 0 opt rows | `474` eager policy skips |
| Epoch 0 compile rows | `0` |
| Epoch 1 sampled opt rows | `25` |
| Epoch 1 compile rows | `25` |
| Epoch 1 cache hits | `0` |
| Epoch 1 mean setup seconds | `11.95 s` |
| Epoch 1 mean opt-step wall time | `16.22 s` |
| Validation after epoch 0 | `MAE_E_per_atom=666.55 meV`, `MAE_F=702.98 meV/A` |

This proves three useful things. First, the policy path works on a real MACE training job with cueq-minus-linear and HybridMuon: epoch 0 used eager force training with `edge_force_compile_disabled_reason=min_repeats`, and epoch 1 entered strict graph compile with `edge_force_gate_accepted=true`. Second, the original `1e-5` absolute gate was too tight for float32+CUEQ+Inductor on RECIO; one strict run (`581317`) failed with max force difference `1.86e-5`, while `5e-5` accepted the same real path. Third, `repeat_only` is only a safety policy, not a speed solution: the second epoch compiled each repeated shape but still had `edge_force_cache_hit=false`, so setup cost dominated and the run was intentionally cancelled to avoid wasting GPU time.

The next speed-focused implementation must therefore move beyond exact-shape repeat-only caching. The practical next target is bucketed padding or a smaller DPA4-style compiled boundary whose cache key does not vary with batch identity. Any bucket implementation must first pass energy, force, scalar loss, and selected parameter-gradient equivalence before being wired into training.

## Edge-Force Bucket Cache Follow-up on RECIO/8k

A bucket-cache experiment was run after adding bucket cache keys and shape metrics. The goal was to test whether Inductor `dynamic=True` could reuse a real-traced edge-force graph across nearby RECIO batches without padding. This was tested with `HybridMuon`, cueq-minus-linear, strict no-fallback mode, atom buckets `320,512,768,1024,1536,2048`, edge buckets `8192,16384,32768,65536,131072,262144`, and `bucket_margin=2.0`.

| Job | Code variant | Result | Evidence |
| --- | --- | --- | --- |
| `581357` | bucket key only | failed on second batch | first batch accepted; second batch hit `aten.index` fake-tensor shape error with baked atom length `469` |
| `581376` | promoted `_num_atoms_arange` as explicit input | failed on second batch | first batch accepted; second batch hit `aten.expand(..., [469])` |
| `581406` | replaced node-head energy/readout selection with `gather` | failed on second batch | first batch accepted; second batch still hit `aten.expand(..., [469])` |
| `581417` | also replaced `head[batch]` with `gather` | failed on second batch | first batch accepted; second batch still hit `aten.expand(..., [469])` |

The first batch in these runs had `edge_force_num_atoms=469` and `edge_force_num_edges=12276`, with strict gate acceptance and compile setup around 34-35 s. Every follow-up failed before recording a second opt row. This is useful negative evidence: changing the cache key to bucket shape is not sufficient because the real `make_fx` trace still bakes the first batch atom dimension into the FX graph. `torch.compile(dynamic=True)` cannot recover full atom-count polymorphism from that already-specialized graph.

The next speed path should therefore not spend more effort on bucket keys without padding. The viable options are: (1) implement real padding plus masks so runtime tensors match the traced bucket sizes without changing energy/force/loss semantics, or (2) move the compiled boundary closer to DPA4's pure tensor core and make all variable graph data explicit symbolic inputs. Until one of those is implemented, `bucket` mode is diagnostic only and should be used with strict gate/no-fallback smoke tests, not as a claimed acceleration path.



## 2026-07-01 8k HybridMuon Edge-Force Compile Gate

The DPA4-style edge-force compiled loss is now wired into the normal `run_train` path and has been tested on the real RECIO/8k dataset, not only fixed-batch profilers. The reproducible SAI entry point is:

```bash
env MACE_CONDA_ENV=mace_develop \
  RUN_ROOT=runs/recio8k_hmuon_compile_bench \
  NAME=recio8k_hmuon_compile_bs16_10ep_retry \
  EDGE_FORCE_COMPILE=True \
  EDGE_FORCE_CACHE_POLICY=bucket \
  EDGE_FORCE_BUCKET_ATOMS=512,768 \
  EDGE_FORCE_BUCKET_EDGES=8192,16384 \
  EDGE_FORCE_BUCKET_MARGIN=0.0 \
  EDGE_FORCE_CACHE_HIT_GATE=False \
  BATCH_SIZE=16 MAX_NUM_EPOCHS=10 SHUFFLE=False ENABLE_CUEQ=False \
  sbatch scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

Use the `env VAR=... sbatch ...` form on SAI. The `sbatch --export=ALL,...` form caused jobs `588834` and `588835` to be cancelled before the batch script body ran and did not leave stdout/stderr files.

Paired 10-epoch HybridMuon runs on `4V100PX`, one V100, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size `16`, and `mace_develop` gave:

| Case | Job | Status | Slurm elapsed | Final epoch MAE E | Final epoch MAE F | Training-path evidence |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| eager HybridMuon | `588862` | completed | `00:03:39` | `123.44 meV/atom` | `331.87 meV/A` | epoch timestamps give about `16.7 s/epoch` |
| edge-force compile HybridMuon | `588869` | completed | `00:04:14` | `124.91 meV/atom` | `334.15 meV/A` | epoch 1-9 `opt_step_seconds` mean `13.061 s`, min `13.053 s`, max `13.091 s` |

Compile job `588869` compiled two bucket executables in epoch 0, then reused them for all later steps: epoch 0 summary reported `steps=475`, `cache_hits=473`, `new_compiles=2`, `fallbacks=0`, `runtime_recompiles=0`, `compile_setup_seconds=47.619`, and buckets `512x8192:469`, `512x16384:6`. Epochs 1-9 all reported `cache_hits=475`, `new_compiles=0`, `fallbacks=0`, and `runtime_recompiles=0`.

This proves the current code can run real RECIO/8k multi-epoch HybridMuon training with the compiled force-backward path and no eager fallback. It also shows the remaining performance boundary: the hot training path is faster than eager by about `22%` per epoch on this V100 case, but a 10-epoch end-to-end Slurm run is still slower because it pays about `47.6 s` of first-epoch compile setup plus the same final evaluation/checkpoint overhead. Longer runs, compile warmup, or lower setup cost are required before claiming end-to-end speedup.

The benchmark script defaults were updated to reflect the measured V100 configuration: `BATCH_SIZE=16`, `MAX_NUM_EPOCHS=10`, `EDGE_FORCE_BUCKET_ATOMS=512,768`, and `EDGE_FORCE_BUCKET_EDGES=8192,16384`. Batch size `32` was not kept as the default because it pads most steps to larger edge buckets and did not show a hot-path speedup in the earlier V100 timing gate.

### CUEQ and bf16 Status on SAI V100

The current working training environment remains `mace_develop` with `torch 2.10.0+cu128`; it does not include CUEQ. The `mace_torch212` environment includes `cuequivariance==0.10.0` and `cuequivariance-torch==0.10.0`, but its `torch 2.11.0+cu128` wheel does not include V100 `sm70` kernels. A 1-epoch CUEQ diagnostic, job `588894`, failed during model conversion with `CUDA error: no kernel image is available for execution on the device` after PyTorch warned that the wheel supports CC `7.5+` but the node GPU is Tesla V100 `7.0`.

An isolated `mace_develop_cueq` environment was cloned from `mace_develop`, then `cuequivariance==0.10.0` and `cuequivariance-torch==0.10.0` were installed with `--no-deps` so torch stayed at `2.10.0+cu128`. With conv fusion enabled, job `588925` failed because `cuequivariance_ops_torch` was unavailable and the fallback `SegmentedPolynomialNaive` object has no `buffer_num_segments`, which MACE's conv-fusion wrapper needs. The `cuequivariance_ops_torch` / `cuequivariance-ops-torch` package was not available from the configured Aliyun PyPI mirror.

After adding a `CUEQ_CONV_FUSION=False` switch to the benchmark script, job `588937` confirmed that no-fusion CUEQ reaches the naive fallback instead of the sm70 torch-kernel failure. It ran for about 5 minutes without producing one RECIO/8k epoch and was cancelled to avoid wasting card time.

A follow-up environment audit corrected the package-name issue but exposed a harder platform limit. The real CUEQ ops packages are `cuequivariance-ops-torch-cu12` and `cuequivariance-ops-cu12`, not `cuequivariance-ops-torch`; both are present on the Aliyun PyPI mirror. `cuequivariance-ops-torch-cu12==0.10.0` downloaded as a `403 kB` wheel, and `cuequivariance-ops-cu12==0.10.0` downloaded as a `31.1 MB` wheel at about `2.1 MB/s`. After adding the missing lightweight dependencies `nvidia-ml-py` and `platformdirs`, `cuequivariance`, `cuequivariance_torch`, `cuequivariance_ops`, and `cuequivariance_ops_torch` all imported in `mace_develop_cueq`.

The resulting V100 diagnostic is still negative. Real RECIO/8k job `589000` reached model conversion and HybridMuon routing with `conv_fusion=True`, then failed at the first CUEQ `uniform_1d` kernel launch with `cudaErrorNoKernelImageForDevice`. Inspecting `libcue_ops.so` from the installed wheel showed only `sm_100` and `sm_120` targets. A wheel scan across `cuequivariance-ops-cu12` versions `0.4.0` through `0.9.1` found no `sm_70` target either: `0.4.0` contains `sm_90`, while `0.5.0` through `0.9.1` contain only `sm_100` and `sm_120`. The official cuEquivariance repository cloned at `reference_repos/cuEquivariance` contains the Python frontend, while NVIDIA's documentation installs CUDA kernels through the separate `cuequivariance-ops-*-cu12/cu13` PyPI packages. No source distribution for `cuequivariance-ops-cu12` was available through `pip download --no-binary`.

Therefore the current SAI V100 platform cannot run NVIDIA's CUEQ accelerated ops from the available binary packages. This is a platform/kernel-architecture limit, not a MACE model-design conclusion: CUEQ compatibility should remain guarded in code, but CUEQ speed claims need a GPU whose compute capability is present in the NVIDIA wheel, or NVIDIA-provided ops source/wheels built for `sm70`. The visible SAI GPU partitions at this check were V100-only (`4V100`, `4V100PX`, `8V100V0`).

bf16 remains unsupported on the visible SAI V100 partitions. `mace.tools.precision.TrainingPrecisionConfig` correctly fails closed when `torch.cuda.is_bf16_supported()` is false. Native bf16 training must be validated on A100/H100 or another bf16-capable partition, not on V100.

### 30 Epoch HybridMuon End-to-End Gate

A longer paired run was submitted to test whether the hot-path speedup from edge-force compile amortizes the first-epoch compile setup. Both jobs used the same real RECIO/8k split, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size `16`, `HybridMuon`, `mace_develop`, no CUEQ, `4V100PX`, and `MAX_NUM_EPOCHS=30`.

| Case | Job | Status | Slurm elapsed | Final valid MAE E | Final valid MAE F | Test MAE E | Test MAE F | Epoch timing |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| edge-force compile HybridMuon | `589058` | completed | `00:09:09` | `75.1 meV/atom` | `274.7 meV/A` | `86.9 meV/atom` | `261.1 meV/A` | epoch 1-29 mean interval `14.70 s`; hot `opt_step_seconds` mean `13.04 s` |
| eager HybridMuon | `589059` | completed | `00:09:21` | `69.3 meV/atom` | `268.6 meV/A` | `78.2 meV/atom` | `252.4 meV/A` | epoch 1-29 mean interval `17.02 s` |

The compile run's epoch 0 summary reported `compile_setup_seconds=46.967`, `opt_step_seconds=68.683`, buckets `512x8192:469` and `512x16384:6`. Epochs 1-29 then reported `cache_hits=475`, `new_compiles=0`, `fallbacks=0`, and `runtime_recompiles=0` for every epoch, with hot-path `opt_step_seconds` between `13.036 s` and `13.053 s`.

This is the first real RECIO/8k multi-epoch evidence that the integrated force-backward compile path can beat eager end-to-end, but the current win is small: `549 s` vs `561 s`, about `2.1%`. The hot training path is materially faster, but first-epoch compile setup, validation, checkpoint/model export, and final evaluation dominate enough that 30 epochs only barely amortizes the setup cost. Accuracy also cannot be claimed unchanged from this single paired run: both paths are stable and same-order, but eager ended with slightly lower validation/test MAE. The next acceptance gate should either reduce setup cost or increase the run length/model size, then repeat with multiple seeds before claiming no precision/generalization regression.

A follow-up configuration tested whether compiling the rare `512x16384` bucket is worthwhile. Job `589137` used the same setup but set `EDGE_FORCE_BUCKET_EDGES=8192`, so the 6 large-edge batches per epoch took the eager `no_bucket_match` path while the common `512x8192` bucket remained compiled. This reduced compile setup to `38.052 s` and produced `compiled=469`, `fallbacks=6`, `cache_hits=469`, `new_compiles=0`, `runtime_recompiles=0` for epochs 1-29. The hot-path `opt_step_seconds` mean was `13.057 s`, and Slurm elapsed improved to `00:09:00` (`540 s`), about `3.7%` faster than eager and `1.6%` faster than the two-bucket compile job.

The speed result is useful but not enough to make this the default. The same run ended with final valid `85.1 meV/atom`, `278.7 meV/A` and test `94.7 meV/atom`, `265.1 meV/A`, worse than both the full two-bucket compile run and the eager run in this single-seed comparison. That may be normal optimizer/numerical trajectory sensitivity, but it means rare-bucket eager fallback should remain an explicit benchmark knob until repeated seeds or longer runs show no accuracy/generalization regression.

A second paired seed check was added after the benchmark script learned `SEED=${SEED:-123}`. Seed `456` jobs `589187` (eager) and `589188` (two-bucket compile) completed in `00:09:19` and `00:09:14`, respectively. The compile run again had no fallback or runtime recompile, with epoch 1-29 mean interval `14.89 s`, hot `opt_step_seconds` mean `13.16 s`, setup `47.571 s`, and bucket distribution `512x8192:464`, `512x16384:11`. Eager's epoch 1-29 mean interval was `16.93 s`.

| Seed | Case | Job | Slurm elapsed | Final table valid MAE E | Final table valid MAE F | Test MAE E | Test MAE F |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `456` | edge-force compile HybridMuon | `589188` | `00:09:14` | `110.4 meV/atom` | `285.7 meV/A` | `116.0 meV/atom` | `269.0 meV/A` |
| `456` | eager HybridMuon | `589187` | `00:09:19` | `68.4 meV/atom` | `281.1 meV/A` | `75.8 meV/atom` | `262.8 meV/A` |

This second seed strengthens the performance conclusion but weakens the accuracy claim. Compile consistently improves hot-path timing and gives small end-to-end walltime wins at 30 epochs, but the validation/test force trend is not yet equal to eager. The likely next debugging target is numerical parity over long optimizer trajectories: log per-step or per-epoch energy/force/loss deltas between eager and compiled edge-force on a fixed validation batch, then decide whether tighter compile/eager equivalence gates, fewer compiled regions, or a longer/full training schedule removes the observed drift.

### Reference Source Checkouts

The DPA4/TACE/NVIDIA reference repositories requested for source-level comparison are now local under this worktree:

```text
reference_repos/deepmd-kit
reference_repos/tace
reference_repos/nvalchemi-toolkit
```

The relevant source anchors found in this pass are DeepMD's `deepmd/pt/model/model/sezm_model.py` and `deepmd/pt/utils/compile_compat.py` for force-backward make_fx/Inductor cache design, TACE's `tace/utils/optimizer/hybrid_muon.py` for `muon_mode={2d,flat,slice}` and parameter routing, and NVIDIA's `nvalchemi/models/dftd3.py` for the Warp-backed `DFTD3ModelWrapper`. The D3 code should be integrated as a separate composable dispersion module/calculator branch that returns energy/forces/virial from the NVIDIA kernel, not copied into MACE's equivariant message-passing layers.

### Periodic gradient parity diagnostics for edge-force compile

After the seed-456 30-epoch compile run showed worse final validation/test metrics than eager, a periodic same-weight parity diagnostic was added to the training wrapper. The new opt-in flags are `--edge_force_compile_parity_check_interval`, `--edge_force_compile_parity_check_gradients/--no-edge_force_compile_parity_check_gradients`, and `--edge_force_compile_parity_check_strict/--no-edge_force_compile_parity_check_strict`. The diagnostic runs an eager position-gradient snapshot and a compiled edge-force snapshot on the same current batch and weights, compares energy, forces, loss, and parameter gradients, logs `edge_force_parity_*` metrics, then clears diagnostic gradients before the real training backward.

Two real RECIO/8k HybridMuon diagnostics were run on SAI `4V100PX` with seed `456`, bucket compile, symbolic tracing, and no CUEQ:

| Job | Schedule | Parity interval | Strict | Result |
| --- | --- | --- | --- | --- |
| `589335` | 3 epochs | 475 compiled steps | false | 3 checks, all accepted under the benchmark tolerance; max force diff `1.53e-5`, max loss diff `2.29e-5`, max parameter-gradient diff `1.25e-4`. |
| `589367` | 1 epoch | 50 compiled steps | false, `EDGE_FORCE_ATOL=1e-5` | 9 checks across real batches; max energy diff `3.05e-5`, max force diff `3.43e-5`, max loss diff `1.83e-4`, max parameter-gradient diff `2.06e-4`; 8/9 checks failed the strict `1e-5` gate. Worst parameters were mainly `interactions.1.conv_tp_weights.layer0/1/2.weight`, `readouts.1.linear_2.weight`, and occasionally structured Adam-routed tensors. |

A no-Inductor FX-only control was then run with the same seed, one epoch, interval 50, and strict `EDGE_FORCE_ATOL=1e-5` diagnostic logging. Job `589415` used `EDGE_FORCE_GRAPH=False`. Compared with the Inductor job `589367`, FX-only reduced but did not eliminate the discrepancy: max energy diff `1.53e-5`, max force diff `1.53e-5`, max loss diff `6.10e-5`, max parameter-gradient diff `1.03e-4`, and 4/9 strict rows failed. Inductor therefore is not the sole source of drift; the edge-vector/bucket/trace route already has `1e-4`-scale gradient differences, and graph compilation roughly doubles the worst observed gradient discrepancy in this sample.

This narrows the seed-456 accuracy gap. The compiled edge-force path is not producing grossly wrong forces, and bucket cache reuse is value-level stable, but its training gradients are not bitwise/tight-tolerance equivalent to eager. The observed `1e-4`-scale parameter-gradient differences are small per step yet plausible enough to alter a 30-epoch HybridMuon trajectory. The next fix should therefore target numerical parity before claiming final accuracy: compare unpadded shape-cache vs bucket-cache diagnostics, inspect whether mask/padding or the pure-Aten spherical harmonics replacement is responsible for the residual FX-only gradient gap, and consider a safer mode that falls back to eager for high-sensitivity buckets or tightens the compile boundary if strict parity is required.

### Why the current end-to-end speedup is still small

A speed breakdown over the real 30-epoch RECIO/8k HybridMuon jobs shows why the current compile path is not yet a strong end-to-end win. For seed `123`, eager training took `512.2 s` from initial validation to training complete and `547.0 s` including final save/evaluation. Bucket compile took `496.7 s` and `531.8 s`, respectively. For seed `456`, eager took `509.7 s` / `544.3 s`, while bucket compile took `503.0 s` / `538.4 s`. This is only `~1-3%` end-to-end.

The hot cached per-batch numbers are better but still modest in the real multi-batch training loop: seed `123` eager median opt step was `0.03194 s`, compile hot-cache median was `0.02734 s`; seed `456` eager median was `0.03160 s`, compile hot-cache median was `0.02748 s`. That is roughly `13-15%` per cached training step, not the `2x+` fixed-batch profiler result. The fixed-batch profiler measured the most favorable part of the system; the real epoch includes varying shapes, bucket padding, loss/optimizer/logging overhead, and validation.

The main current amortization problem is compile setup. In 30 epochs there are `14250` optimizer steps and only two bucket compilations, but setup still costs `46.97-47.57 s`. The compile hot path saves roughly `50-58 s` of optimizer-step time over eager, so setup consumes most of the gain. After setup is subtracted, the cached compile path is meaningfully faster; before subtraction, the net optimizer-step saving is small. Final save/evaluation adds another `~35 s` to both eager and compile jobs, further diluting the percentage win reported by Slurm elapsed time.

The current SAI V100 environment is also not the ideal DPA4-style target. Available NVIDIA cuEquivariance ops wheels do not contain `sm70` kernels, so the tested production path is effectively no-CUEQ or cueq-minus-linear guarded behavior rather than full CUEQ+compile on a supported GPU. V100 also lacks native bf16 tensor cores. Therefore the present result should be read as a correctness and hot-path prototype on V100, not as the expected H100/H20/A100-class endpoint.

A no-eval 10 epoch paired check confirms that validation is not the main reason. Jobs `589634` (eager) and `589635` (bucket compile) used seed `456`, `EVAL_INTERVAL=999`, batch size `16`, and no CUEQ. Eager finished training in `167.0 s` and total job time after final evaluation/save in `201.9 s`; compile finished training in `202.0 s` and total in `237.1 s`. The compile run was slower because setup cost `49.922 s`. Its cached epochs were faster (`~13.1 s/epoch`) than eager's implied training epochs (`~15.3 s/epoch`), but nine hot epochs only save about `20 s`, far less than the one-time setup. This means short RECIO/8k runs need either many more epochs, much lower setup, or a cache/export path reused across runs before compile becomes an end-to-end win.

Practical implication: the next optimization target is not another optimizer tweak. It is reducing compile setup and padding overhead, or moving to a DPA4-like dynamic shape cache that compiles once per train/eval topology instead of bucket-sized executables. Without that, real RECIO/8k epoch throughput improves only modestly even though the compiled force subpath itself works.

### Dynamic cache and Inductor control

The DPA4-style dynamic cache path was then tested directly on the same RECIO/8k seed `456` HybridMuon setup. Unlike the earlier bucket path, dynamic cache uses one symbolic edge-force executable for all observed atom/edge sizes, avoids padding, and records `buckets=none`.

| Case | Job | Graph lowering | Slurm elapsed | Compile setup | Hot opt epoch | Final valid MAE E | Final valid MAE F | Test MAE E | Test MAE F |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager HybridMuon | `589187` | none | `00:09:19` | `0 s` | `~16.93 s` interval | `68.4 meV/atom` | `281.1 meV/A` | `75.8 meV/atom` | `262.8 meV/A` |
| bucket compile | `589188` | Inductor | `00:09:14` | `47.571 s` | `13.16 s` opt | `110.4 meV/atom` | `285.7 meV/A` | `116.0 meV/atom` | `269.0 meV/A` |
| dynamic compile | `589767` | Inductor | `00:06:57` | `38.152 s` | `8.44 s` opt | `115.7 meV/atom` | `287.2 meV/A` | `121.1 meV/atom` | `273.6 meV/A` |
| dynamic FX-only | `589866` | no Inductor | `00:09:07` | `6.063 s` | `14.65 s` opt | `70.4 meV/atom` | `278.3 meV/A` | `76.5 meV/atom` | `262.3 meV/A` |

This is the clearest split so far. Dynamic Inductor compile is the first real 30-epoch run with a substantial end-to-end training-speed improvement on V100: total elapsed `417 s` versus eager `559 s`, and hot optimizer epochs around `8.4 s` versus eager's `~16.9 s` epoch interval. It compiled once, then reported `cache_hits=14249`, `new_compiles=1`, `fallbacks=0`, and `runtime_recompiles=0`.

However, the same dynamic Inductor run did not preserve the HybridMuon training trajectory: final valid/test energy MAE was much worse than eager. The FX-only dynamic control keeps the same edge-force conservative rewrite, same dynamic cache policy, same HybridMuon routing, same seed, same data split, and same 14,250 real optimizer steps, but disables Inductor graph lowering. It recovered eager-level quality: valid `70.4/278.3` and test `76.5/262.3`, effectively matching eager `68.4/281.1` and `75.8/262.8`.

The current root-cause hypothesis is therefore narrow: the edge-vector conservative force path and the current conservative HybridMuon routing are not inherently responsible for the 30-epoch quality regression. The regression is most likely introduced by the Inductor-lowered dynamic graph, its fusion/reduction choices, or compiler numeric settings. Periodic parity diagnostics already showed the same pattern at smaller scale: FX-only had max parameter-gradient differences around `1.03e-4`, while Inductor roughly doubled the worst observed discrepancy to `2.06e-4`.

The next engineering step should be to keep `dynamic` as the preferred cache policy, but add a safer Inductor profile rather than falling back to bucket mode. Candidate knobs are deterministic reduction/fusion limits, disabling selected high-risk fusions, isolating the force backward region into smaller compiled callables, and keeping sensitive reductions or spherical harmonics derivatives on the FX/eager side. Acceptance should require both speed and quality: no fallback/recompile, hot opt speed close to job `589767`, and final RECIO/8k validation/test metrics close to the FX-only/eager controls.

### 2026-07-01 TF32/AMP and Inductor numeric debug update

MACE did not previously have an explicit DPA4-style training TF32 switch. A new `--train_tf32/--no-train_tf32` option now routes through `TrainingPrecisionConfig` and applies `torch.set_float32_matmul_precision("high")` only inside the training precision context, restoring the previous PyTorch setting afterwards. The benchmark script exposes this as `TRAIN_TF32=True/False`. The default remains `False` so existing baselines are unchanged. On the current visible SAI V100 partitions this is not expected to accelerate matmuls because V100 has no TF32 tensor cores; the option is present for Ampere/Hopper-class validation.

The DPA4-style partial AMP boundary remains guarded by `--train_amp_dtype=bf16`. On V100 it still fails closed through `torch.cuda.is_bf16_supported()`, which is the correct behavior for accuracy-preserving training. The intended supported-GPU policy is: keep geometry-sensitive preprocessing and normalization effectively FP32, use autocast only for CUDA training regions where PyTorch selects safe lower-precision kernels, then require RECIO/8k parity and multi-epoch accuracy checks before claiming bf16 speedup.

Three targeted Inductor numeric experiments were added after the dynamic-cache quality regression:

| Hypothesis | Job / check | Result | Interpretation |
| --- | ---: | --- | --- |
| Inductor `shape_padding` is causing drift | `591429`, dynamic Inductor with `EDGE_FORCE_SHAPE_PADDING=False` | max parameter-gradient diff `2.27e-4`, mean `8.70e-5`, 7 failed parity rows | Not an improvement over default dynamic Inductor (`1.77e-4`, mean `8.46e-5`). Shape padding is not the root cause. |
| Over-fusion contributes to drift | `591570`, dynamic Inductor with `EDGE_FORCE_MAX_FUSION_SIZE=1` | max parameter-gradient diff `1.56e-4`, mean `7.74e-5`, 6 failed parity rows | Slightly better than default but still above the strict `1e-5` gate. Fusion contributes, but limiting global fusion is not enough. |
| Polynomial SH replacement is the main issue | `591858`, dynamic Inductor with `EDGE_FORCE_SH=e3nn` | failed during `make_fx`: e3nn TorchScript spherical harmonics tried to access a FakeTensor data pointer | Original e3nn SH cannot be directly traced inside the symbolic force-backward closure. The hypothesis cannot be tested by simply swapping e3nn back into the compiled region. |

The e3nn failure is useful evidence rather than a dead end. The current polynomial spherical-harmonics path is not proven to be the quality root cause: dynamic FX-only with the same polynomial path recovered 30-epoch eager-level quality in job `589866`. The sharper conclusion is that symbolic tracing cannot include e3nn's TorchScript SH kernel as-is, while Inductor lowering of the pure-Aten dynamic force graph still introduces enough higher-order-gradient drift to move the HybridMuon trajectory. The next root-cause split should isolate sensitive geometry/SH derivatives or selected reductions outside the Inductor-lowered region, rather than reverting wholesale to e3nn inside `make_fx`.

Two real 20k-step HybridMuon benchmarks were submitted to test the safe path after this debug pass. Both use RECIO/8k, seed `456`, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size `16`, `MAX_NUM_EPOCHS=42`, `EVAL_INTERVAL=14`, no CUEQ, no TF32, and no bf16. Since the training split has 7600 structures, this is `475` updates per epoch, or `19950` optimizer steps total.

| Case | Job | Purpose |
| --- | ---: | --- |
| dynamic FX-only edge-force compile + HybridMuon | `592048` | Long-run speed/accuracy test of the currently safe compile path: dynamic cache, no padding, no Inductor lowering. |
| eager HybridMuon baseline | `592068` | Matched 20k-step baseline for end-to-end time, batch/epoch speed, and validation/test accuracy. |

Both jobs completed. They are the current acceptance gate for the safe implementation; the dynamic Inductor path remains experimental until it can match FX-only/eager RECIO/8k quality, even though it is much faster.

### 20k-step HybridMuon benchmark result

The 42-epoch RECIO/8k benchmark corresponds to `19950` optimizer updates. Both runs used the same seed, validation split, model size, loss weights, and HybridMuon routing. MACE loaded the epoch-28 checkpoint for final evaluation because the run used `EVAL_INTERVAL=14`; epochs 0, 14, and 28 are the evaluated checkpoint candidates.

| Case | Job | Slurm elapsed | Training phase | Hot epoch timing | Selected checkpoint | Valid E | Valid F | Test E | Test F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic FX-only edge-force compile | `592048` | `00:12:14` | `683.7 s` | `14.72 s` optimizer epoch mean after epoch 0 | epoch 28 | `137.8 meV/atom` | `281.6 meV/A` | `145.6 meV/atom` | `263.6 meV/A` |
| eager HybridMuon | `592068` | `00:12:33` | `703.6 s` | about `16.6 s/epoch` from eval intervals | epoch 28 | `66.5 meV/atom` | `301.7 meV/A` | `71.1 meV/atom` | `284.5 meV/A` |

The speed result is real but still modest on V100: dynamic FX-only is `19 s` faster by Slurm elapsed and `19.8 s` faster inside the training phase, about `2.5-2.8%` end-to-end/training-phase improvement. The cache behavior is clean: epoch 0 compiled once (`new_compiles=1`, `compile_setup_seconds=6.175`), then epochs 1-41 had `cache_hits=475`, `new_compiles=0`, `fallbacks=0`, and `runtime_recompiles=0`.

The accuracy result is not a simple win/loss. FX-only lands on a different energy-force Pareto point: force is better than eager by about `20.1 meV/A` on validation and `20.9 meV/A` on test, while energy is worse by about `71 meV/atom` on validation and `74.5 meV/atom` on test. Because the configured loss is `energy_weight=1`, `forces_weight=100`, the FX-only epoch-28 validation loss (`0.6056`) is lower than eager's (`0.7303`) despite worse energy. This suggests a small optimizer-trajectory change that favors the force-dominated objective, not a gross conservative-force failure. For production claims we still need repeated seeds and, ideally, a stricter energy-force balanced acceptance criterion or schedule/weight ablation.

### Adam/Muon x compile 2w-step matrix and CUEQ downgrade check

A full optimizer/compile comparison was run for the V100-executable part of the requested matrix. All jobs used RECIO/8k, seed `456`, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size `16`, `MAX_NUM_EPOCHS=42`, `EVAL_INTERVAL=14`, no TF32, no bf16, and CUEQ off. This is `19950` optimizer updates per job.

| Optimizer | Compile path | CUEQ | Job | Slurm elapsed | Training phase | Selected checkpoint | Valid E | Valid F | Test E | Test F | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adam | eager | off | `592759` | `00:12:41` | `710.3 s` | epoch 14 | `119.0 meV/atom` | `388.1 meV/A` | `124.2 meV/atom` | `379.4 meV/A` | Epoch 28 had lower energy (`101.7`) but higher force-heavy validation loss, so epoch 14 was selected. |
| Adam | dynamic FX-only force compile | off | `592762` | `00:12:05` | `675.7 s` | epoch 28 | `64.1 meV/atom` | `316.6 meV/A` | `72.6 meV/atom` | `296.9 meV/A` | One compile in epoch 0 (`6.251 s` setup); hot optimizer epoch mean `14.56 s`; no fallback/recompile. |
| HybridMuon | eager | off | `592068` | `00:12:33` | `703.6 s` | epoch 28 | `66.5 meV/atom` | `301.7 meV/A` | `71.1 meV/atom` | `284.5 meV/A` | Best balanced E/F among current V100-safe runs. |
| HybridMuon | dynamic FX-only force compile | off | `592048` | `00:12:14` | `683.7 s` | epoch 28 | `137.8 meV/atom` | `281.6 meV/A` | `145.6 meV/atom` | `263.6 meV/A` | Better force but much worse energy; force-heavy validation loss selected this trajectory. |

This answers two questions directly. First, the measured compile path is training compile acceleration for conservative force training, not inference compile. It rewrites the energy-to-force training path and reuses a symbolic force/backward closure during optimizer steps. Second, HybridMuon is compatible with this compile path in the engineering sense: the HybridMuon jobs run for `19950` updates with no fallback or recompile. The accuracy interaction is optimizer-dependent, so compatibility is not the same as "identical optimizer trajectory".

The optimizer/compile interaction is nontrivial. Adam benefits strongly from dynamic FX-only compile in this single seed: it is faster and improves both energy and force versus Adam eager. HybridMuon dynamic FX-only compile is also faster and improves force versus HybridMuon eager, but it sacrifices energy substantially. This reinforces that future acceptance should compare both E and F and should not use only force-heavy validation loss as the success criterion.

#### CUEQ on/off status and downgrade test

A CUEQ-on probe was submitted with `mace_develop_cueq`, `ENABLE_CUEQ=True`, and current `cuequivariance==0.10.0` / `cuequivariance-ops-cu12==0.10.0` / `cuequivariance-ops-torch-cu12==0.10.0`. Job `592769` failed in the initial validation forward pass at `torch.ops.cuequivariance.uniform_1d` with `cudaErrorNoKernelImageForDevice`. Inspecting the installed `libcue_ops.so` showed only `sm_100` and `sm_120` targets, no `sm_70` target for Tesla V100.

The downgrade hypothesis was tested without modifying the environment by downloading and unpacking every available `cuequivariance-ops-cu12` wheel from the configured PyPI mirror:

| `cuequivariance-ops-cu12` version | CUDA architecture strings in `libcue_ops.so` |
| --- | --- |
| `0.4.0` | `sm_90` |
| `0.5.0`, `0.5.1`, `0.6.0`, `0.6.1`, `0.7.0`, `0.8.0`, `0.8.1`, `0.9.0`, `0.9.1`, `0.10.0` | `sm_100`, `sm_120` |

`pip index versions cuequivariance-ops-cu12` lists only `0.4.0` through `0.10.0`, and `pip download --no-binary=:all:` found no source distribution for `cuequivariance-ops-cu12`. The local NVIDIA cuEquivariance clone contains the Python frontends and wrappers, but not the CUDA source for `libcue_ops.so`; the README documents that CUDA kernels are installed from separate `cuequivariance-ops-*-cu12/cu13` packages.

Therefore CUEQ-on 2w-step training is not a valid V100 benchmark on the current SAI hardware: pip downgrade cannot produce an `sm70` kernel from the available wheels. To fill the CUEQ on/off matrix, we need either a GPU supported by the NVIDIA ops wheels (`sm90`, `sm100`, `sm120` class based on the scanned wheels) or NVIDIA-provided/source-built CUEQ ops containing `sm70`. Until then, CUEQ on should remain a guarded incompatibility row, not a failed MACE training result.

### Correction: CUEQ 0.6.1 JIT backend works on V100

The earlier CUEQ conclusion was too broad because it only tested the newer `0.10.0` ops stack. The user's memory that `cueq==0.6.1` did not error on V100 is correct.

Existing environments show two distinct CUEQ backend families:

| Env | Torch | CUEQ packages | Backend behavior on V100 |
| --- | --- | --- | --- |
| `mace_develop_cueq` / `mace_torch212` | `2.10/2.11+cu128` | `cuequivariance==0.10.0`, `cuequivariance-ops-cu12==0.10.0` | Fails at `torch.ops.cuequivariance.uniform_1d` because `libcue_ops.so` contains only `sm_100/sm_120`. |
| `mace_env` | `2.8.0+cu128` | `cuequivariance==0.6.1`, `cuequivariance-ops-cu12==0.6.1`, `cuequivariance-ops-torch-cu12==0.6.1` | Uses the older `cuequivariance_ops::tensor_product_uniform_1d_jit` path. It completed V100 RECIO/8k training probes. |

The difference is not just the Python frontend version. CUEQ `0.6.1` uses a JIT tensor-product uniform-1d backend (`tensor_product_uniform_1d_jit`) which can generate `sm70` kernels at runtime. The newer `0.10.0` path calls `uniform_1d` in a precompiled `libcue_ops.so`; the available wheels do not include `sm70`.

Two short V100 probes completed successfully:

| Case | Job | Result |
| --- | ---: | --- |
| CUEQ 0.6.1 eager Adam, 1 epoch | `593646` | Completed in `00:02:09`; model converted to CUEQ, trained one epoch, exported CUEQ back to E3NN. |
| CUEQ 0.6.1 + dynamic FX-only edge-force compile, Adam, 1 epoch | `593803` | Completed in `00:02:20`; proves the safe FX-only compile path is at least smoke-compatible with CUEQ 0.6.1. |

Therefore the actionable V100 route is not "CUEQ unavailable"; it is "pin to the older CUEQ 0.6.1 JIT backend for V100, and avoid the newer 0.10.0 precompiled ops stack unless running on supported architectures." Four CUEQ-on 2w-step jobs have been submitted with `mace_env`:

| Optimizer | Compile path | CUEQ | Job |
| --- | --- | --- | ---: |
| Adam | eager | 0.6.1 on | `593905` |
| Adam | dynamic FX-only edge-force compile | 0.6.1 on | `593926` |
| HybridMuon | eager | 0.6.1 on | `593934` |
| HybridMuon | dynamic FX-only edge-force compile | 0.6.1 on | `593960` |

These runs are not perfectly apples-to-apples with the earlier CUEQ-off matrix because `mace_env` uses `torch 2.8.0+cu128`, while the CUEQ-off matrix used `mace_develop` with `torch 2.10.0+cu128`. For a strict CUEQ on/off attribution, run matching CUEQ-off jobs in `mace_env` as well, or build a single torch/CUEQ environment that supports both modes. Still, the immediate correction is clear: CUEQ 0.6.1 on V100 is viable and should be benchmarked.

### Completed 8-way 2w-step matrix with CUEQ 0.6.1

The CUEQ 0.6.1 jobs completed, so the requested optimizer/compile/CUEQ matrix now has one full seed. All rows use RECIO/8k, seed `456`, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, batch size `16`, `MAX_NUM_EPOCHS=42`, and `EVAL_INTERVAL=14`, corresponding to `19950` optimizer updates. CUEQ-off rows used `mace_develop` (`torch 2.10.0+cu128`); CUEQ-on rows used `mace_env` (`torch 2.8.0+cu128`, `cuequivariance==0.6.1`). This means CUEQ on/off speed attribution is slightly confounded by torch version, but the V100 CUEQ feasibility and end-to-end behavior are now measured.

| Optimizer | CUEQ | Compile path | Job | Slurm elapsed | Training phase | Compile setup | Hot opt epoch | Valid E | Valid F | Test E | Test F |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam | off | eager | `592759` | `00:12:41` | `710.3 s` | - | - | `119.0` | `388.1` | `124.2` | `379.4` |
| Adam | off | dynamic FX-only force compile | `592762` | `00:12:05` | `675.7 s` | `6.251 s` | `14.56 s` | `64.1` | `316.6` | `72.6` | `296.9` |
| Adam | 0.6.1 on | eager | `593905` | `00:09:50` | `543.7 s` | - | - | `79.2` | `296.6` | `88.1` | `284.1` |
| Adam | 0.6.1 on | dynamic FX-only force compile | `593926` | `00:09:36` | `530.1 s` | `67.733 s` | `9.43 s` | `80.9` | `294.0` | `88.6` | `275.8` |
| HybridMuon | off | eager | `592068` | `00:12:33` | `703.6 s` | - | - | `66.5` | `301.7` | `71.1` | `284.5` |
| HybridMuon | off | dynamic FX-only force compile | `592048` | `00:12:14` | `683.7 s` | `6.175 s` | `14.72 s` | `137.8` | `281.6` | `145.6` | `263.6` |
| HybridMuon | 0.6.1 on | eager | `593934` | `00:10:25` | `575.9 s` | - | - | `103.9` | `287.7` | `113.2` | `272.4` |
| HybridMuon | 0.6.1 on | dynamic FX-only force compile | `593960` | `00:09:42` | `534.9 s` | `67.382 s` | `9.57 s` | `105.1` | `299.7` | `114.0` | `284.9` |

Observations:

- CUEQ 0.6.1 is the largest practical V100 speed lever tested so far. Adam eager improves from `710.3 s` training phase to `543.7 s`; HybridMuon eager improves from `703.6 s` to `575.9 s`.
- FX-only force compile remains compatible with CUEQ 0.6.1, but setup cost is much higher with CUEQ (`~67 s`) than without CUEQ (`~6 s`). After setup, the hot optimizer epoch is much faster with CUEQ+compile (`~9.4-9.6 s`) than CUEQ eager implied epoch timing, but at `19950` updates the end-to-end gain is still modest for Adam (`00:09:50 -> 00:09:36`) and larger for HybridMuon (`00:10:25 -> 00:09:42`). Longer runs or cache reuse would amortize the CUEQ+compile setup better.
- Accuracy is optimizer-dependent. Adam+CUEQ is a strong result in this seed: much faster than Adam no-CUEQ and with much better force/energy than Adam eager no-CUEQ. HybridMuon+CUEQ improves speed and force relative to HybridMuon eager no-CUEQ, but energy is worse than HybridMuon no-CUEQ eager. HybridMuon+CUEQ+compile is fastest among the HybridMuon rows but has worse force than HybridMuon+CUEQ eager in this seed.
- The current best balanced accuracy row is still HybridMuon no-CUEQ eager (`71.1 meV/atom`, `284.5 meV/A` test) or Adam+CUEQ+compile for force-biased speed (`88.6 meV/atom`, `275.8 meV/A` test). The current fastest training row is Adam+CUEQ+compile (`530.1 s` training phase), closely followed by HybridMuon+CUEQ+compile (`534.9 s`).

Practical next step: keep CUEQ pinned to `0.6.1` on V100, and benchmark whether the `~67 s` compile setup can be reduced or cached. For CUEQ 0.10+, do not use V100 unless an `sm70` wheel/source build is available.

## RECIO/8k 20k-Step Ablation: Adam/Muon x CUEQ x Edge-Force Compile

Latest authoritative run directory: `runs/recio8k_ablation_20k_tace_muon_compile_cueq`. The latest-job-only machine summary is stored in `summary_latest_jobs.json` and `summary_latest_jobs.csv`; this avoids mixing stale resubmission logs from earlier failed CUEQ attempts.

Environment and model: `mace_develop`, SAI `4V100PX`, one V100 GPU, RECIO `8k/train.xyz`, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `num_interactions=2`, `correlation=3`, batch size `16`, 43 epochs, about 20,425 optimizer updates, `train_tf32=True`, `train_amp_dtype=none`. CUEQ cases used `optimize_channelwise=True`, `optimize_symmetric=True`, `conv_fusion=True`, with `optimize_linear=False` and `optimize_fctp=False` for the current stable training profile. HybridMuon cases used `hybrid_muon_mode=slice` and `hybrid_muon_routing=tace`.

| Case | Job | Wall time | Max RSS | Hot opt step | Final test E | Final test F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adam_nocueq_eager` | `598537` | `13:25` | `1980 MB` | n/a | `28.1 meV/atom` | `203.1 meV/A` |
| `adam_nocueq_compile` | `598538` | `13:28` | `2070 MB` | `14.63 s/epoch` | `33.8 meV/atom` | `204.3 meV/A` |
| `adam_cueq_eager` | `598832` | `10:59` | `2999 MB` | n/a | `37.5 meV/atom` | `239.1 meV/A` |
| `adam_cueq_compile` | `598884` | `10:48` | `3613 MB` | `9.57 s/epoch` | `42.2 meV/atom` | `238.7 meV/A` |
| `muon_nocueq_eager` | `598541` | `14:37` | `2002 MB` | n/a | `77.7 meV/atom` | `328.9 meV/A` |
| `muon_nocueq_compile` | `598542` | `14:00` | `2111 MB` | `15.74 s/epoch` | `54.0 meV/atom` | `327.2 meV/A` |
| `muon_cueq_eager` | `598834` | `12:13` | `2973 MB` | n/a | `74.3 meV/atom` | `368.3 meV/A` |
| `muon_cueq_compile` | `598885` | `11:35` | `3688 MB` | `10.62 s/epoch` | `73.2 meV/atom` | `368.6 meV/A` |

Acceptance facts:

- CUEQ is now installed in `mace_develop` (`cuequivariance==0.6.1`, `cuequivariance-torch==0.6.1`) and the 20k-step CUEQ eager jobs completed.
- CUEQ plus edge-force compile now completes real RECIO training. The compile summaries report `compiled=475/475`, `fallbacks=0`, and `runtime_recompiles=0` for every epoch after the single initial compile.
- The root cause of the earlier CUEQ+compile crash was that symbolic edge-force compile disabled e3nn TorchScript codegen only around the original MACE construction, while CUEQ conversion built a fresh target model outside that context. The target retained e3nn `Linear._compiled_main`, and `make_fx` failed under FakeTensor tracing. `run_train.py` now wraps CUEQ/OEQ conversion in the same model-build context.
- Another startup bug was fixed in `InteractionBlock`: `conv_fusion` is now set only when the selected backend is actually enabled. This prevents a requested-but-unavailable CUEQ backend from forcing ordinary e3nn `TensorProduct` through the fused 4-argument call path.
- TACE-style HybridMuon now covers nearly all trainable MACE matrix parameters in this model after CUEQ conversion: the logged route was `234112` Muon parameters and `1424` Adam parameters. This is structurally much closer to TACE than the previous conservative route.

Current negative/neutral findings:

- End-to-end compile gain is small after CUEQ is already enabled: Adam CUEQ eager `10:59` to Adam CUEQ compile `10:48`, and Muon CUEQ eager `12:13` to Muon CUEQ compile `11:35`. The hot training path is faster, but the first compile/parity setup is about `69 s`, and final evaluation/checkpoint conversion dominates a short 43-epoch run.
- CUEQ improves end-to-end speed on this benchmark, but the current CUEQ profile has an accuracy regression relative to no-CUEQ Adam: final test force MAE is about `239 meV/A` for Adam+CUEQ versus `203 meV/A` for Adam no-CUEQ. This must be investigated before the current CUEQ training profile is treated as accuracy-safe.
- The current broad TACE-style HybridMuon route is not accepted for accuracy. It trains stably and is compatible with compile/CUEQ, but its final force MAE is much worse than Adam in this short 20k-step RECIO schedule. Next tests should reduce the Muon LR factor, revisit decoupled weight decay, and ablate which equivariant/path blocks are safe to route to Muon.
- On V100, `train_amp_dtype=bf16` is still not an acceleration claim. The benchmark used TF32-style matmul precision (`torch.set_float32_matmul_precision("high")`) and FP32 training; bf16 should be gated on native bf16 hardware or treated only as a precision-safety experiment here.

Next acceptance gates:

1. Run a CUEQ/e3nn parity and train-step gradient parity check for the exact stable CUEQ profile used here, because the 20k training accuracy regression suggests either numerical/semantic differences or an optimization-dynamics issue.
2. Add a longer schedule after CUEQ parity is clean; 43 epochs is enough for speed and smoke accuracy but not for final precision conclusions.
3. Tune HybridMuon conservatively: keep slice/path-aware routing, but compare Adam fallback for symmetric contraction weights, product linears, and skip tensor products separately before accepting the broad TACE-style coverage.
4. For compile speed, prioritize reducing the initial `gate_reference/gate_candidate` setup cost and moving final evaluation/checkpoint conversion out of short benchmark timing, but do not weaken parity checks until accuracy parity is established.



## Post-Parity Correction: CUEQ Profile Must Be Self-Consistent

A follow-up CUEQ/e3nn force-loss parity sweep found that the previous partial CUEQ profile was not a safe baseline. The problematic profile was `optimize_channelwise=True`, `optimize_symmetric=True`, `conv_fusion=True`, with `optimize_linear=False` and `optimize_fctp=False`. On a real RECIO batch it produced e3nn-vs-CUEQ differences around `9.7e-4 eV` in energy and `5.9e-3 eV/A` in forces before any training dynamics. Therefore the CUEQ accuracy regression in the 20k matrix should not be interpreted as an optimizer or compile effect. It was at least partly a CUEQ conversion/layout-profile issue.

Two concrete bugs were fixed:

- `convert_e3nn_cueq.transfer_weights()` no longer drops ordinary `symmetric_contractions.*` keys when the target model is not using optimized CUEQ symmetric contraction. The no-optimized converted model now preserves ordinary MACE symmetric contraction weights.
- Interaction-block `reshape_irreps` no longer switches to CUEQ layout merely because `cueq_config.enabled=True`. The ordinary no-optimized converted model now keeps the e3nn/MACE product layout and passes force-loss parity.

After these fixes, the CUDA parity sweep gives:

| CUEQ profile | Result | Key parity numbers | Interpretation |
| --- | --- | --- | --- |
| all optimize flags off | pass | e3nn-vs-target `E=1.2e-7`, `F=7.6e-8`, no grad failures | conversion baseline is now correct |
| previous partial profile | fail | `E~9.7e-4`, `F~5.9e-3` | not accuracy-safe |
| symmetric only | fail | `E~2.0e-1`, `F~2.0e-1` | current symmetric-only transfer/profile is not valid |
| channelwise/linear/fctp only | runtime layout error | ordinary product receives the wrong layout | partial CUEQ needs explicit adapters before use |
| optimize_all | scalar parity pass | `E<=4.8e-7`, `F<=8.4e-8`; parameter names differ because cuet modules merge/rewrite weights | current safest accelerated CUEQ profile |

A real RECIO/8k smoke run with `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `train_tf32=True`, and edge-force compile completed successfully: job `599165`, directory `runs/recio8k_smoke_cueq_optall_compile_20260702/adam_cueq_compile`. It trained two epochs on the 8k data. Compile summaries showed epoch 0 `compiled=475/475`, `fallbacks=0`, `runtime_recompiles=0`, `compile_setup_seconds=67.276`, `opt_step_seconds=77.501`; epoch 1 had `compiled=475/475`, `new_compiles=0`, `fallbacks=0`, `runtime_recompiles=0`, and `opt_step_seconds=10.158`.

Actionable correction: future CUEQ-on RECIO benchmarks should use the self-consistent `optimize_all` profile unless explicit layout adapters are added and parity-tested for partial profiles. The old 20k CUEQ partial-profile accuracy rows remain useful as negative evidence, but they should not be used as accepted CUEQ accuracy measurements.


### Corrected 20k CUEQ optimize_all matrix

After the CUEQ profile correction, four real RECIO/8k 20k-step jobs were rerun with `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `optimize_linear=True`, `optimize_channelwise=True`, `optimize_symmetric=True`, `optimize_fctp=True`, `conv_fusion=True`, `train_tf32=True`, `train_amp_dtype=none`, seed `456`, batch size `16`, and the same 43-epoch schedule as the earlier matrix. Run directory: `runs/recio8k_ablation_20k_cueq_optall_fix_20260702`.

| Case | Job | Wall time | Max RSS | Compile behavior | Final test E | Final test F |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `adam_cueq_eager` | `599220` | `11:25` | `3151 MB` | n/a | `28.3 meV/atom` | `201.5 meV/A` |
| `adam_cueq_compile` | `599221` | `11:21` | `3259 MB` | epoch 0 setup `67.255 s`; hot opt epochs mostly `10.24-10.47 s`; `fallbacks=0`, `runtime_recompiles=0` | `29.3 meV/atom` | `203.3 meV/A` |
| `muon_cueq_eager` | `599222` | `12:35` | `3105 MB` | n/a | `33.7 meV/atom` | `243.2 meV/A` |
| `muon_cueq_compile` | `599223` | `11:45` | `3691 MB` | epoch 0 setup `66.776 s`; hot opt epochs mostly `10.82-11.10 s`; `fallbacks=0`, `runtime_recompiles=0` | `36.8 meV/atom` | `242.4 meV/A` |

This corrects the earlier CUEQ interpretation. The previous partial profile produced Adam+CUEQ test force MAE around `239 meV/A`; with the self-consistent `optimize_all` profile, Adam+CUEQ returns to about `201-203 meV/A`, matching or slightly improving the no-CUEQ Adam baseline from job `598537` (`203.1 meV/A`). The issue was not CUEQ itself or force-backward compile; it was the unsafe partial CUEQ layout/conversion profile.

Compile compatibility is now solid for the corrected CUEQ profile: both compile jobs completed all epochs with `compiled=475/475` and no fallback/recompile after the single initial compile. End-to-end compile speedup remains modest because the initial compile/parity gate costs about `67 s`: Adam improves only `11:25 -> 11:21`, while Muon improves `12:35 -> 11:45`. The hot epoch path is fast enough to matter, but the setup cost still dominates 43-epoch RECIO/8k jobs.

HybridMuon is now more credible than in the partial-profile matrix, but still not accepted as a final optimizer recipe. Under CUEQ optimize_all it reaches `~242-243 meV/A` test force, much better than the partial-profile Muon+CUEQ rows (`~368 meV/A`) but still worse than Adam+CUEQ (`~201-203 meV/A`). Its routing under CUEQ optimize_all sends 111,232 parameters to Muon and 124,304 to Adam; most cuet linear/FCTP parameters appear as effective-rank-one flattened tensors and currently stay Adam-routed. Further TACE/DPA4-inspired Muon work should tune which CUEQ symmetric/product/radial blocks are genuinely safe, rather than just increasing coverage.


### HybridMuon LR-Factor Ablation Under CUEQ Optimize-All

To test whether the remaining HybridMuon accuracy gap was mainly caused by an overly large Muon update scale, four additional real RECIO/8k 20k-step jobs were run with the same `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `train_tf32=True`, `train_amp_dtype=none`, seed `456`, batch size `16`, and 43-epoch schedule. The only changed optimizer variable was `hybrid_muon_lr_factor`; routing stayed `hybrid_muon_mode=slice`, `hybrid_muon_routing=tace`. Run directories: `runs/recio8k_muon_lr005_cueq_optall_20260702` and `runs/recio8k_muon_lr003_cueq_optall_20260702`.

| Case | Job | Wall time | Max RSS | Compile behavior | Final test E | Final test F |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `muon_cueq_eager`, factor `0.05` | `599346` | `12:04` | `3160 MB` | n/a | `38.2 meV/atom` | `254.1 meV/A` |
| `muon_cueq_compile`, factor `0.05` | `599347` | `11:54` | `3286 MB` | epoch 0 setup `67.176 s`; epoch 42 hot opt `11.064 s`; `fallbacks=0`, `runtime_recompiles=0` | `38.3 meV/atom` | `254.7 meV/A` |
| `muon_cueq_eager`, factor `0.03` | `599348` | `12:33` | `3106 MB` | n/a | `37.9 meV/atom` | `260.5 meV/A` |
| `muon_cueq_compile`, factor `0.03` | `599349` | `11:44` | `3681 MB` | epoch 0 setup `66.813 s`; epoch 42 hot opt `10.841 s`; `fallbacks=0`, `runtime_recompiles=0` | `38.6 meV/atom` | `260.3 meV/A` |

This is useful negative evidence. Reducing the Muon learning-rate factor from `0.1` to `0.05` or `0.03` does not close the gap to Adam+CUEQ; it makes force MAE worse than the factor-`0.1` corrected CUEQ runs (`~242-243 meV/A`). Compile remains compatible with HybridMuon and CUEQ in all four runs, but it does not change the optimizer accuracy trend.

The next Muon work should therefore not be a simple LR-factor search. The more likely issues are block routing and update recipe details: whether optimized symmetric-contraction weights should be Adam-routed or damped, whether DPA4's Magma-lite/AdamW fallback behavior should be ported more faithfully, and whether MACE needs explicit irrep/path metadata instead of relying only on TACE-style effective shape routing.


### Radial-Only HybridMuon Routing Ablation

The next routing ablation kept the same real RECIO/8k 20k-step setup and CUEQ `optimize_all=True`, but changed HybridMuon from broad TACE-style routing to conservative MACE routing: `hybrid_muon_mode=2d`, `hybrid_muon_routing=mace`, `hybrid_muon_lr_factor=0.1`. This routes only the dense radial `conv_tp_weights` MLP matrices to Muon and sends `products.*.symmetric_contractions.weight` back to Adam. Run directory: `runs/recio8k_muon_radialonly_cueq_optall_20260702`.

The logged route was 8 Muon tensors / 74,752 parameters and 14 Adam tensors / 160,784 parameters. The two optimized symmetric-contraction weights were Adam-routed:

```text
Adam: products.0.symmetric_contractions.weight shape=(5, 86, 64) reason=sensitive-name
Adam: products.1.symmetric_contractions.weight shape=(5, 28, 64) reason=sensitive-name
```

| Case | Job | Wall time | Max RSS | Compile behavior | Final test E | Final test F |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `muon_cueq_eager`, radial-only | `599492` | `11:53` | `3269 MB` | n/a | `31.3 meV/atom` | `222.9 meV/A` |
| `muon_cueq_compile`, radial-only | `599493` | `11:40` | `3381 MB` | epoch 0 setup `67.311 s`; epoch 42 hot opt `10.715 s`; `fallbacks=0`, `runtime_recompiles=0` | `35.7 meV/atom` | `223.8 meV/A` |

This is the strongest HybridMuon routing evidence so far. Moving symmetric-contraction tensors back to Adam improves force MAE from broad TACE-style `~242-260 meV/A` to `~223 meV/A`, while compile and eager agree closely on force. The remaining gap to Adam+CUEQ (`~201-203 meV/A`) means radial-only Muon is not yet a final replacement for Adam, but it is a much safer candidate than broad TACE-style slice routing for current MACE/CUEQ.

The immediate conclusion is structural: MACE's optimized symmetric-contraction tensors should not be routed by a generic `effective_shape + slice` rule as if they were ordinary DPA4/TACE multiplicity matrices. Either keep them on Adam by default, or add explicit MACE path/irrep metadata plus block health damping before attempting Muon there again. The next optimizer implementation work should therefore be a named conservative default, for example `hybrid_muon_routing=mace` or a new `mace_safe` preset, plus optional experimental `tace` routing kept behind an explicit flag.

The same structural conclusion was re-tested with `hybrid_muon_mode=slice`, `hybrid_muon_routing=mace`, which keeps the radial MLP matrices on Muon and additionally routes the optimized MACE symmetric-contraction tensors as independent slices. Both real RECIO/8k jobs used CUEQ `optimize_all=True`, TF32, no AMP, seed `456`, batch size `16`, 43 epochs/about 20,425 optimizer updates, `hybrid_muon_lr_factor=0.1`, and `mace_develop`. Run directory: `runs/recio8k_muon_mace_slice_cueq_optall_20k_20260702`.

The route summary was 10 Muon tensors / 111,232 parameters and 12 Adam tensors / 124,304 parameters:

```text
Muon: products.0.symmetric_contractions.weight shape=(5, 86, 64) reason=equivariant-slice-muon mode=slice matrix_batch=5 matrix_shape=(86, 64)
Muon: products.1.symmetric_contractions.weight shape=(5, 28, 64) reason=equivariant-slice-muon mode=slice matrix_batch=5 matrix_shape=(28, 64)
```

| Case | Job | Slurm state | Elapsed | MaxRSS | Mean opt time after epoch 0 | Compile behavior | Stage-two test E | Stage-two test F |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `muon_maceslice_cueq_eager` | `604708` | `COMPLETED` | `00:12:45` | `3654540K` | `25.46 ms/batch` | n/a | `34.1 meV/atom` | `242.6 meV/A` |
| `muon_maceslice_cueq_compile` | `604709` | `COMPLETED` | `00:12:02` | `3767944K` | `23.27 ms/batch` | epoch 0 setup `67.790 s`; hot opt mean `11.053 s/epoch`; `fallbacks=0`, `runtime_recompiles=0`; final cache hits `20424` | `34.7 meV/atom` | `243.1 meV/A` |

This is a direct negative ablation for expanding the current MACE route into symmetric contractions. Compile and eager remain compatible and close in final force (`242.6` versus `243.1 meV/A`), and compile gives a modest hot-step improvement (`25.46` to `23.27 ms/batch`, about `9%`). The optimizer quality, however, falls back to the broad TACE-style result range and is much worse than the radial-only route (`~223 meV/A`) and Adam+CUEQ (`~201-203 meV/A`) at the same 20k scale. Therefore `products.*.symmetric_contractions.weight` should stay Adam-routed by default until MACE exposes explicit path/irrep block metadata and the new block route passes full RECIO validation.


### Completed 200k-Step CUEQ Long-Run Matrix

A longer RECIO/8k CUEQ matrix was run to test whether compile setup cost amortizes and whether radial-only HybridMuon catches up to or diverges from Adam over a longer schedule. All four jobs used `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `train_tf32=True`, `train_amp_dtype=none`, seed `456`, batch size `16`, `max_num_epochs=422`, `start_swa=316`, and about 200,450 optimizer updates. HybridMuon rows used the radial-only safe route: `hybrid_muon_mode=2d`, `hybrid_muon_routing=mace`, `hybrid_muon_lr_factor=0.1`.

Run directory: `runs/recio8k_ablation_200k_cueq_radialonly_20260702`.

| Case | Job | Slurm state | Elapsed | MaxRSS | Epoch-420 valid E | Epoch-420 valid F | Stage-two test E | Stage-two test F |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `adam_cueq_eager` | `599551` | `COMPLETED` | `01:27:34` | `3534368K` | `22.40 meV/atom` | `241.66 meV/A` | `19.1 meV/atom` | `137.2 meV/A` |
| `adam_cueq_compile` | `599552` | `COMPLETED` | `01:23:49` | `4367040K` | `20.83 meV/atom` | `243.12 meV/A` | `18.5 meV/atom` | `136.7 meV/A` |
| `muon_cueq_eager`, radial-only | `599553` | `COMPLETED` | `01:35:17` | `3410876K` | `22.36 meV/atom` | `238.38 meV/A` | `21.7 meV/atom` | `147.0 meV/A` |
| `muon_cueq_compile`, radial-only | `599554` | `COMPLETED` | `01:26:07` | `4763004K` | `20.02 meV/atom` | `239.48 meV/A` | `19.6 meV/atom` | `147.4 meV/A` |

Compile timing and health:

| Case | Initial compile setup | Mean hot epoch opt step | Hot epoch min/max | Fallbacks / runtime recompiles |
| --- | ---: | ---: | ---: | ---: |
| `adam_cueq_compile` | `67.524 s` | `10.220 s` | `10.188 / 10.417 s` | `0 / 0` |
| `muon_cueq_compile`, radial-only | `67.024 s` | `10.548 s` | `10.487 / 10.803 s` | `0 / 0` |

The long-run result changes the short-20k interpretation in two ways. First, compile is now a real end-to-end training speedup under CUEQ `optimize_all`, not just a hot-step profiler win: Adam improves from `01:27:34` to `01:23:49` (about `4.3%` wall-time), and radial-only HybridMuon improves from `01:35:17` to `01:26:07` (about `9.6%` wall-time). This is still far smaller than DPA4's compile+bf16 reports because this V100 run has no native bf16 speed path and the CUEQ-accelerated MACE hot path is already short.

Second, radial-only HybridMuon is stable and compatible with CUEQ+compile in a 200k-step real RECIO run, but it is not yet better than Adam as a final optimizer recipe. It stays competitive through validation (`~239 meV/A` force at epoch 420 versus Adam `~242-243 meV/A`), but the stage-two test force remains about `10 meV/A` worse than Adam (`147 meV/A` versus `137 meV/A`). This supports keeping the conservative radial route as a safe baseline while further TACE/DPA4-style Muon expansion should be block-aware and separately ablated, not enabled broadly by default.

A setup-gate control was added after this run to separate safety policy from setup-cost experiments: `--edge_force_compile_setup_gate=strict` keeps the existing first-compile reference/candidate equivalence gate and remains the default, while `--edge_force_compile_setup_gate=none` skips that expensive setup gate and reports `edge_force_gate_accepted=None`. The non-strict mode is only for guarded benchmark runs that already have strict parity evidence and should be paired with cache-hit or periodic parity diagnostics before being used for accuracy claims. The SAI benchmark wrapper accepts this as `EDGE_FORCE_SETUP_GATE=none`.

The control was validated on real RECIO/8k CUEQ compile jobs rather than only a toy batch. A two-epoch smoke run, job `600212`, used `EDGE_FORCE_SETUP_GATE=none` with Adam+CUEQ+compile and completed in `00:03:31`. Its epoch-0 compile summary reported `compile_setup_seconds=1.919`, with `setup_phases=input_prep:0.000, trace:1.918, gate_compile:0.000`, `fallbacks=0`, and `runtime_recompiles=0`. This confirms that the expensive strict setup reference/candidate gate is what caused the previous `~67 s` setup number.

Two 20k-step setup-none validation jobs then completed under the same RECIO/8k, CUEQ `optimize_all`, TF32, batch-16, seed-456 schedule:

| Case | Job | Slurm state | Elapsed | MaxRSS | Compile behavior | Stage-two test E | Stage-two test F |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `adam_cueq_compile`, `setup_gate=none` | `600246` | `COMPLETED` | `00:11:19` | `3322128K` | epoch 0 setup `1.876 s`; hot opt epochs mostly `10.23-10.47 s`; `fallbacks=0`, `runtime_recompiles=0` | `29.1 meV/atom` | `203.5 meV/A` |
| `muon_cueq_compile`, radial-only, `setup_gate=none` | `600247` | `COMPLETED` | `00:11:41` | `3413164K` | epoch 0 setup `1.866 s`; hot opt epochs mostly `10.84-11.05 s`; `fallbacks=0`, `runtime_recompiles=0` | `32.3 meV/atom` | `222.3 meV/A` |

Accuracy is consistent with the strict-gate 20k references: Adam strict job `599221` ended at `29.3 meV/atom`, `203.3 meV/A`, and radial-only Muon strict job `599493` ended at `35.7 meV/atom`, `223.8 meV/A`. Wall time barely changed despite setup dropping from `~67 s` to `~1.9 s` (`599221` `00:11:21` versus `600246` `00:11:19`; `599493` `00:11:40` versus `600247` `00:11:41`). The likely reason is that these short 43-epoch Slurm measurements still include a slow first optimizer epoch, final evaluation/export, and scheduler/accounting overhead. Therefore `setup_gate=none` is useful for debugging and isolating strict parity-gate cost, but it should not distract from the main optimization target: reducing the real hot force-backward step and maintaining accuracy under CUEQ/HybridMuon.


### HybridMuon Implementation Follow-Up

A second source pass over DeepMD/TACE and the new `review/` notes supports a conservative optimizer direction. DPA4 gets high Muon coverage because its architecture stores most learnable transformations as legal per-degree or per-SO2-stratum channel matrices. Current MACE/CUEQ does not: the extra MACE tensors that TACE-style broad slice routing can reach are structured path/species/contraction weights, and the real RECIO 20k/200k results above show that broad routing is stable but not accuracy-competitive with Adam. Therefore `hybrid_muon_routing=mace` remains the safer default.

The optimizer implementation now has an opt-in DPA4/TACE-style Magma-lite damping switch, `--hybrid_muon_magma_lite`. It computes a per-matrix or per-slice EMA of momentum-gradient cosine alignment and scales Muon updates continuously in `[0.1, 1.0]`; Adam-routed parameters and the default radial-only route are unchanged unless the flag is set. Focused tests cover the new CLI flag, optimizer-group wiring, DPA4-compatible momentum-buffer semantics, and damping of a misaligned Muon block. The next real benchmark should compare radial-only HybridMuon with and without `--hybrid_muon_magma_lite` under CUEQ `optimize_all` and edge-force compile, before considering any wider TACE-style routing.


That 20k benchmark has now been run. Both jobs used `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `train_tf32=True`, `train_amp_dtype=none`, seed `456`, batch size `16`, radial-only MACE routing, and `--hybrid_muon_magma_lite`. Run directory: `runs/recio8k_muon_magma_cueq_optall_20k_20260702`.

| Case | Job | Slurm state | Elapsed | MaxRSS | Compile behavior | Epoch-42 valid E | Epoch-42 valid F | Stage-two test E | Stage-two test F |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `muon_cueq_eager_magma`, radial-only | `600548` | `COMPLETED` | `00:11:56` | `3264824K` | n/a | `27.85 meV/atom` | `249.85 meV/A` | `31.4 meV/atom` | `224.5 meV/A` |
| `muon_cueq_compile_magma`, radial-only | `600549` | `COMPLETED` | `00:12:00` | `3478324K` | setup `67.932 s`; hot opt mean `11.117 s`; `fallbacks=0`, `runtime_recompiles=0` | `28.25 meV/atom` | `249.53 meV/A` | `31.7 meV/atom` | `224.5 meV/A` |

Magma-lite is stable and compile-compatible, but it does not close the optimizer gap. Compared with the no-Magma radial-only 20k references (`599492` eager: `31.3 meV/atom`, `222.9 meV/A`; `599493` compile: `35.7 meV/atom`, `223.8 meV/A`), Magma-lite makes compile and eager agree better on energy but leaves force essentially unchanged or slightly worse. It remains far behind Adam+CUEQ at the same 20k scale (`~201-203 meV/A`). This means the next accepted optimizer direction should not be broader generic Muon coverage; it should either keep radial-only Muon as an experimental recipe, or design explicit MACE block metadata for any new equivariant Muon route and require full RECIO validation before adoption.

### 2026-07-02 Spherical-Harmonics and Inductor Parity Split

The periodic parity metrics were not initially visible in epoch summaries, so the training loop now accumulates and logs them in the `Edge-force compile epoch ... summary` line: number of parity checks, accepted/failed counts, max energy/force/loss/parameter-gradient differences, worst gradient name, and the first failed check names. This is logging only; it does not change the training loss or optimizer step. The focused regression test is `tests/test_compile.py::test_train_one_epoch_logs_edge_force_parity_summary`, and the local check passed together with the existing parity and nonblocking-transfer tests.

Three RECIO/8k one-epoch diagnostics then isolated the spherical-harmonics and Inductor effects under the same seed-456 HybridMuon setup, `EDGE_FORCE_PARITY_CHECK_INTERVAL=50`, strict `1e-5` tolerance, no CUEQ, and no bf16:

| Case | Job | Result |
| --- | ---: | --- |
| polynomial spherical harmonics, FX-only | `600730` | Completed. `475/475` compiled steps, one dynamic trace, no fallback/recompile. Parity: `9` checks, `6` accepted, `3` failed; max energy `1.526e-05`, max forces `9.537e-06`, max loss `4.578e-05`, max parameter-gradient diff `4.578e-05`. |
| e3nn spherical harmonics, FX-only | `600649` | Failed during `make_fx`: e3nn's TorchScript `_spherical_harmonics` path tried to access a FakeTensor data pointer. This confirms that e3nn SH cannot simply be placed inside the symbolic force-backward closure. |
| polynomial spherical harmonics, Inductor graph lowering | `600752` | Completed. `475/475` compiled steps, one dynamic trace, no fallback/recompile. Parity: `9` checks, only `1` accepted; max energy `6.104e-05`, max forces `3.004e-05`, max loss `9.918e-05`, max parameter-gradient diff `1.144e-04`. |

This supports the user's suspicion in a narrower form. The pure-Aten polynomial spherical-harmonics replacement is not bitwise equivalent to the eager/e3nn route, but in FX-only mode its force discrepancy is still within `1e-5` and the worst parameter-gradient drift is around `4.6e-5` in this run. Inductor lowering amplifies the same path to `1e-4`-scale gradient drift and fails most strict parity rows, while also increasing setup and memory for this one-epoch diagnostic (`~38 s` setup and `~5.1 GB` MaxRSS versus `~6 s` and `~3.3 GB` for FX-only).

The accepted V100-safe compile mode is therefore still `EDGE_FORCE_CACHE_POLICY=dynamic`, `EDGE_FORCE_GRAPH=False`, and `EDGE_FORCE_SH=polynomial`, ideally with periodic parity checks enabled for benchmark runs. Full Inductor graph lowering remains experimental until it can pass the same real-batch parity and RECIO/8k accuracy gates. A future safer Inductor design should isolate the sensitive geometry/spherical-harmonics or reduction pieces instead of tracing e3nn SH directly into `make_fx`, because the direct e3nn path is currently incompatible with FakeTensor tracing.

### 2026-07-02 Safe Compile Defaults and WSD Scheduler Fairness Gate

The RECIO SAI wrapper now follows the current accepted V100-safe compile path by default: `EDGE_FORCE_CACHE_POLICY=dynamic`, `EDGE_FORCE_GRAPH=False`, symbolic tracing, polynomial spherical harmonics, no eager fallback, and strict setup gating unless explicitly changed. Enabling full Inductor graph lowering now requires `EDGE_FORCE_GRAPH=True` plus `EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True`, because the latest real-batch diagnostics show that Inductor amplifies parameter-gradient drift relative to FX-only. This prevents the common benchmark entry point from silently using a faster but accuracy-unaccepted path.

A DPA4-style opt-in `WSD` scheduler was added to MACE for Adam/Muon fairness ablations. It is exposed as `--scheduler=WSD` with `--lr_wsd_warmup_steps`, `--lr_wsd_warmup_ratio`, `--lr_wsd_warmup_start_factor`, `--lr_wsd_stop_lr_ratio`, `--lr_wsd_decay_phase_ratio`, and `--lr_wsd_decay_type={inverse_linear,cosine,linear}`. The default WSD recipe mirrors the DPA4 review notes at the level supported by MACE's current scheduler hook: warmup ratio `0.03`, stop LR ratio `1e-3`, decay phase ratio `0.1`, and inverse-linear decay. Because MACE's scheduler is currently epoch-level, this is an epoch-ratio WSD implementation; a future training-loop refactor can move it to true per-optimizer-step WSD.

This matters for interpreting the Adam vs HybridMuon results. The completed 20k/200k RECIO runs show that Adam+CUEQ remains the strongest tested final-force recipe, while radial-only HybridMuon is stable but not yet accuracy-superior. However, those results were not a full DPA4/TACE recipe comparison because they used the existing MACE LR/SWA flow rather than WSD. The next fair optimizer benchmark should therefore cross `optimizer={adam,hybrid_muon}` with `scheduler={ReduceLROnPlateau,WSD}` under the already accepted CUEQ 0.6.1 + dynamic FX-only compile path.

The focused local verification for this change passed: `tests/test_lr_scheduler.py`, `tests/test_recio8k_scaling_scripts.py`, and `tests/test_compile.py::test_arg_parser_accepts_edge_force_compile_flags` reported `12 passed`. The WSD scheduler is checkpointable via `state_dict/load_state_dict`, and the RECIO wrapper/static generator tests now assert that the benchmark scripts record scheduler settings and default to FX-only dynamic compile.

### 2026-07-02 WSD Scheduler Adam vs HybridMuon Result

The first real WSD fairness check has completed on RECIO/8k. Both jobs used `mace_develop`, CUEQ `0.6.1`, `optimize_all=True`, `train_tf32=True`, dynamic FX-only edge-force compile, strict setup gating, seed `456`, batch size `16`, 43 epochs/about 20k optimizer updates, and `--scheduler=WSD`. HybridMuon used the conservative radial-only MACE route: `hybrid_muon_mode=2d`, `hybrid_muon_routing=mace`, `hybrid_muon_lr_factor=0.1`. Run directory: `runs/recio8k_scheduler_wsd_cueq_compile_20k_20260702`.

| Case | Job | Slurm state | Elapsed | MaxRSS | Compile behavior | Stage-one test E | Stage-one test F | Stage-two test E | Stage-two test F |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `adam_cueq_compile`, WSD | `601110` | `COMPLETED` | `00:11:20` | `3326520K` | epoch 0 setup `68.165 s`; hot epochs mostly `10.18-10.37 s`; `fallbacks=0`, `runtime_recompiles=0` | `154.9 meV/atom` | `336.6 meV/A` | `30.2 meV/atom` | `213.6 meV/A` |
| `muon_cueq_compile`, radial-only, WSD | `601111` | `COMPLETED` | `00:12:17` | `3595180K` | epoch 0 setup `69.454 s`; hot epochs mostly `10.91-11.77 s`; `fallbacks=0`, `runtime_recompiles=0` | `98.4 meV/atom` | `282.0 meV/A` | `38.8 meV/atom` | `231.6 meV/A` |

This answers the scheduler-fairness concern in a more nuanced way. WSD materially changes the early optimizer behavior: by the stage-one checkpoint, radial-only HybridMuon is clearly ahead of Adam (`282.0 meV/A` versus `336.6 meV/A` test force). After MACE's stage-two/SWA phase, however, Adam becomes better (`213.6 meV/A` versus `231.6 meV/A`). Therefore the previous Adam advantage cannot be dismissed as merely a missing DPA4/TACE-style scheduler, but the scheduler does expose a real recipe interaction that the current HybridMuon implementation is not handling.

The practical conclusion is that WSD should remain available for fair optimizer ablations, but it is not by itself a final HybridMuon fix. The next optimizer work should inspect the interaction among WSD phase boundaries, `start_swa`, loss weight switching, Muon LR factor, Magma-lite damping, and MACE's conservative radial-only routing. A true DPA4/TACE-quality recipe likely needs coordinated per-step WSD and explicit block metadata rather than only an epoch-level scheduler swap.



### 2026-07-02 CUEQ+Compile Speed-Gap Diagnosis

The current evidence explains why MACE `cueq+edge_force_compile` is not showing DPA4-like `~3x` end-to-end training acceleration. The compiled edge-force closure is working and stable, but the compiled region is only the guarded edge-vector energy/force/loss closure inside `EdgeForceCompiledLossModule.compiled_force_training_loss`. The outer `take_step` still performs batch transfer, Python closure dispatch, guard checks, optimizer step, LR scheduling, logging, validation, checkpointing, and model export outside the compiled graph. DPA4's reported compile speedup comes from a more compiler-native tensor-only model ABI plus compact edge representation, make_fx graph repair, Inductor lowering, bf16 AMP, and TF32; it is not equivalent to wrapping the current MACE training loop.

For RECIO/8k batch size `16`, CUEQ already removes a large fraction of the expensive tensor-product/symmetric-contraction work. Existing completed CUEQ 20k/200k pairs show the current compile path improves the raw optimizer-step timer only modestly:

| Run | Eager raw opt median | Compile raw opt median | Compile closure EMA | Slurm elapsed eager | Slurm elapsed compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20k Adam+CUEQ | `22.681 ms` | `21.602 ms` | `10.462 ms` | `00:11:25` (`599220`) | `00:11:21` (`599221`) |
| 20k HybridMuon+CUEQ | `25.357 ms` | `22.846 ms` | `10.312 ms` | `00:12:35` (`599222`) | `00:11:45` (`599223`) |
| 200k Adam+CUEQ | `22.697 ms` | `21.488 ms` | `10.391 ms` | `01:27:34` (`599551`) | `01:23:49` (`599552`) |
| 200k HybridMuon+CUEQ | `24.513 ms` | `22.163 ms` | `10.265 ms` | `01:35:17` (`599553`) | `01:26:07` (`599554`) |

This means the compiled closure itself is about half of the logged optimizer-step time, while the rest is still outside the compiled graph. The gap is therefore not simply one-time compilation overhead. Even in the 200k-step run, where setup is fully amortized, Adam+CUEQ compile is only about `4.3%` faster end-to-end and HybridMuon+CUEQ compile about `9.6%` faster. The next compile work should target higher coverage of the force-training hot path and lower wrapper overhead, not only cache setup.

The user's hypothesis that the current RECIO model/batch may be too small was tested directly. Two larger single-stage per-step WSD jobs completed on `4V100PX` using `num_channels=128`, `max_L=2`, `correlation=3`, batch size `8`, Adam+CUEQ, and `20,900` optimizer steps. Job `601653` was CUEQ eager and job `601654` was CUEQ+edge-force compile.

| Large-model case | Job | Slurm elapsed | MaxRSS | Raw opt median | Compile closure EMA | Valid E | Valid F | Test E | Test F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam+CUEQ eager, `C=128/L=2` | `601653` | `00:18:58` | `4016180K` | `34.369 ms` | n/a | `54.7 meV/atom` | `232.4 meV/A` | `63.2 meV/atom` | `217.4 meV/A` |
| Adam+CUEQ dynamic FX-only edge-force compile, `C=128/L=2` | `601654` | `00:19:47` | `4116104K` | `33.662 ms` | `16.289 ms` | `54.7 meV/atom` | `231.2 meV/A` | `62.3 meV/atom` | `214.7 meV/A` |

Increasing the model size therefore did not reveal a hidden DPA4-like speedup. Accuracy did not regress, but hot-step speed improved by only about `2.1%`, and total Slurm time became worse because strict setup gating took `234.023 s` in epoch 0. The compiled closure is still roughly half of the raw optimizer-step time; the remaining half is outside the compiled region.

A dedicated phase profiler then measured the same `C=128/L=2/correlation=3` model with RECIO indices `0:32`, CUEQ enabled, `torch 2.10.0+cu128`, `Tesla V100-SXM2-16GB`, and `--no-edge-compile-graph`, matching the currently accepted FX-only training path. SAI job `601751` completed successfully with no parity failures:

| Optimizer | Mode | Setup | Gate | Total median | Forward/loss median | Backward+clip median | Optimizer median |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Adam | position eager | `0.0 ms` | n/a | `14.886 ms` | `7.535 ms` | `7.076 ms` | `0.256 ms` |
| Adam | edge eager | `20191.1 ms` | pass | `14.664 ms` | `7.410 ms` | `6.994 ms` | `0.254 ms` |
| Adam | edge compile, FX-only | `27105.7 ms` | pass | `14.896 ms` | `7.425 ms` | `7.219 ms` | `0.256 ms` |
| HybridMuon | position eager | `0.0 ms` | n/a | `16.364 ms` | `7.566 ms` | `7.121 ms` | `1.640 ms` |
| HybridMuon | edge eager | `20177.1 ms` | pass | `16.189 ms` | `7.459 ms` | `7.002 ms` | `1.637 ms` |
| HybridMuon | edge compile, FX-only | `27101.6 ms` | pass | `16.256 ms` | `7.416 ms` | `7.200 ms` | `1.638 ms` |

The profiler is the clearest explanation so far. FX-only `edge_compile` is not an Inductor kernel-fusion path; it executes the repaired FX GraphModule and preserves conservative-force gradients, but it is not faster than the edge-eager closure. In this profile, forward/loss and outer backward/clip are each about `7 ms`, while Adam's optimizer update is tiny and HybridMuon's update is still only about `10%` of the total step. The next speed work should therefore stop treating FX-only as the final acceleration mechanism and instead focus on a safe Inductor-lowered force-training core, with CUEQ Linear and dynamic-fusion parity handled explicitly.

Three short Inductor graph profiler gates were then run from the same `mace_develop` environment on `4V100PX`, same RECIO `0:32` batch and `C=128/L=2/correlation=3` model. All used Adam only, `--edge-compile-graph`, dynamic Inductor, and strict gate comparison against position-eager conservative forces:

| Inductor gate | Job | CUEQ flags reported by profiler | Slurm | MaxRSS | Gate | Total median eager | Total median Inductor | Max force diff | Max param-grad diff |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `optimize_all` | `601872` | `optimize_all=True`, explicit `optimize_linear=False` | `00:01:27` | `5168228K` | pass | `15.059 ms` | `6.011 ms` | `3.58e-07` | `5.96e-07` |
| minus-linear | `601871` | `channelwise/fctp/symmetric=True`, `linear=False` | `00:01:28` | `5166252K` | pass | `13.911 ms` | `6.000 ms` | `2.16e-07` | `2.38e-07` |
| explicit-linear | `601922` | `optimize_all=True`, `optimize_linear=True` | `00:01:32` | `5208616K` | pass | `14.636 ms` | `5.779 ms` | `2.98e-07` | `6.26e-07` |

This changes the diagnosis in a useful way. On a fixed representative batch, Inductor graph lowering can deliver the expected DPA4-like per-step speedup and pass a same-weight force/loss/parameter-gradient gate, even with explicit CUEQ Linear enabled. Therefore the current blocker is no longer simply "CUEQ Linear cannot compile". The unresolved part is production training integration: symbolic multi-batch cache behavior, compile setup and gate cost, dynamic-shape correctness over real RECIO shape variation, long optimizer-trajectory accuracy, and whether the safe Inductor knobs used here remain stable when moved from the profiler into `take_step`.

The SAI RECIO wrapper now has a reusable `CUEQ_PROFILE={safe,minus_linear,full,linear,off}` switch. The default remains `safe`, preserving previous benchmark behavior, while `full` and `linear` allow the profiler-validated CUEQ Linear path to be tested in real `run_train` jobs without hand-writing long CLI commands.

A first production-loop gate then moved the profiler result into real multi-batch training. Job `601990` used the same RECIO/8k split, `mace_develop`, Adam, CUEQ full (`optimize_all=True`, `optimize_linear=True`, `optimize_channelwise=True`, `optimize_symmetric=True`, `optimize_fctp=True`, `conv_fusion=True`), TF32, `EDGE_FORCE_GRAPH=True`, dynamic cache, `C=64/L=1/correlation=3`, batch size `16`, three epochs, and parity logging every 200 steps. A matching eager control was job `602038`.

| 3-epoch gate | Job | Slurm | MaxRSS | Hot opt median | Final valid E | Final valid F | Test E | Test F | Compile/parity notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adam+CUEQ full eager | `602038` | `00:02:34` | `3524168K` | `23.046 ms` | `146.2 meV/atom` | `374.9 meV/A` | `155.5 meV/atom` | `378.6 meV/A` | no compile |
| Adam+CUEQ full Inductor graph | `601990` | `00:03:35` | `5235908K` | `12.924 ms` | `133.8 meV/atom` | `378.3 meV/A` | `144.2 meV/atom` | `382.9 meV/A` | `compiled=1425`, `cache_hits=1424`, `fallbacks=0`, `runtime_recompiles=0`, setup `44.495 s`, closure EMA `5.666 ms` |

This is the first real `run_train` evidence that full CUEQ plus Inductor graph lowering can produce a large cached-step speedup in MACE's conservative force training loop: hot median opt time improved from `23.05 ms` to `12.92 ms`, about `1.78x`. It is still not an acceptance result. The short total Slurm time is worse because one-time setup is `44.5 s`; peak memory rises by about `1.7 GB`; and strict periodic parity is not fully clean. Epoch summaries reported no fallback/recompile, but parity had small `1e-5` to `1e-4` differences: epoch 0 `max_grad=5.15e-05`, epoch 1 `8.58e-05`, and epoch 2 had `2/3` strict parity failures with `max_grad=4.53e-05` on `interactions.1.conv_tp_weights.layer0.weight` and `readouts.1.linear_2.weight`. This is acceptable evidence for continuing the Inductor integration, but not yet enough to claim no long-run precision or generalization impact.

A longer full-CUEQ production-loop check then ran the same Adam, CUEQ `optimize_all`, TF32, seed `456`, batch size `16`, `C=64/L=1/correlation=3` setup for the full 43-epoch RECIO/8k 20k-step schedule. Job `602670` enabled `EDGE_FORCE_GRAPH=True`, dynamic cache, strict setup gate, and non-strict parity logging every 1000 steps; job `602671` was the matching CUEQ eager control. Both completed with Slurm `ExitCode 0:0`.

| 20k full-CUEQ Adam gate | Job | Slurm | MaxRSS | Hot opt median | Hot opt mean | Final valid E | Final valid F | Test E | Test F | Compile/parity notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Adam+CUEQ full eager | `602671` | `00:10:22` | `3519060K` | `22.182 ms` | `22.214 ms` | `57.9 meV/atom` | `234.7 meV/A` | `65.1 meV/atom` | `210.2 meV/A` | no compile |
| Adam+CUEQ full Inductor graph | `602670` | `00:08:19` | `4809616K` | `12.732 ms` | `12.847 ms` | `61.5 meV/atom` | `237.1 meV/A` | `69.2 meV/atom` | `214.6 meV/A` | `compiled=20425`, `cache_hits=20424`, `fallbacks=0`, `runtime_recompiles=0`, setup `43.636 s`, closure EMA `5.614 ms`; parity `20` checks, `4` failed, max force diff `1.907e-05`, max grad diff `1.297e-04` |
| Adam+CUEQ full Inductor graph + compiled tensor loss | `603768` | `00:08:28` | `5303732K` | `12.723 ms` | `12.843 ms` | `60.0 meV/atom` | `235.6 meV/A` | `66.8 meV/atom` | `213.6 meV/A` | `compiled=20425`, `edge_force_compile_loss=20425`, `cache_hits=20424`, `fallbacks=0`, `runtime_recompiles=0`, setup `47.473 s`, closure EMA `5.652 ms`; parity `20` checks, `4` failed, max force diff `1.717e-05`, max grad diff `1.383e-04` |

This is the strongest current evidence that the production Inductor path is a real cached-step speedup: hot optimizer-step time improves by about `1.74x`, and total Slurm time improves by about `1.25x` even with strict setup and final evaluation included. It also shows why this path should remain gated rather than enabled by default. Peak memory rises by about `1.29 GB`, and the compiled trajectory is slightly worse on this run (`+4.4 meV/A` test force, `+4.1 meV/atom` test energy versus eager). The parity failures are small in absolute terms but systematic enough to keep the acceptance bar high: failed checks hit `grad:interactions.1.conv_tp_weights.layer0.weight`, `grad:node_embedding.linear.weight`, `grad:products.0.symmetric_contractions.weight`, `forces`, and `grad:readouts.1.linear_2.weight`.

The result narrows the next root-cause hypothesis. Cache misses, runtime recompiles, and CUEQ Linear are not the main blocker in this accepted 20k run: there is one initial compile, all later steps are cache hits, and no fallback occurs. The remaining speed gap to DPA4 is more likely from incomplete compiled coverage of the training step. The current compiled callable returns `energy, edge_grad`; atomic-force scatter, loss construction, outer `loss.backward()`, gradient clipping, optimizer update, LR scheduling, and logging still run outside the compiled executable. The next MACE-specific compile target should therefore be a tensor-only compiled loss closure that includes `edge_grad -> atomic forces -> energy/force loss`, followed by a separate decision on whether any outer backward/optimizer pieces can be compiled without breaking HybridMuon routing or numerical parity.

The practical implication is that `compile+CUEQ` remains a first-class optimization path, but the present implementation should be treated as a partial DPA4-style force-compile integration. To approach DPA4's compile benefit, MACE needs a deeper compiled training core: fewer Python-side wrappers per step, more explicit tensor-only inputs, safe Inductor graph lowering promoted from profiler to the guarded train loop, and native bf16 validation on suitable hardware. Until then, CUEQ alone already captures much of the V100 hot path, so incremental compile speedup is expected to be modest on small/medium RECIO settings.

A first code-side step toward that deeper core has now landed: for `WeightedEnergyForcesLoss`, the traced/compiled executable can return `(energy, forces, loss)` directly instead of returning `(energy, edge_grad)` and constructing atomic forces plus loss in Python. Custom losses and unsupported outputs still use the old edge-gradient path. Local verification in `mace_env` passed for the tensor-only loss equivalence, compiled-loss backward, legacy custom-loss snapshot, detach stripping, bucket padding, cache-hit gate, and periodic parity tests (`8 passed`). A short SAI GPU sanity gate, job `603652`, then ran Adam + full CUEQ + Inductor graph + TF32 for three RECIO/8k epochs in `mace_develop`; it completed with Slurm `COMPLETED`, `ExitCode 0:0`, elapsed `00:03:48`, and MaxRSS `5411116K`. The JSON log confirmed `compiled=1425`, `edge_force_compile_loss=True` on all compiled steps, `cache_hits=1424`, `fallbacks=0`, and no runtime recompiles.

This change is a correctness and architecture-boundary improvement, not yet a proven speed breakthrough. In job `603652`, hot optimizer-step median was `13.34 ms`, while the compiled executable EMA was `5.70 ms`; this is close to the earlier 3-epoch full-Inductor result (`12.92 ms` hot median, `5.67 ms` closure EMA). Periodic parity remained in the same small-difference regime: `7` checks, `2` failed under strict tolerance, max force diff `1.526e-05`, max grad diff `1.526e-04`. The 20k rerun, job `603768`, confirmed the same interpretation at full RECIO/8k gate scale. It did not improve hot-step speed relative to `602670` (`12.723 ms` versus `12.732 ms`) and used more memory (`5303732K` versus `4809616K`), but it slightly improved the compiled final test error (`213.6 meV/A` versus `214.6 meV/A`) while remaining close to but not equal to eager (`210.2 meV/A`). The next compile target is therefore still the outer higher-order backward/gradient-management region, not merely moving scalar loss arithmetic into the executable.

A targeted Inductor fusion-size ablation then tested the hypothesis from `review/manual-spherical-harmonics-vs-e3nn-compile.md` that overly aggressive fusion around the polynomial spherical-harmonics/reduction path may be driving the remaining trajectory drift. Two real RECIO/8k 20k-step jobs used the same full CUEQ, Adam, TF32, dynamic Inductor graph, seed `456`, batch size `16`, and 43-epoch setup as the `602670`/`603768` references, changing only `--edge_force_compile_max_fusion_size` from the default `8` to `4` or `2`. Run directory: `runs/recio8k_inductor_fusion_ablation_20k_20260702`.

| Inductor fusion cap | Job | Slurm state | Elapsed | MaxRSS | Hot median / mean | Compile setup | Closure EMA | Epoch-42 valid E/F | Test E/F | Compile health |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `4` | `604886` | `COMPLETED` | `00:08:31` | `5114344K` | `12.808 / 13.164 ms` | `47.357 s` | `5.718 ms` | `58.3 meV/atom`, `276.3 meV/A` | `66.5 meV/atom`, `253.1 meV/A` | `compiled=20425`, `cache_hits=20424`, `fallbacks=0`, `runtime_recompiles=0`; epoch-42 parity accepted with max force `9.537e-06`, max grad `1.235e-04` |
| `2` | `604891` | `COMPLETED` | `00:08:32` | `5134232K` | `12.836 / 13.194 ms` | `47.992 s` | `5.768 ms` | `79.9 meV/atom`, `281.2 meV/A` | `87.7 meV/atom`, `264.2 meV/A` | `compiled=20425`, `cache_hits=20424`, `fallbacks=0`, `runtime_recompiles=0`; epoch-42 parity accepted with max force `1.526e-05`, max grad `5.245e-05` |

This rejects the simple "reduce Inductor fusion size" fix. Lowering the fusion cap did not improve hot-step speed, memory, or final accuracy; it made the training trajectory much worse than the fusion-8 references (`602670`: `214.6 meV/A`, `603768`: `213.6 meV/A` test force). Same-weight periodic parity stayed in the familiar `1e-5` force / `1e-4` gradient range and did not predict the large trajectory difference. The next numeric-debug direction should therefore not be a global fusion cap. More targeted controls are needed: isolate polynomial spherical harmonics or force scatter as separate graph boundaries, compare polynomial versus e3nn SH in a short guarded run, and add fixed validation-batch trajectory probes rather than relying only on sparse parity checks.

A fixed-batch full-backward prototype now tests that next target directly. The step profiler has a new `edge_compile_grads` mode: instead of returning a loss and calling outer `loss.backward()`, its traced closure computes `energy -> forces -> weighted loss -> torch.autograd.grad(loss, model parameters)` and returns the parameter-gradient tensors; the step then writes those tensors to `p.grad` before clipping and `optimizer.step()`. A CPU FX-only smoke on a tiny `C=8/L=1` model passed the gate with max parameter-gradient difference `1.49e-08`.

SAI job `604099` then ran the real V100/CUEQ/Inductor fixed-batch gate in `mace_develop` on `4V100PX`: RECIO `0:32`, `C=128/L=2/correlation=3`, Adam, full CUEQ `optimize_all`, dynamic Inductor, warmup `3`, repeats `10`. All gates passed. The new full-backward executable had max energy diff `3.58e-07`, max force diff `4.17e-07`, max loss diff `1.43e-06`, and max parameter-gradient diff `5.96e-07` versus position-eager conservative training.

| Fixed-batch mode | Setup | Total median | Forward/loss/grads median | Backward+clip median | Optimizer median | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| position eager | `0.0 ms` | `14.884 ms` | `7.487 ms` | `7.130 ms` | `0.257 ms` | eager reference; mean is inflated by one warmup-adjacent dynamic outlier |
| edge compile | `43833.0 ms` | `6.017 ms` | `2.657 ms` | `3.098 ms` | `0.256 ms` | current compiled force/loss path; outer backward still about half of the compiled step |
| edge compile grads | `64201.5 ms` | `7.026 ms` | `6.497 ms` | `0.264 ms` | `0.255 ms` | complete parameter-gradient closure works, but is slower than leaving outer backward outside on this fixed batch |

This is an important negative/diagnostic result. It proves the full conservative force-loss parameter-gradient graph can be captured with CUEQ and Inductor on V100, but simply moving `loss.backward()` into the compiled closure is not automatically a speed win. The current production bottleneck is therefore not just Python's outer backward call; it is the generated graph/kernel quality and dynamic training-loop integration. The next compile work should inspect `edge_compile` versus `edge_compile_grads` generated graphs/kernels, reduce redundant saved-tensor/recompute work, and only then consider wiring full-gradient execution into `take_step` behind the same strict parity gates.

A follow-up graph-statistics profiler, job `604221`, reran only `edge_compile` and `edge_compile_grads` with the same V100/CUEQ/Inductor `C=128/L=2/correlation=3` fixed batch, warmup `1`, repeats `3`. The result explains the slowdown structurally rather than only by timing. `edge_compile` produced `1484` FX nodes and `3` output tensors; `edge_compile_grads` produced `2880` FX nodes and `25` output tensors. Median step time was `6.329 ms` versus `7.317 ms`; setup was `43.769 s` versus `63.904 s`.

The largest target-count deltas in `edge_compile_grads - edge_compile` were mostly higher-order gradient algebra and layout traffic: `view` `+237`, `permute` `+207`, `mul.Tensor` `+166`, `add.Tensor` `+131`, `transpose` `+98`, `bmm` `+79`, `slice_backward` `+56`, `squeeze` `+49`, `t` `+49`, `slice` `+48`, `mm` `+33`, `clone` `+28`, and `_unsafe_view` `+28`. The CUEQ tensor product call itself doubled from `8` to `16`, but the bigger cost signal is that tracing parameter gradients materializes many dense matrix-gradient/layout transforms inside one large Inductor graph. This points away from a one-piece full-gradient executable as the immediate production path. A better next experiment is to keep the fast compiled force/loss executable and target only selected high-cost outer-backward blocks, or split the full-gradient graph into smaller compiled regions where Inductor does not have to optimize all parameter-gradient outputs and layout transforms at once.

The profiler now has `--edge-compile-grad-filter`, a diagnostic-only substring filter for `edge_compile_grads`. It compiles and verifies gradients for only the selected trainable parameters, allowing the split-compile hypothesis to be measured without changing production training. CPU smoke verified the subset parity path on a tiny model; a readout-only graph returned one gradient tensor plus energy/forces/loss and passed with max grad diff `0`.

Two V100/CUEQ subset jobs then tested whether smaller parameter-gradient regions reduce the graph enough to be useful:

| Filter | Job | Grad tensors | FX nodes | Output tensors | Setup | Median step | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| all trainable params | `604221` | n/a in pre-metadata run | `2880` | `25` | `63.904 s` | `7.317 ms` | pass |
| `readouts` | `604281` | `3` | `2060` | `6` | `57.869 s` | `5.250 ms` | pass |
| `conv_tp_weights` | `604282` | `8` | `2542` | `11` | `62.257 s` | `6.610 ms` | pass |

This is the first evidence that a split-gradient strategy may be worth pursuing. Readout-only compiled gradients are much smaller and faster than the all-parameter gradient graph, while conv-TP-weight gradients still carry most of the heavy tensor-product/layout cost. This does not yet define a production algorithm, because a training step still needs all gradients without double-computing the conservative force path. But it gives a concrete next implementation target: experiment with blockwise gradient compilation around readout/radial/TP groups and measure whether any split can reuse the compiled force/loss core or avoid redundant forward/force recomputation. If every subset requires replaying the full conservative force graph, split compile will likely lose despite smaller individual graphs.

A sequence-mode replay diagnostic now tests the main risk in that split-gradient idea. `edge_compile_grads_sequence` compiles multiple filtered gradient closures and executes them sequentially in one optimizer step. This is intentionally diagnostic rather than a production algorithm: if each filtered closure has to recompute `energy -> forces -> loss`, the smaller subgraphs will not compose into a faster complete training step. Local CPU FX-only smoke passed on a tiny `C=8/L=1` model: the `readouts` and `conv_tp_weights` subgraphs both matched position-eager energy/force/loss and selected gradients, with max gradient differences at `1e-8` scale.

SAI job `604479` then ran the real V100/CUEQ/Inductor fixed-batch diagnostic in `mace_develop` on `4V100PX`: RECIO `0:32`, `C=128/L=2/correlation=3`, Adam, full CUEQ `optimize_all`, dynamic Inductor, warmup `1`, repeats `3`, and filters `readouts,conv_tp_weights` versus the sequence `readouts:conv_tp_weights`. All gates passed. The same-weight parity differences remained small: `edge_compile` max force diff `4.17e-07` and max parameter-gradient diff `6.85e-07`; combined `edge_compile_grads` max force diff `3.58e-07` and max selected-gradient diff `6.56e-07`; sequence substeps had max selected-gradient diffs `4.77e-07` for `readouts` and `3.58e-07` for `conv_tp_weights`.

| Fixed-batch diagnostic | Job | Grad tensors | FX nodes | Output tensors | Setup | Mean step | Median step | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `edge_compile` force/loss | `604479` | n/a | `1484` | `3` | `43.478 s` | `6.485 ms` | `6.247 ms` | pass |
| combined `edge_compile_grads`, filter `readouts,conv_tp_weights` | `604479` | `11` | `2589` | `14` | `62.051 s` | `8.549 ms` | `6.749 ms` | pass |
| sequence `readouts:conv_tp_weights` | `604479` | `11` | `4632` | `17` | `76.880 s` | `12.251 ms` | `11.075 ms` | pass |

This closes one tempting but flawed path. Splitting parameter-gradient compile by semantic block can reduce each individual graph, but naively executing multiple filtered closures replays the conservative force path and loses the benefit. The two sequence subgraphs had `2071` and `2561` nodes, summing almost exactly to the reported `4632` aggregate, and the step time moved toward the replay cost rather than toward the fast force/loss closure. Therefore the production strategy should not be "compile many independent gradient subsets and run all of them." The next viable direction is either a shared compiled force/loss core with selective reusable backward pieces, or a better one-piece parameter-gradient graph that reduces higher-order layout traffic without exploding output tensors.

A separate diagnostic tested PyTorch 2.10's private compiled-autograd path as a narrower alternative: keep the fast `edge_compile` force/loss closure, but execute the outer `loss.backward()` under `torch._dynamo.compiled_autograd._enable`. This was added only as `edge_compile_compiled_autograd` in the step profiler, not in production training, because the API is private and `torch._dynamo.config` in the current `mace_develop` environment exposes no public compiled-autograd flag. Local CPU smoke verified the mode's plumbing with exact parity.

SAI job `604603` then compared `edge_compile` and `edge_compile_compiled_autograd` on the same V100/CUEQ/Inductor fixed batch: RECIO `0:32`, `C=128/L=2/correlation=3`, Adam, full CUEQ `optimize_all`, warmup `2`, repeats `5`. Both gates passed. The compiled-autograd variant had max force diff `3.87e-07` and max parameter-gradient diff `5.96e-07`, comparable to plain `edge_compile` (`4.17e-07` force, `6.26e-07` gradient). It did not improve speed:

| Fixed-batch mode | Job | Setup | Forward/loss mean | Backward+clip mean | Optimizer mean | Total mean | Total median | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `edge_compile` | `604603` | `44.096 s` | `2.648 ms` | `3.100 ms` | `0.259 ms` | `6.010 ms` | `5.942 ms` | pass |
| `edge_compile_compiled_autograd` | `604603` | `32.335 s` | `2.625 ms` | `3.144 ms` | `0.257 ms` | `6.029 ms` | `5.966 ms` | pass |

This rules out a simple "wrap outer backward with compiled autograd" fix for the current V100/CUEQ path. The remaining performance gap is not Python's backward dispatcher alone; it is the structure and kernel quality of the higher-order backward graph and the amount of layout/tensor-product gradient work that must be generated. Compiled autograd may still matter in a future PyTorch version or after a deeper tensor-only ABI refactor, but in the present environment it should remain a profiler-only diagnostic.

### 2026-07-02 Per-Step WSD and Single-Stage Muon Gate

The WSD implementation has been upgraded from an epoch-level compatibility shim to a DPA4/TACE-style per-optimizer-step scheduler. The new CLI flag is `--lr_scheduler_interval={auto,epoch,step}`. `auto` now uses per-step updates for `--scheduler=WSD` and preserves per-epoch updates for legacy `ReduceLROnPlateau`/`ExponentialLR`. `WSD + step` computes its total schedule length from `max_num_epochs * len(train_loader)`, logs the current LR on optimizer-step records, and advances only after a non-skipped optimizer step. If loss guarding skips a batch, the WSD schedule does not advance ahead of parameter updates.

This also changes the Stage Two interaction. In MACE's original loop, entering Stage Two stopped the normal LR scheduler and let `SWALR` take over. That made the previous WSD benchmark an epoch-level WSD only before Stage Two, followed by MACE's SWA LR recipe. With per-step WSD active, the WSD schedule continues through Stage Two; Stage Two can still change loss weights and update the averaged model, but it no longer overrides WSD with `SWALR`. This is closer to DPA4/TACE, where WSD is a step-level training recipe rather than a validation-epoch callback.

The RECIO scaling generator now supports both controls needed for a clean Muon recipe ablation: `--lr-scheduler-interval step` and `--single-stage`. The latter omits `--swa`, `--start_swa`, and Stage Two loss/LR flags, so Muon can be tested under a pure WSD single-stage schedule instead of inheriting MACE's Adam-oriented Stage Two recipe.

Focused local verification passed after this change: `tests/test_lr_scheduler.py`, `tests/test_recio8k_scaling_scripts.py`, and the relevant `train_one_epoch` compile/scheduler tests reported `18 passed`. The tests cover WSD per-step defaults, checkpointable WSD state, parser flags, single-stage benchmark generation, and the training-loop invariant that per-step LR advances only after successful optimizer updates.

Four real RECIO/8k CUEQ+compile 20k-step jobs then completed on `4V100PX`. Each run used seed `456`, batch size `16`, `ScaleShiftMACE`, `num_channels=64`, `max_L=1`, `correlation=3`, dynamic FX-only edge-force compile, CUEQ `0.6.1`, and per-step WSD. All four jobs completed with Slurm `ExitCode 0:0`; the compile guard reported `fallbacks=0` and `runtime_recompiles=0`.

| Case | Job | Slurm elapsed | MaxRSS | Hot opt mean | Final test E | Final test F | Final valid E | Final valid F |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam, per-step WSD, Stage Two | `601392` | `00:11:26` | `3679M` | `10.287 s/epoch` | `27.7 meV/atom` | `214.4 meV/A` | `26.5 meV/atom` | `237.6 meV/A` |
| HybridMuon, per-step WSD, Stage Two | `601390` | `00:11:47` | `3769572K` | `10.826 s/epoch` | `32.2 meV/atom` | `232.5 meV/A` | `28.1 meV/atom` | `254.1 meV/A` |
| Adam, per-step WSD, single-stage | `601389` | `00:10:59` | `3278344K` | `10.481 s/epoch` | `65.0 meV/atom` | `211.4 meV/A` | `57.3 meV/atom` | `234.2 meV/A` |
| HybridMuon, per-step WSD, single-stage | `601391` | `00:11:11` | `3662524K` | `10.859 s/epoch` | `65.4 meV/atom` | `222.9 meV/A` | `59.8 meV/atom` | `247.1 meV/A` |

The single-stage Muon result supports the user's hypothesis that MACE Stage Two is not automatically compatible with the current radial-only HybridMuon recipe: removing Stage Two improves Muon's final test force from `232.5` to `222.9 meV/A`. However, it does not make Muon beat Adam on this 20k-step RECIO gate. The best force result is still single-stage Adam (`211.4 meV/A`), while Stage Two Adam gives much better energy (`27.7 meV/atom`) with slightly worse force (`214.4 meV/A`).

This separates the causes more cleanly. Per-step WSD and single-stage training are now implemented and verified, but they are not sufficient by themselves to close the Adam/Muon gap. The next optimizer work should focus on the HybridMuon update and routing recipe: DPA4's Muon implementation details, TACE's broader but structure-aware MACE-like parameter coverage, WSD phase ratios, Muon LR factor, Magma-lite damping, and whether explicit block metadata can safely expose more legal matrix blocks without flattening equivariant or path-semantic axes.

### TACE Adaptive Loss Placement Note

A source check of `reference_repos/tace/tace/utils/loss/uncertainty.py` confirms that TACE's adaptive loss is Kendall-style uncertainty weighting. It introduces trainable `log_sigmas` for each task and optimizes `0.5 * exp(-log_sigma) * task_loss + log_sigma`. This can be useful for MACE because energy/force/stress fitting is a multi-task problem, but it is not an optimizer-only tweak: it changes the training objective and directly competes with MACE's hand-tuned Stage Two loss-weight schedule.

Therefore the right migration path is not to enable adaptive loss on top of the current Stage Two recipe. It should be tested as a separate loss-schedule alternative after the per-step WSD/stage ablation above finishes. The clean comparison is likely `weighted + Stage Two` versus `uncertainty/adaptive + single-stage WSD`, with Adam and conservative radial-only HybridMuon both included. This keeps the effect attributable and avoids mixing two independent loss-weight schedules.

### 2026-07-02 Fixed-Batch Compile Trajectory Probe

The current-batch parity checks were not enough to explain why some compile variants keep same-weight differences around `1e-5` force / `1e-4` parameter-gradient scale but still diverge to much worse RECIO/8k validation trajectories. A new fixed-batch probe now instruments that missing axis without changing the training objective. When `--edge_force_compile_fixed_probe_interval N` is enabled, the training wrapper freezes the first successfully compiled batch on CPU, then every `N` compiled training steps replays that same batch through both the eager position-gradient path and the cached compiled edge-force path at the current model weights. Metrics are logged under `edge_force_fixed_probe_*`. By default it compares energy, forces, and scalar loss only; `--edge_force_compile_fixed_probe_gradients` adds parameter-gradient comparisons, and `--edge_force_compile_fixed_probe_strict` can turn the diagnostic into a hard gate.

This is deliberately separate from the existing `edge_force_parity_*` current-batch check. Current-batch parity answers whether the compiled path matches eager on the batch being optimized at that step. The fixed probe answers whether a stable representative geometry accumulates larger eager-vs-compiled discrepancies as weights move during training. That is the more direct diagnostic for the observed trajectory drift, especially after the negative fusion-size ablation showed that sparse current-batch parity did not predict final force MAE.

Implementation details are intentionally conservative: the frozen batch is a detached CPU clone of the original batch dictionary, so it does not keep extra GPU memory alive; the probe runs only after a compiled cache entry exists; it reuses the compiled executable associated with the frozen batch's cache key; and it clears diagnostic gradients before the real training loss is evaluated. The feature is default-off and should not affect existing benchmarks unless explicitly enabled.

Local verification passed in `mace_env`. The new RED test first failed because `EdgeForceCompileConfig` had no `fixed_probe_interval`; after implementation, `tests/test_compile.py::test_edge_force_compile_fixed_probe_reuses_first_batch` and `tests/test_compile.py::test_arg_parser_accepts_edge_force_compile_flags` passed. The full compile regression file then passed with `80 passed, 21 skipped` in `370.75 s`. A first SAI sanity job, `605017`, confirmed the CLI flags reached `run_train` and full RECIO/CUEQ/Inductor training completed with `fallbacks=0`, but it exposed that step-level fixed-probe metrics were not included in the epoch summary. The summary aggregation was then fixed under a RED test in `test_train_one_epoch_logs_edge_force_parity_summary`; the targeted fixed-probe/parser/summary tests passed with `3 passed`.

The real RECIO/CUEQ summary gate now passes. SAI job `605076` ran one RECIO/8k epoch in `mace_develop` with Adam, full CUEQ `optimize_all`, TF32, dynamic Inductor edge-force graph, and `EDGE_FORCE_FIXED_PROBE_INTERVAL=50`. It completed with Slurm `COMPLETED`, elapsed `00:03:55`, MaxRSS `5378628K`, `compiled=475`, `cache_hits=474`, `fallbacks=0`, and `runtime_recompiles=0`. The epoch summary reported `fixed_probe=checks=9 accepted=9 failed=0 shape=190x4710 max_energy=7.629e-06 max_forces=3.052e-05 max_loss=1.068e-04 max_grad=0.000e+00`. This establishes the diagnostic path for the next longer compile-vs-eager trajectory run; it is not itself a training-speed or accuracy benchmark.

### 2026-07-02 Fixed-Probe 20k Trajectory Gate

A matched real RECIO/8k 20k-step trajectory gate now compares eager full-CUEQ Adam against full-Inductor edge-force compile with the fixed-batch probe enabled. All runs used `mace_develop`, seed `456`, batch size `16`, `max_num_epochs=43` (`20425` optimizer steps), `ScaleShiftMACE` with `num_channels=64`, `max_L=1`, `correlation=3`, full CUEQ `optimize_all`, TF32, and Adam.

| Case | Job | Slurm elapsed | MaxRSS | Hot opt median | Hot opt mean | Test E | Test F | Final valid E | Final valid F | Compile/probe diagnostics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eager full-CUEQ Adam | `605112` | `00:13:17` | `3534492K` | `26.56 ms` | `28.50 ms` | `67.6 meV/atom` | `244.2 meV/A` | `62.5 meV/atom` | `323.9 meV/A` | n/a |
| full Inductor compile, fixed value probe | `605116` | `00:09:05` | `5151696K` | `13.16 ms` | `13.60 ms` | `114.5 meV/atom` | `240.4 meV/A` | `113.2 meV/atom` | `265.3 meV/A` | `compiled=20425`, `fallbacks=0`, `runtime_recompiles=0`; current-batch parity `20/20` accepted, max force `2.29e-5`, max grad `1.74e-4`; fixed value probe `20/20` accepted, max force `3.05e-5`, max loss `2.59e-4` |
| full Inductor compile, fixed gradient probe | `605152` | `00:08:36` | `5269448K` | `12.60 ms` | `12.99 ms` | `74.3 meV/atom` | `244.7 meV/A` | `67.1 meV/atom` | `266.3 meV/A` | `compiled=20425`, `fallbacks=0`; current-batch parity `20/20` accepted, max force `2.29e-5`, max grad `1.54e-4`; fixed gradient probe `20/20` accepted, max force `3.81e-5`, max grad `2.61e-4` |

The speed result is now consistent: dynamic full-Inductor edge-force compile plus CUEQ gives about `2.0x` hot-step speedup over eager full-CUEQ Adam (`26.56 ms -> 13.16 ms` or `12.60 ms`) and about `1.46x` end-to-end Slurm speedup in the value-probe run (`13:17 -> 9:05`), at the cost of roughly `46-49%` higher MaxRSS. The compile cache is stable: one compile, all subsequent steps cache-hit, no fallback, and no runtime recompile.

The precision/generalization conclusion is more subtle. The value-only fixed-probe run improves test force slightly versus this eager rerun (`240.4` vs `244.2 meV/A`) but degrades energy badly (`114.5` vs `67.6 meV/atom`). Current-batch parity and fixed-batch value parity both remain accepted throughout, so value parity alone does not explain the energy trajectory difference. The gradient-probe run shows fixed-batch parameter-gradient differences up to `2.61e-4`, but its final energy (`74.3 meV/atom`) is much closer to eager while force (`244.7 meV/A`) is essentially eager-like. Since enabling an every-1000-step diagnostic backward should not intentionally change the optimizer update, this is evidence that the current compile/CUEQ/V100 training trajectory is sensitive to small numerical/order effects or nondeterministic GPU reductions. It also means one 20k seed is not enough to claim precision is preserved, even when all same-weight parity gates pass.

The next debug target should therefore shift from cache correctness to trajectory reproducibility: run a small seed sweep or deterministic-mode probe, and add a cross-run fixed validation-batch trajectory comparison between eager and compile checkpoints/predictions. The fixed-probe tool is useful for catching same-weight eager-vs-compiled mismatch, but it does not by itself certify that optimizer trajectories remain statistically equivalent over 20k steps.

A repeat compile value-probe run clarifies the interpretation above. Job `605212` reran the same configuration as `605116` (`EDGE_FORCE_FIXED_PROBE_GRADIENTS=False`, seed `456`) and completed with Slurm `COMPLETED`, elapsed `00:08:31`, MaxRSS `5123672K`, `compiled=20425`, `fallbacks=0`, and `runtime_recompiles=0`. Its hot opt median was `12.56 ms`, final validation E/F was `69.9 meV/atom` / `271.4 meV/A`, and final test E/F was `78.5 meV/atom` / `248.2 meV/A`. Current-batch parity again accepted all `20` checks with max force `1.91e-5` and max grad `1.70e-4`; fixed value probe accepted all `20` checks with max force `2.67e-5`.

This repeat makes `605116` look like an energy-trajectory outlier rather than a deterministic compile-vs-eager bias. Across the two value-probe compile runs and the gradient-probe compile run, hot-step speed is stable (`12.56-13.16 ms` median), cache behavior is stable, and force test error stays near eager (`240.4-248.2 meV/A` vs eager `244.2 meV/A`). Energy varies much more (`74.3-114.5 meV/atom` across compile runs, eager `67.6 meV/atom`). The same-weight parity gates do not fail in any run, so the remaining risk is optimizer-trajectory variance from tiny numerical differences rather than an obvious broken compiled force objective. A useful next acceptance gate is therefore a small multi-seed/repeat statistical comparison, not a single-seed pass/fail.

### 2026-07-02 Three-Seed Compile Accuracy Statistics

A small multi-seed gate now gives a better estimate than the single-seed trajectory above. Two additional seeds, `123` and `789`, were run for both eager full-CUEQ Adam and full-Inductor compile with fixed value probe. Together with seed `456` (`eager_reference_20k` and the repeat compile run `fixedprobe_20k_rep2`), this gives a 3-seed comparison at the same RECIO/8k 20k-step setting.

| Seed | Mode | Job/source | Hot opt median | Test E | Test F | Valid E | Valid F | Compile diagnostics |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `456` | eager | `605112` | `26.56 ms` | `67.6` | `244.2` | `62.5` | `323.9` | n/a |
| `456` | compile | `605212` | `12.56 ms` | `78.5` | `248.2` | `69.9` | `271.4` | `compiled=20425`, `fallbacks=0`, parity `20/20`, fixed probe `20/20` |
| `123` | eager | `610065` | `22.70 ms` | `84.8` | `256.6` | `65.8` | `275.6` | n/a |
| `123` | compile | `610066` | `12.70 ms` | `71.4` | `265.7` | `71.9` | `283.7` | `compiled=20425`, `fallbacks=0`, parity `20/20`, fixed probe `20/20` |
| `789` | eager | `610067` | `22.32 ms` | `70.1` | `251.3` | `83.0` | `275.3` | n/a |
| `789` | compile | `610068` | `13.44 ms` | `81.8` | `267.7` | `96.9` | `290.1` | `compiled=20425`, `fallbacks=0`, parity `20/20`, fixed probe `20/20` |

Across these three seeds, eager hot-step median is `23.86 +/- 2.34 ms`; compile hot-step median is `12.90 +/- 0.48 ms`, a `1.85x` median-step speedup. Compile memory remains higher: the new seed-123/789 compile jobs used about `5.05 GB` MaxRSS versus about `3.02-3.06 GB` for eager, consistent with the earlier seed-456 runs. Cache behavior remains stable: one dynamic compile, no fallback, no runtime recompile.

Accuracy is not a simple pass yet. Test energy means are close (`74.2 +/- 9.3 meV/atom` eager versus `77.2 +/- 5.3` compile). Test force is modestly worse for compile in this small sample (`250.7 +/- 6.2 meV/A` eager versus `260.5 +/- 10.7` compile), while validation force is better for compile on average (`291.6 +/- 28.0` eager versus `281.7 +/- 9.5` compile). This is enough to say the compiled conservative force objective is not obviously broken, but not enough to claim no precision/generalization loss.

The next acceptance criterion should be stricter than same-weight parity: either a larger seed/repeat table or a deterministic/checkpoint prediction audit. In particular, the compile path has now met the speed and cache-stability gate, but precision preservation remains conditional. Before enabling it as a default training acceleration, the remaining work is to reduce memory overhead and establish statistically non-inferior E/F accuracy under the target MACE training recipe, including HybridMuon compatibility.


### 2026-07-03 Position-Gradient Compile Diagnostic

A new opt-in compile mode, `--edge_force_compile_force_gradient_mode=positions`, now traces the fuller conservative path from `positions -> edge vectors -> energy -> forces -> loss`. The default remains `edge`, which uses the faster edge-vector leaf path and scatters edge gradients back to atomic forces. The position-gradient mode is intended for DPA4-style diagnostic comparison, not as the production default.

The first real RECIO/8k CUEQ smoke exposed a concrete bug in the fuller path. Job `615062` used `EDGE_FORCE_GRAPH=True`, full CUEQ, HybridMuon, TF32, dynamic cache, and `force_gradient_mode=positions`; it failed after `00:02:19` because `shifts` was captured as a constant in the traced graph. The next batch had a different edge count, so Inductor saw `positions[receiver] - positions[sender]` with symbolic edge size but the old constant `shifts` shape, producing a fake-tensor shape mismatch. This was fixed by promoting `shifts` to a compiled input in positions mode.

The retry `615081` proved that fix removed the shape mismatch, but it then showed another cache-design problem: `shifts` was included in the dynamic cache key with its concrete first dimension, causing a new trace for nearly every batch. A phase-logging diagnostic (`615160`) made the issue explicit: successive cache keys contained `shifts` shapes such as `(6242, 3)`, `(5398, 3)`, `(5288, 3)`, and each one retraced for about `1.6-2.4 s`. This was fixed by making the first dimension of `shifts` and `unit_shifts` dynamic in `edge_force_compile_dynamic_cache_key`.

After the cache-key fix, the FX-only position-gradient path completed a full real RECIO/8k epoch. Job `615171` used `EDGE_FORCE_GRAPH=False`, full CUEQ, HybridMuon, TF32, batch size `16`, and dynamic cache. It completed with Slurm `COMPLETED`, elapsed `00:02:49`, MaxRSS `3649132K`, `compiled=475`, `cache_hits=474`, `new_compiles=1`, `fallbacks=0`, and `runtime_recompiles=0`. Current-batch parity accepted all `9/9` checks with max force diff `7.629e-06` and max grad diff `3.052e-05`; fixed-probe parity accepted all `9/9` checks with max force diff `1.144e-05` and max grad diff `5.627e-05`. The expensive setup phases were `trace=2.138 s`, `gate_reference=49.783 s`, and `gate_candidate=15.902 s`; hot optimizer work was `102.423 s` for the epoch.

The full Inductor position-gradient path also completed after the dynamic-key fix, but it is slower, not faster. Job `615183` used `EDGE_FORCE_GRAPH=True` with the same RECIO/8k/CUEQ/HybridMuon setup and completed with Slurm `COMPLETED`, elapsed `00:04:04`, MaxRSS `4937408K`, `compiled=475`, `cache_hits=474`, `new_compiles=1`, and no fallbacks. Parity again passed: current-batch checks `9/9`, max force diff `2.814e-05`, max grad diff `1.984e-04`; fixed-probe checks `9/9`, max force diff `2.289e-05`, max grad diff `9.346e-05`. Setup was dominated by `gate_candidate=45.544 s` plus `gate_reference=11.487 s`; hot optimizer work was `162.480 s` for the epoch.

This closes an important design question. A more literal positions-to-force compiled conservative path now works with real RECIO/8k, CUEQ, TF32, and HybridMuon, and its parity is acceptable at the current tolerances. However, it is not a training acceleration path for this MACE implementation: FX-only positions mode is already much slower than the edge-gradient compile path, and full Inductor positions mode is slower still. The practical DPA4-inspired direction for MACE should therefore keep the edge-gradient compiled force/loss core as the production path, while using positions mode as a correctness diagnostic for boundary conditions, cache policy, and future tensor-only ABI experiments.


### 2026-07-03 AMP Boundary for Compiled Force Training

The training precision boundary has been tightened to match the DPA4 safety principle more closely. Previously, `take_step` wrapped the entire training loss construction in `get_training_precision_context`, so enabling `--train_amp_dtype=bf16` or `fp16` would also place `compiled_force_training_loss` under autocast. For the edge-force compile path, that is too broad: it can lower precision for geometry-sensitive operations such as edge-vector construction, spherical harmonics, coordinate/edge derivatives, and force-loss graph tracing.

`take_step` now separates the two precision controls. `torch.set_float32_matmul_precision("high")` still applies around the whole closure when `--train_tf32` is enabled, but autocast is applied only to the ordinary eager model forward branch. The compiled force-loss hook executes outside autocast, preserving FP32 geometry/force construction unless a future, explicitly segmented AMP implementation is added inside the compiled MACE energy path. This keeps current edge-force compile correctness gates meaningful and avoids silently mixing low precision into conservative force derivatives.

Local regression coverage now includes a `take_step` test that monkeypatches autocast and verifies the compiled force-loss hook runs with autocast disabled while the normal eager forward still runs with autocast enabled. `tests/test_training_precision.py` and the compiled `take_step` hook tests passed (`15 passed`). This is not a bf16 speedup claim; on the visible V100 partitions bf16 remains fail-closed. The next bf16 benchmark should be run only on native-bf16 hardware, or with an explicitly opt-in diagnostic that compares RECIO/8k parity and accuracy before using AMP for production.

### 2026-07-03 SAI Wrapper AMP Wiring

The hand-written RECIO/8k edge-cache Slurm wrapper now forwards `TRAIN_AMP_DTYPE` to `mace.cli.run_train` as `--train_amp_dtype`, with the default kept at `none`. This aligns the wrapper with the generated 20k scaling cases, which already emitted `--train_amp_dtype=none` or bf16-specific variants. The purpose is benchmark integrity: future real RECIO/8k runs can explicitly compare FP32, TF32-only, and opt-in AMP settings from the same SAI wrapper instead of relying on Python defaults.

This does not change the current production precision boundary. On V100, bf16 remains fail-closed through `TrainingPrecisionConfig`, and the compiled force-loss hook still runs outside autocast so geometry and conservative force derivatives stay FP32. Therefore earlier `run_edge_force_cache_policy_sai.sh` results should be interpreted as FP32/TF32 runs unless their generated case or command line explicitly contained `--train_amp_dtype`. A true DPA4-style bf16 speed/accuracy comparison still requires a native-bf16 GPU or a separate diagnostic run that proves parity and RECIO accuracy.

A real SAI smoke verified the wrapper wiring. Job `619830` ran `TRAIN_AMP_DTYPE=none`, `TRAIN_TF32=True`, `EDGE_FORCE_COMPILE=True`, `EDGE_FORCE_GRAPH=False`, no CUEQ, `mace_develop`, batch size `16`, and one RECIO/8k epoch. It completed with Slurm `COMPLETED`, elapsed `00:01:16`, MaxRSS `3274852K`, and the training log reported `Using training float32 matmul precision: high` plus `Edge-force compile epoch 0 summary: steps=475, compiled=475, cache_hits=474, new_compiles=1, fallbacks=0, runtime_recompiles=0`. This is a wrapper/CLI smoke only, not a speed or accuracy benchmark.

### 2026-07-03 Review Refresh: Current Bottleneck Choice

A refresh against `review/dpa4-compile-cache-vs-mace.md` shows that one early diagnosis is now outdated. The old concern was batch-identity compile caching; the current `EdgeForceCompiledTrainingWrapper` already uses a dynamic cache key for the production edge mode where `positions`, `edge_index`, `node_attrs`, `batch`, masks, and related tensors have dynamic atom/edge axes, and real RECIO/8k smoke job `619830` compiled once and hit the cache for the remaining `474/475` batches. Therefore the next high-value compile task is not another cache-key rewrite. The remaining speed gap to DPA4 is more likely from the compile boundary and operator mix: CUEQ custom kernels hide most tensor-product work from Inductor, geometry/scatter/loss/optimizer/dataloader work stays outside the compiled closure, and the fuller `positions` force mode is correct but slower.

The optimizer review also needs an updated interpretation. Current `mace.tools.hybrid_muon` already includes DPA4/TACE-inspired pieces that the earlier comparison treated as missing: DeepSeek-style two-stage Newton-Schulz coefficients, optional Magma-lite damping, `muon_mode={2d,slice}`, and `hybrid_muon_routing={mace,tace}`. The open question is no longer whether these knobs can be represented in code, but whether TACE-style `routing=tace, muon_mode=slice` improves real MACE training without destabilizing equivariant product/symmetric-contraction parameters. Because optimizer time is a small fraction of the training step, this should be benchmarked for accuracy/stability after the compile/CUEQ/precision path is stable, not treated as the primary throughput fix.

The next benchmark design should therefore separate two axes. For throughput, continue with Adam or conservative radial-only HybridMuon and profile the real training-step phase breakdown under CUEQ eager versus CUEQ+edge compile, especially outside the compiled closure. For optimizer quality, run a smaller controlled RECIO/8k comparison of `hybrid_muon_routing=mace, hybrid_muon_mode=2d` against `hybrid_muon_routing=tace, hybrid_muon_mode=slice`, ideally with single-stage per-step WSD and the same compile setting. Mixing both investigations in one first run would make a speed or accuracy change hard to attribute.

### 2026-07-03 Fixed-Batch Phase Breakdown: CUEQ Eager vs Edge Compile

SAI job `619891` ran a focused phase profiler after the branch push, using `mace_develop`, one `Tesla V100-SXM2-32GB`, RECIO `8k/train.xyz` indices `0:32` (`286` atoms), Adam, CUEQ enabled with `optimize_all=True`, `C=64`, `L=1`, `num_interactions=2`, `correlation=3`, warmup `5`, repeats `30`, and modes `position_eager,edge_compile`. This is a fixed-batch profiler, not a full multi-batch training benchmark.

| Mode | Setup | Total mean | Forward/loss mean | Backward+clip mean | Optimizer mean | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `position_eager` | `0 ms` | `12.309 ms` | `6.350 ms` (`51.6%`) | `5.743 ms` (`46.7%`) | `0.214 ms` (`1.7%`) | ordinary conservative force training |
| `edge_compile` | `26735 ms` | `4.392 ms` | `1.883 ms` (`42.9%`) | `2.295 ms` (`52.3%`) | `0.211 ms` (`4.8%`) | compiled edge-force closure, gate accepted |

The fixed-batch hot-step speedup is about `2.8x` (`12.309 -> 4.392 ms`). This confirms that the compiled edge-force subpath is not inherently weak: it materially reduces both the forward/loss phase and the later backward/clip phase. The remaining hotspot inside the compiled case is still backward/clip, which accounts for about half the step even after compile. Optimizer time stays around `0.21 ms`, so optimizer implementation micro-optimizations cannot explain end-to-end training speed.

The setup cost was `26.7 s`, which explains why short or heavily validated training runs show much smaller wall-clock gains. Real RECIO training also adds dataloader transfer, Python logging, validation, checkpoint/export, and changing batch shapes. Therefore the next optimization target is not cache-key correctness, which is already stable for dynamic cache, but reducing the work outside the compiled closure and validating whether the Inductor graph path can keep this fixed-batch hot-step advantage across real multi-batch epochs without accuracy drift.


### 2026-07-03 Multi-Batch Setup-Gate Diagnostic

A follow-up epoch profiler run showed that the fixed-batch speedup above is not yet a safe multi-batch training conclusion. Job `619940` attempted RECIO `0:256`, batch size `32`, three epochs, Adam and HybridMuon, `position_eager` and `edge_compile`, CUEQ minus-linear, shape cache, and Inductor enabled. The Adam `position_eager` child completed, but the Adam `edge_compile` child failed on the second real batch. Energy, forces, and loss still matched (`<= 5e-7` scale), but many parameter-gradient checks failed (`1e-3` to `1e-2` scale, with `readouts.1.linear_2.weight` reported as missing on one side).

The failure is not specific to CUEQ or Inductor. Single-batch diagnostics passed for `no-CUEQ + Inductor` (`619949`), `CUEQ minus-linear + FX-only` (`619950`), and `CUEQ minus-linear + Inductor` on the failing batch as the first batch (`619951`). However, two-batch runs failed for CUEQ and no-CUEQ, FX-only and Inductor, even with `lr=0`, cache clearing on miss, compiler reset, and a diagnostic `--skip-training-step` mode. This points to the repeated setup-gate procedure itself: running a full position-vs-edge gradient gate on batch 0 in the live model process is enough to make the next setup gate unreliable, even when the actual training backward/optimizer step is skipped.

Two code-level safeguards were added to the epoch profiler while debugging this: cache-miss setup gates now clear model gradients before the real step, and the SAI epoch template now defaults to `mace_develop` without SAI-disallowed `--ntasks-per-node`. Additional diagnostic knobs are available only for profiling: `--edge-reset-compile-state`, `--edge-clear-cache-on-miss`, and `--skip-training-step`. Local tests for these profiler/template changes passed (`32 passed, 1 warning`).

The practical conclusion is that the current repeated setup-gate epoch profiler should not be used as the acceptance benchmark for 8k/20w training. The next implementation step is to decouple correctness validation from the live training model: either validate new compile signatures in an isolated child process/fresh model snapshot, or switch to a production training path with bounded setup gates plus separate fixed-probe parity checks. Until that is done, fixed-batch hot-step numbers remain useful, but multi-batch compile+CUEQ training speed and accuracy are not yet established.

### 2026-07-03 C128/L2 200k WSD Long Run and OC20NEB FPS Setup

A larger RECIO/8k single-stage WSD check now gives the strongest end-to-end evidence for the full-Inductor CUEQ path. Both jobs used `mace_develop`, CUEQ `optimize_all=True`, Adam, `num_channels=128`, `max_L=2`, `correlation=3`, batch size `8`, `max_num_epochs=211` (about `200450` optimizer updates), `train_tf32=True`, and `train_amp_dtype=none`.

| Mode | Job | Slurm elapsed | MaxRSS | Final valid E/F | Final test E/F | Compile health |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CUEQ eager | `620948` | `02:25:08` | `3778784K` | `43.8` meV/atom / `221.3` meV/A | `49.9` meV/atom / `114.8` meV/A | n/a |
| CUEQ + full Inductor edge-force compile | `620950` | `01:34:47` | `6390748K` | `41.0` meV/atom / `221.1` meV/A | `47.7` meV/atom / `114.4` meV/A | `fallbacks=0`, `runtime_recompiles=0`; periodic parity and fixed probe accepted |

This is a materially better result than the earlier C64/L1 200k Stage-Two matrix: for the larger C128/L2 model, compile improves end-to-end wall time by about `1.53x` while preserving validation/test force accuracy and slightly improving energy MAE in this seed. The tradeoff is memory: full Inductor increased MaxRSS from about `3.78 GB` to `6.39 GB` on the V100 stack. The conclusion is now model-size dependent rather than uniformly modest: CUEQ already accelerates small C64/L1 runs enough that compile adds little, but larger equivariant workloads expose enough compiled force/backward work for a clear end-to-end gain.

A new OC20NEB FPS benchmark path was also added for cross-dataset comparison against the DeepMD DPA4 benchmark split. `scripts/benchmarks/oc20neb_fps/convert_deepmd_mixed_to_extxyz.py` converts the DeepMD mixed split at `reference_repos/deepmd-kit/benchmarks/dpa4_oc20neb_fps/oc20neb_fps5k_train_random50_valid/deepmd_mixed` into MACE `train.extxyz` and `valid.extxyz`, using `real_atom_types.npy` rather than the placeholder `type.raw`. The real conversion produced `5000` train frames and `10000` valid frames, with `28-100` atoms. A smoke compile job `621336` passed on the 2-frame converted subset with full CUEQ, full Inductor graph, strict setup gate, parity checks, and fixed probe: `compiled=2`, `cache_hits=1`, `fallbacks=0`, `runtime_recompiles=0`, parity `2/2`, fixed probe `2/2`.

Two formal OC20NEB L=1/C64 jobs are now running from `runs/oc20neb_fps_l1c64_wsd_20k_20260703`: eager CUEQ Adam WSD job `621339` and full-Inductor CUEQ Adam WSD job `621340`. Both use `batch_size=8`, `max_num_epochs=32` (about `20k` optimizer updates), independent `valid.extxyz`, TF32 enabled, AMP disabled, and full CUEQ. The important SAI submission lesson from the smoke setup is that explicit `sbatch --export=ALL,...` caused immediate `CANCELLED by 0` jobs with no stdout/stderr on this cluster. Environment-prefix submission, for example `RUN_ROOT=... EDGE_FORCE_COMPILE=True sbatch scripts/benchmarks/oc20neb_fps/run_mace_oc20neb_fps_sai.sh`, works and should be used for these wrappers.
