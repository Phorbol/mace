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

