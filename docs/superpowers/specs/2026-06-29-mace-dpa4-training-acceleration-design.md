# MACE DPA4-Inspired Training Acceleration Design

## Scope

This design covers the first implementation phase for accelerating MACE training by borrowing training-system ideas from DeePMD-kit DPA4/SeZM without copying the DPA4 model architecture into MACE.

The first phase targets ordinary MACE/MACELES small-to-medium dataset training on the RECIO 8k benchmark:

- Dataset root: `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k`
- Training file: `train.xyz`
- Baseline config: `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/config.yaml`
- Hardware target: SAI single V100 job, `4V100` partition, `--gpus-per-node=1`, `--qos=improper-gpu`
- Environment: `source /opt/envs/anaconda3.env && conda activate mace_env`
- Baseline model: `MACELES`, 2 interactions, 128 channels, `max_L=1`, `correlation=3`, `r_max=5.0`, `batch_size=32`, `default_dtype=float32`, `enable_cueq=true`

The first phase excludes GPU D3 and a full training-framework rewrite. Those remain future phases, but the interfaces added here must not block later nvalchemi D3 integration or a TACE-style trainer split.

## Existing Evidence

The MACE repository is checked out at upstream `ACEsuit/mace` `develop`, commit `a7c5ec5 update cuda test`.

Reference repositories were cloned under `/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/reference_repos`:

- `deepmd-kit`, commit `a9bcbc50 feat(pt): add custom save behaviors (#5589)`
- `nvalchemi-toolkit`, commit `526cb8b Update AGENTs.md file (#121)`
- `tace`, commit `c669bee opt code`

Relevant reference points:

- DeePMD-kit DPA4/SeZM has explicit training compile machinery in `deepmd/pt_expt/train/training.py` and compile workarounds in `deepmd/pt/utils/compile_compat.py`.
- DeePMD-kit has `HybridMuonOptimizer` in `deepmd/pt/optimizer/hybrid_muon.py`.
- TACE uses a more modern Lightning/DataModule/strategy split and also carries a HybridMuon optimizer.
- MACE already has inference/forward compile helpers in `mace/tools/compile.py` and compile tests in `tests/test_compile.py`.
- MACE training currently constructs optimizers in `mace/tools/scripts_utils.py` and performs single-batch training steps in `mace/tools/train.py`.

## Design Principles

MACE must remain a MACE model. The acceleration work must preserve the physical contract:

- Energy is a scalar invariant of positions, species, and graph topology.
- Forces remain conservative forces from `-dE/dR`.
- Equivariance/invariance are model-architecture constraints, not optional implementation details.
- Stress and virials must keep the existing sign, unit, and autograd conventions.

DPA4 acceleration ideas are borrowed only where they are algorithmically compatible:

- compile boundary management,
- dtype/autocast policy,
- optimizer parameter routing,
- benchmark and fallback discipline.

The implementation must be opt-in and independently testable. The user must be able to enable or disable compile, bf16 AMP, and HybridMuon separately.

## Approach

Implement a staged opt-in acceleration path with three independent features:

1. `train_compile`
2. `train_amp_dtype=bf16`
3. `optimizer=hybrid_muon`

The first useful combinations are:

- fp32 Adam cueq baseline,
- bf16 Adam cueq,
- fp32 HybridMuon cueq,
- bf16 HybridMuon cueq,
- compile variants after the non-compile variants pass smoke tests.

This avoids conflating numerical drift from dtype, optimizer dynamics, and compiler graph changes.

## Feature 1: Training Compile

### Goal

Add an opt-in training compile path that speeds up the repeated forward/loss/backward workload without compiling checkpointing, logging, data loading, optimizer state updates, or unsupported dynamic Python control flow.

### Boundary

The first compile boundary should be conservative:

- Prepare the model using the existing `mace.tools.compile.prepare` behavior where applicable.
- Compile after optional cuEquivariance conversion, because RECIO/8k baseline uses `enable_cueq=true`.
- Compile either the model forward region or a small wrapper around model forward plus loss, but keep the optimizer step outside Dynamo.
- Do not compile the LBFGS path.
- Disable or fall back automatically for unsupported combinations.

### Constraints

MACE force training calls autograd through positions. The compile path must verify:

- energy parity,
- force parity,
- gradient existence for trainable parameters,
- no detach that breaks force-loss gradients,
- no graph mutation that invalidates cueq conversion.

DeepMD-kit's DPA4 compile code is useful as a design reference for:

- disabling DDP optimizer graph splitting for inner compiled regions,
- keeping optimizer step outside compile,
- logging explicit fallbacks,
- guarding PyTorch-version-sensitive behavior.

It should not be copied wholesale because MACE's graph and force path are structurally different.

## Feature 2: bf16 AMP

### Goal

Add partial bf16 training acceleration that reduces memory and improves throughput while preserving MACE's force and energy accuracy.

### Policy

Use fp32 master weights. Use autocast only around safe forward regions.

Keep these in fp32:

- positions and displacement tensors used for force/stress autograd,
- energy reductions and output energy tensor before loss,
- loss computation,
- gradient clipping,
- optimizer state,
- EMA/SWA state updates,
- atomic reference energies,
- stress/virial-sensitive paths.

Allow bf16 where runtime checks show it is stable:

- dense radial/readout MLP operations,
- selected tensor operations that PyTorch/cueq handle reliably on the target environment,
- non-accumulating intermediate activations.

The user-facing interface should allow `--amp_dtype bf16` or YAML equivalent. `none` remains the default.

### Safety

The first implementation must include a parity smoke test comparing fp32 and bf16 on a fixed RECIO mini-batch:

- finite loss,
- finite forces,
- no missing parameter gradients,
- energy/force drift logged with thresholds.

bf16 should be disabled automatically on devices or PyTorch builds that do not support it.

## Feature 3: HybridMuon Optimizer

### Goal

Add `hybrid_muon` as a MACE optimizer option that combines Muon-style matrix updates for safe dense parameters with Adam/AdamW for sensitive or non-matrix parameters.

### Routing

Do not route by tensor dimensionality alone. MACE parameter names and module ownership must drive routing.

Adam/AdamW route:

- biases,
- 1D parameters,
- normalization, scale, shift, and gate parameters,
- atomic energies and reference/baseline terms,
- embedding tables,
- LES-specific selector/core weights unless explicitly proven safe,
- equivariant contraction tensors whose axes encode irreps/correlation structure,
- any parameter whose owner module is ambiguous.

Muon route:

- clearly dense 2D MLP-like weights in radial networks,
- clearly dense 2D readout/fitting weights,
- other named dense matrix parameters only after inspection and test coverage.

The optimizer must log a route summary at startup:

- number of parameters and elements per route,
- names of Muon-routed tensors,
- names of Adam-routed tensors that would have been Muon by shape alone.

### State and Compatibility

Optimizer state must remain checkpoint-compatible with MACE's existing checkpoint handler. Restart behavior must be tested.

Muon math can borrow the high-level DeePMD-kit/TACE algorithm, but the MACE implementation should be locally owned:

- simple, readable PyTorch implementation first,
- optional Triton/flash path only after correctness and speed are established,
- fp32 Adam moments,
- stable behavior under gradient clipping, EMA, SWA, and cueq.

## RECIO/8k Benchmark Design

### Baseline

The existing full baseline log reports final stage-two validation near:

- `MAE E`: `22.1 meV/atom`
- `MAE F`: `260.2 meV/A`

The historical single-step training time in `results/RECIO-8k_run-123_train.txt` is roughly `0.06-0.07 s` for many early optimizer steps.

### Fast Gate

For development and sbatch smoke tests:

- Use the same `train.xyz`, split seed `123`, and 10% validation.
- Run short jobs with reduced epochs, initially 5-20 epochs.
- Keep batch size 32 unless testing memory headroom.
- Record wall time, mean step time after warmup, max GPU memory, and validation metrics.
- Treat any NaN, missing gradient, checkpoint failure, or force parity failure as a hard fail.

### Accuracy Gate

After fast gates pass:

- Run longer RECIO/8k jobs for promising combinations.
- Compare validation curves against fp32 Adam cueq baseline.
- Final acceptance requires no material regression in validation energy/force MAE and no degraded generalization signal on the held-out 800 validation structures.

The first acceptance threshold is:

- speed or memory improvement must be measurable on SAI V100,
- validation `MAE E` and `MAE F` should remain within normal seed-to-seed noise of the baseline,
- any improvement claim must cite the exact sbatch logs and parsed metric files.

## SAI Execution

Use SAI Slurm scripts based on the existing RECIO 8k script:

- `#SBATCH --partition=4V100`
- `#SBATCH --nodes=1`
- `#SBATCH --ntasks=1`
- `#SBATCH --gpus-per-node=1`
- `#SBATCH --ntasks-per-node=1`
- `#SBATCH --qos=improper-gpu`

The script should keep the submit environment clean and activate conda inside the job:

```bash
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
source /opt/envs/anaconda3.env
conda activate mace_env
python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/mace/cli/run_train.py --config=config.yaml
```

The implementation plan should generate separate benchmark directories rather than modifying the historical RECIO/8k run in place.

## Tests

Add narrow unit tests before broad training runs:

- Optimizer routing test with a small synthetic MACE-like module.
- HybridMuon checkpoint save/load test.
- AMP smoke test on CPU-disabled or CUDA-guarded path.
- Compile smoke test using existing `tests/test_compile.py` patterns.
- Combined cueq/compile/amp tests should skip clearly when dependencies or CUDA are unavailable.

Add RECIO sbatch tests as integration evidence, not as unit tests.

## Error Handling

Each feature must fail closed:

- If compile is unsupported, log the reason and either fall back only when explicitly allowed or raise a clear error.
- If bf16 is unsupported, raise before training starts.
- If HybridMuon sees unknown parameter groups, route them to Adam and log the route.
- If cueq and compile conflict, keep cueq baseline behavior and disable compile unless the user explicitly requests hard failure.

## Future Phases

GPU D3 with nvalchemi:

- Add a separate optional dispersion backend using `nvalchemiops` DFT-D3 kernels.
- Preserve MACE's existing dispersion semantics and unit conventions.
- Do not couple D3 kernel work to the training acceleration changes.

Training framework refactor:

- Use TACE's Lightning/DataModule/strategy structure as a reference.
- Decompose MACE training into config parsing, data module, model construction, precision policy, optimizer factory, train loop, checkpointing, and benchmark harness.
- Do this only after the first acceleration path has isolated correctness and performance evidence.

## Open Implementation Questions

These questions must be answered by implementation experiments, not assumptions:

- Whether cueq-converted MACE forward is stable under `torch.compile` for RECIO/8k force training.
- Whether bf16 on the SAI V100 stack improves speed enough to justify any numerical drift.
- Which exact MACE parameter names are safe for Muon after inspecting the instantiated RECIO/8k model.
- Whether HybridMuon needs a lower default learning rate than Adam for this MACELES case.

## Acceptance Criteria

The first phase is complete only when all of the following are true:

- `optimizer=hybrid_muon` works for RECIO/8k and restarts from checkpoint.
- bf16 AMP works for RECIO/8k without NaNs or missing gradients.
- training compile works or has a documented, tested fallback for unsupported cueq/PyTorch combinations.
- The combinations `bf16+Adam`, `fp32+HybridMuon`, and `bf16+HybridMuon` have SAI sbatch evidence.
- At least one compile-enabled combination has SAI sbatch evidence or a precise upstream/toolchain blocker with a minimal reproducer.
- Accuracy and generalization are compared against the RECIO/8k baseline validation metrics.
- Historical RECIO data and logs are not overwritten.
