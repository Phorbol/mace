# Reference Ops and Training Frameworks

This note records the first source-level comparison against three external
references for the DPA4 training-acceleration branch.

Reference revisions:

- NVIDIA ALCHEMI Toolkit-Ops: `2ae41e2`
  (`/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/reference_repos/nvalchemi-toolkit-ops`)
- DeePMD-kit: `6c3b985`
  (`/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/reference_repos/deepmd-kit`)
- TACE: `80d68e9`
  (`/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/reference_repos/tace`)

The goal is not to vendor these projects wholesale. The useful pattern is to
reuse stable op boundaries and training-control ideas while preserving MACE's
model semantics as the single source of truth.

## NVIDIA Toolkit-Ops

### Neighbor List

Relevant source:

- `nvalchemiops/torch/neighbors/__init__.py`
- `nvalchemiops/torch/neighbors/_dispatch.py`
- `nvalchemiops/torch/neighbors/_autograd.py`
- `nvalchemiops/torch/neighbors/batch_cell_list.py`
- DeePMD adapter: `deepmd/pt/utils/nv_nlist.py`

Capabilities that map well to MACE:

- Torch-facing `neighbor_list()` supports `naive`, `cell_list`,
  `cluster_tile`, batched variants, dual cutoff, half lists, partial source
  rows, explicit preallocated buffers, and `rebuild_flags`.
- It exposes both dense matrix output and COO output. The dense matrix form is
  the preferred performance path; COO conversion is explicitly called out as a
  potential overhead.
- Strategy selection is host-side and can be run before compilation via
  `suggest_neighbor_list_method()` / `estimate_neighbor_list_costs()`.
  Inside compiled code, the method should be explicit.
- Optional `return_distances` and `return_vectors` are differentiable. The Warp
  forward enumerates pairs, but backward reconstructs
  `r = x_j - x_i + shift @ cell` in pure Torch. That is important because force
  training needs higher-order derivatives through geometry.
- Toolkit-Ops uses `torch.library.custom_op` plus fake/no-op fake registration
  for many launchers, matching the custom-op boundary recommended for our
  compile work.

Immediate MACE reuse plan:

1. Add an optional MACE neighbor backend behind a feature flag, not as the
   default. Start with data preprocessing / inference graph construction where
   current MACE uses CPU `matscipy.neighbours`.
2. Convert Toolkit-Ops dense matrix output to MACE's current `edge_index`,
   `unit_shifts`, and `shifts` tensors. Keep the original MACE forward
   unchanged.
3. For training, treat neighbor enumeration as non-differentiable and rebuild
   differentiable edge vectors from input positions and cell, following the
   `_autograd.py` pattern.
4. Benchmark three regimes separately: small OC20NEB-like batches, dense
   high-cutoff systems, and large periodic systems. Do not assume GPU neighbor
   construction wins for the small-batch training path.

Risks:

- Toolkit-Ops `cluster_tile` requires fully periodic, sorted contiguous batches
  and `float32`-oriented CUDA paths. MACE datasets can include non-periodic or
  mixed cell cases.
- MACE currently expects COO edges. A naive matrix-to-COO conversion can erase
  much of the win. The better long-term route is to keep matrix output for
  selected fused geometry/radial kernels, but that requires model-side support.
- `method=None` may sync to host for shape/metadata. Production use should
  preselect method and capacity outside compiled regions.

### Ewald / PME for MACELES

Relevant source:

- `nvalchemiops/torch/interactions/electrostatics/ewald.py`
- `nvalchemiops/torch/interactions/electrostatics/pme.py`
- `nvalchemiops/torch/interactions/electrostatics/parameters.py`
- stress tests under `test/interactions/electrostatics/`

Useful properties:

- Public Ewald/PME APIs use energy autograd as the training contract:
  forces are expected to be obtained by differentiating total energy.
- PME keeps FFT-dependent workflow in Torch while Warp provides launchers for
  kernels such as real-space terms, corrections, and virial pieces.
- Ewald and PME support batched systems, charge gradients, virial/stress tests,
  and finite-difference strain checks. The tests include `gradgradcheck` and
  stress-loss gradients for real-space cell paths.
- Direct force/virial outputs are retained mostly as no-autograd inference/MD
  escape hatches and are deprecated for training-style usage.

MACELES implication:

- The most viable reuse is a long-range electrostatic module that takes MACE
  predicted charges/multipoles plus cell and neighbor data, returns an energy
  term, and lets MACE's existing `get_outputs()` derive forces/stress by
  autograd.
- We should not wire direct PME force outputs into training loss unless a
  dedicated second-derivative validation suite is added.
- The first implementation target should be scalar charges and periodic cells.
  Multipoles/quadrupoles are a later target because they add more orientation
  and mixed-derivative surface area.

Validation required before relying on it:

- charge-gradient check;
- force gradient parity against finite differences;
- strain/stress finite-difference check;
- force-loss and stress-loss second-derivative checks;
- parity across single-system and batched-system paths.

## DeePMD-kit Training Framework

Relevant source:

- `deepmd/dpmodel/utils/learning_rate.py`
- `deepmd/utils/compat.py`
- `deepmd/pt/train/training.py`
- `deepmd/pt/train/wrapper.py`
- `deepmd/pt/loss/ener.py`
- `deepmd/pt/optimizer/hybrid_muon.py`

Training-control details worth absorbing:

- The resolved training unit is `num_steps`. Epoch input is only one way to
  derive total update count.
- Learning-rate warmup belongs to the `learning_rate` config, not to the
  generic `training` section. DeePMD migrates legacy warmup keys and rejects
  conflicting definitions.
- Schedulers are value functions over global step: `exp`, `cosine`, and `wsd`
  all share linear warmup via `BaseLR`.
- The PyTorch backend wraps this value function in `LambdaLR` with
  `last_epoch=start_step - 1`, so checkpoint resume continues the schedule by
  global step.
- Energy/force/virial loss prefactors depend on current learning rate:
  `pref = limit + (start - limit) * current_lr / starter_lr`. This gives a
  smooth stage transition coupled to LR decay.
- Optimizer construction passes current runtime named parameters into
  `HybridMuonOptimizer.set_param_names()`, avoiding checkpoint-persistent
  Python object-id routing.
- DeepMD's latest HybridMuon supports match-RMS scaling, Magma, shape padding,
  foreach helpers, and FSDP/DTensor guard paths.

MACE implications:

- Keep MACE's update-based training mode and make it the preferred path for
  acceleration experiments.
- Add a first-class LR schedule config with `warmup_steps` or `warmup_ratio`,
  `warmup_start_factor`, `start_lr`, `stop_lr`/`stop_lr_ratio`, and schedule
  type `exp|cosine|wsd`.
- Make resume semantics explicit: schedule state must derive from completed
  optimizer updates, not epoch count.
- Add an optional smooth loss-prefactor schedule for energy/force/stress:
  preserve the current hard stage switch for ablations, but introduce a
  DeePMD-style LR-coupled schedule for production runs.
- Keep HybridMuon config hash/checkpoint metadata aware of LR scale mode,
  match-RMS coefficient, Magma settings, and stage transitions.

## TACE Training Framework

Relevant source:

- `example/train/tace.yaml`
- `tace/lightning/lit_model.py`
- `tace/lightning/trainer.py`
- `tace/utils/lr_scheduler/warmup.py`
- `tace/utils/lr_scheduler/wsd.py`
- `tace/lightning/skip.py`
- `tace/utils/optimizer/hybrid_muon.py`

Training-control details worth absorbing:

- Config uses Hydra `_target_` for optimizer and scheduler. That makes ablation
  matrices easy to express without changing training code.
- Scheduler metadata includes `extra.interval: step|epoch`, `frequency`, and
  `monitor`. Step-level scheduler use is explicit.
- The example provides both validation-driven `ReduceLROnPlateau` and
  update-driven cosine warmup restart / WSD alternatives.
- TACE uses an EMA callback by default in the example, with checkpoint loading
  capable of choosing EMA weights.
- It defines a composite validation metric with configurable weights over
  energy, force, and stress metrics.
- It has a LossSkipController for NaN or large-loss skip with DDP-wide
  agreement. This is useful as an optional debug guard, not as a default
  because it changes effective sampling.
- Optimizer setup separates weight-decay and no-decay groups, and passes
  `named_parameters` to HybridMuon.

MACE implications:

- Add an experiment-config layer for optimizer/scheduler/loss-stage ablations.
  This does not require adopting Lightning; a typed config object plus CLI/env
  mapping is enough.
- Add a composite validation metric for model selection in OC20NEB experiments,
  including stress once stress labels are used.
- Expose step-level checkpointing and eval cadence consistently in logs.
- Consider EMA as an explicit experimental option after HybridMuon baseline
  conclusions are stable.

## Proposed Priority Order

1. Training framework cleanup:
   introduce a reusable update-based LR schedule object with warmup and WSD,
   and use it in the current MACE training loop.
2. Loss schedule:
   support both hard stage switch and smooth LR-coupled E/F/stress prefactors.
   Re-run the 20k then 200k OC20NEB AdamW vs HybridMuon matrix under both.
3. HybridMuon polish:
   compare current MACE implementation against latest DeepMD/TACE HybridMuon
   for foreach coverage, Magma defaults, and DTensor/FSDP guard behavior.
4. NVIDIA neighbor backend prototype:
   optional preprocessing/inference backend first; training path only after
   parity and speed are measured.
5. MACELES long-range prototype:
   scalar-charge Ewald/PME energy-only training contract first; forces/stress
   derived by autograd and validated by finite differences.

## Non-goals For The First Integration

- Do not replace MACE forward with a Toolkit-Ops or DeepMD-style forward.
- Do not make Toolkit-Ops neighbor construction the default without OC20NEB and
  ABACUS dataset benchmarks.
- Do not use direct PME force or virial outputs in training before
  second-derivative tests pass.
- Do not adopt Lightning wholesale just to get scheduler features.
