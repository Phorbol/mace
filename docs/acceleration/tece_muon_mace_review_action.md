# TECE Muon MACE Review Action Notes

Date: 2026-07-17

Source review: `/home/gengjianrui/bin/TECE_Muon_MACE_review.md`.

This note records how the TECE/Muon review changes the immediate engineering
priority for `dpa4-training-accel-20260703`. It is intentionally shorter than
the source review and focuses on current branch facts and actionable decisions.

## Current Branch Facts

HybridMuon already has several correctness fixes and useful building blocks:

- `OptimSpec(route, matrix_axes, batch_axes, slice_specs, lr_scale, weight_decay)`
  exists and validates matrix/batch axis coverage.
- e3nn flat-weight instruction reconstruction uses stable parameter names rather
  than Python object ids.
- runtime-only metadata such as `matrix_specs`, per-parameter LR scales, and
  matrix layouts is stripped from optimizer `state_dict()` and restored from the
  live optimizer on load.
- Muon-routed tensors without a valid `MatrixSpec`, `OptimSpec` layout, or
  matrix view now raise instead of silently skipping the update.
- match-RMS scaling, rectangular canonicalization, same-short-side batching,
  Magma-lite warmup controls, per-parameter LR scales, and DTensor/sharded
  rejection are present.

The main remaining issue is semantic coverage, not mechanical plumbing:
`routing="tace"` no longer falls back to a broad matrix-like Muon route, but
the module-declared route contract still needs broader architecture snapshots
and future complex-Muon support before TECE/TACE/DPA4 semantics are complete.

## Accepted Review Conclusions

1. Muon should only operate on true channel-input to channel-output matrices.
   Degree, parity, local `m`, CG path, correlation order, species, focus, head,
   expert, and low-rank coefficient axes must be independent blocks unless a
   module explicitly declares otherwise.
2. `routing="module"` is the correct long-term default. It should route
   declared matrices through `OptimSpec` and send unknown tensors to AdamW.
3. `routing="tace"` should become a deprecated conservative compatibility mode,
   not a broad matrix-like heuristic.
4. Final readout and calibration parameters should default to AdamW. Hidden
   readout-like MLP matrices can use Muon only through a module declaration.
5. Complex SO(2) `w1_w2` parameters should not be split into independent real
   and imaginary Muon updates. Until complex Muon is implemented, they should be
   routed to AdamW.
6. Route manifests should become JSON-compatible artifacts that can be compared
   on resume. Shape or route drift should fail by default.
7. The current edge-force compile path should continue to reject edge-mode
   stress/virial training. Position-gradient compile can handle stress/virial
   tensor losses, but edge-force-virial needs explicit edge-vector/cell/PBC ABI
   and finite-difference parity before enabling.

## Current Compile Capability Reading

The current `training_compile.py` capability table is conservative in the way
the review asks for:

- `WeightedEnergyForcesLoss`, `WeightedForcesLoss`, and
  `WeightedEnergyForcesL1L2Loss` are edge-force supported.
- `WeightedEnergyForcesStressLoss`, `WeightedHuberEnergyForcesStressLoss`,
  `UniversalLoss`, and `WeightedEnergyForcesVirialsLoss` have
  `edge_force_supported=False` but `compiled_tensor_loss_supported=True`.
- `TrainingCompileManager` falls back to eager if `stress` or `virials` is
  requested while `force_gradient_mode != "positions"`.
- `_EDGE_FORCE_INPUT_KEYS` still lacks explicit `shifts` and `cell`; these are
  only present in `_POSITION_FORCE_INPUT_KEYS`.

So it is accurate to say: stress/virial losses can use the compiled tensor-loss
adapter in position-gradient mode, but edge-force mode is not yet an
edge-force-virial implementation.

## Implementation Order After The Running 200k Job

Slurm job `676508` now runs the missing broad-TACE HybridMuon 200k leg from a
frozen detached checkout at commit `4a21a2f`, so current development can proceed
without changing that experiment source. That running result remains a legacy
broad-routing stress test, not the final TECE/DPA4 optimizer policy.

After freezing that experiment source, apply these changes:

1. Add semantic fields to optimizer specs, initially backwards-compatible:
   `semantic_axes`, `matrix_structure`, `min_matrix_dim`, `max_aspect_ratio`,
   and `spec_version`. Status: fields landed; `min_matrix_dim` is explicit
   opt-in by default for backward compatibility with existing module specs.
2. Add validation rules that reject Muon for semantic axes such as `degree`,
   `m`, `parity`, `path`, `correlation`, `species`, `focus`, `head`, and
   `expert` when they appear as matrix axes. Status: initial validation landed,
   along with non-real matrix-structure rejection and explicit size/aspect gates.
3. Change `routing="tace"` fallback from broad `tace-matrix-muon` to a
   conservative allowlist: module-declared specs, known e3nn flat instruction
   specs, and safe dense hidden matrices only. Status: initial fallback removal
   landed; unknown tensors now default to AdamW.
4. Prefer `routing="module"` in new experiment scripts and manifests; keep
   `routing="tace"` only for compatibility/ablation runs with an explicit
   warning in route summaries. Status: CLI and `get_optimizer()` now default
   HybridMuon to module-declared routing, `routing="module"` fails early when
   the model's `named_modules()` are unavailable, the current OC20NEB fullcase/FPS
   scripts default new HybridMuon runs to module routing, and route summaries now
   warn whenever deprecated `routing="tace"` compatibility routes are present;
   AdamW-routed complex SO(2) module specs retain `matrix_structure` in route
   manifests for auditability.
5. Emit a route manifest/hash into experiment artifacts and checkpoint metadata.
   Compare it on resume and fail on route or shape drift. Status: checkpoint
   route manifest/hash landed in `HybridMuon.state_dict()`; `load_state_dict()`
   now rejects saved route hashes that differ from the current optimizer route,
   except for its own recorded Stage Two Muon-to-Adam/AdamW route transition;
   Stage Two manifests retain former-Muon matrix views so slice/layout drift is
   still rejected on resume;
   `get_optimizer()` now writes a rank-0 JSON route manifest artifact under
   `log_dir` (or `model_dir`) for each HybridMuon run; module-declared
   `OptimSpec` route contracts, including semantic axes, normalized
   matrix/batch axes, size/aspect gates, LR scale, weight decay, matrix
   structure, and spec version, are included per parameter in route manifests.
6. Add architecture-level route snapshot tests for MACE/e3nn, CUEQ-converted
   flat weights, and any future TACE/TECE/DPA4 modules. Status: initial MACE
   safe-default route manifest snapshot landed with per-parameter reason and
   owner module type recorded in the checkpoint manifest; e3nn flat-weight path
   metadata is now recorded as `[instruction_index, i_in, i_out]` in matrix views;
   module-declared CUEQ-like slice specs are snapshot as `module_slice_spec`
   views with path metadata so they are distinguishable from e3nn-reconstructed
   flat specs on checkpoint resume; a DPA4/SO2-like module snapshot now records
   `focus` and `m` as batch axes while only channel axes form Muon matrices, and
   keeps scalar/path-like coefficients on AdamW.
7. Only after the Muon route contract is stable, revisit edge-force-virial:
   document edge-vector sign, add explicit `edge_vec`, `shifts`, `cell`, and PBC
   inputs, and validate force/virial signs with position and strain finite
   differences.

## Experiment Interpretation

The review also changes how to interpret current HybridMuon results:

- Full broad `routing="tace"` experiments are useful as stress tests, but they
  should not be treated as the final TECE/DPA4 optimizer policy.
- The completed no-stage 200k broad-TACE stress test supports that caution:
  `cueq_adamw` reached 45.01 meV/atom energy MAE and 28.00 meV/A force MAE,
  while `cueq_hybrid_muon_tace` reached 77.78 meV/atom and 41.18 meV/A and ran
  about 6.6% slower per update.
- The more defensible comparison is `routing="module"` or a conservative TACE
  route with explicit semantic manifests. New default OC20NEB demo cases now use
  explicit `hybrid_muon_module` and `cueq_hybrid_muon_module` names so summaries
  cannot be confused with deprecated broad routing.
- The review-aligned 20k CUEQ comparison completed from committed source
  `020e12d` with Stage One `E:F=1:100` and Stage Two at 15k updates with
  `E:F=100:1`. The stable machine-readable comparison is now generated in
  `runs/oc20neb_fullcase200_ef_20k/module-stage-cueq-020e12d-20260717-20k/matrix_comparisons.json`:
  `cueq_hybrid_muon_module` reached final force MAE 36.29 meV/A versus
  53.88 meV/A for `cueq_adamw` (delta -17.59 meV/A, ratio 0.674), but final
  energy MAE was 150.06 meV/atom versus 33.70 meV/atom (delta +116.36 meV/atom,
  ratio 4.45). It ran about 0.905x as fast by train-metrics updates/s, with
  optimizer step time about 7.97x higher and no meaningful memory increase.
- If module-declared Muon improves 20k/200k metrics, the next question is which
  declared channel blocks drive the gain. If it does not, sweep LR scale mode,
  Muon LR factor, and Stage Two route before broadening the routed parameter set.
- Muon bulk training followed by an AdamW tail is now measured on the same
  committed source `ae243d8` in
  `runs/oc20neb_fullcase200_ef_20k/cueq-module-tail-ae243d8-20260717-20k/`.
  The `cueq_hybrid_muon_module_adamw_tail` log confirms the intended Stage Two
  transition at 15k updates with `Switched 1 HybridMuon param group from Muon
  to AdamW for Stage Two`. The ablation is therefore valid, but the result is a
  mixed/negative one: final energy improves substantially versus module-Muon
  keep (`51.97` vs `150.06` meV/atom), yet remains worse than `cueq_adamw`
  (`33.70` meV/atom), and final force becomes worse than both (`61.64` meV/A
  versus `36.29` for module-Muon keep and `53.86` for AdamW). The best force
  before Stage Two remains `25.20` meV/A because the first 12k updates match the
  module-Muon keep trajectory. AdamW-tail also remains slower than AdamW by
  train-metrics throughput (`15.12` vs `16.12` updates/s, `0.938x`) while being
  slightly faster than module-Muon keep (`15.12` vs `14.77` updates/s) after the
  Stage Two switch reduces optimizer-step cost. This rules out a simple
  keep-Muon-then-AdamW-tail recipe as the current answer; the next ablation
  should address Muon LR scale/match-RMS and which declared readout/embedding
  matrices are allowed to remain on Muon, rather than broadening routing.
- The AdamW-tail run also exposed a checkpoint/evaluation correctness bug: after
  Stage Two changes the optimizer route, result evaluation tried to load the
  pre-Stage-Two checkpoint including optimizer state and correctly hit
  `HybridMuon route manifest hash mismatch`. Full training resume should keep
  that strict failure mode, but result evaluation only needs model weights. The
  checkpoint loader now supports model-only loads, and `run_train.py` uses that
  path for post-training Stage One/Stage Two evaluation.
