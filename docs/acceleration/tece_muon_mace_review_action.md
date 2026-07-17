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
   `OptimSpec` route contracts, including semantic axes, matrix/batch axes,
   size/aspect gates, LR scale, weight decay, matrix structure, and spec version,
   are included per parameter in route manifests.
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
- The more defensible comparison is `routing="module"` or a conservative TACE
  route with explicit semantic manifests.
- If broad TACE Muon improves 20k/200k metrics, the next question is which
  declared channel blocks drive the gain, not whether every matrix-like tensor
  should stay on Muon.
- Muon bulk training followed by lower-LR AdamW tail calibration is consistent
  with the review and should be tested after the no-stage 200k AdamW vs
  HybridMuon run finishes.
