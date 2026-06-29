# Edge-Vector Force Training Compile Integration Design

## Goal

Move the DPA4-style edge-vector force compile path from benchmark-only probes into an opt-in MACE training acceleration gate, without changing default training behavior or altering MACE's conservative-force objective.

## Current Evidence

The accepted probe path is edge-vector based, not whole-model `torch.compile`. The existing RECIO gates show that MACE can compute the same energy, forces, force loss, and second-order parameter gradients when the force endpoint is moved from atom positions to detached edge vectors. The repaired make_fx graph preserves second-order gradients by removing saved-tensor detach chains, and dynamic Inductor works on CPU and CUDA no-cueq. On SAI V100 with cueq, the larger gate only passes when cueq optimized Linear is excluded while tensor-product and symmetric cueq acceleration remain enabled.

## Recommended Approach

Use a staged, fail-closed integration. The next implementation should introduce a small `EdgeVectorForceCompileCandidate` helper under `mace.tools.training_compile` or a focused sibling module. It should reuse `mace.tools.force_compile` for tracing, detach repair, graph rebuilding, and optional Inductor compile. It should own only training-time candidate construction and equivalence gating; it must not become a general replacement for `ScaleShiftMACE.forward`.

The first runnable integration point should be a pre-training gate invoked from tests and benchmark scripts: build a candidate for one batch, compare eager position-gradient force loss against edge-vector make_fx/compiled force loss, and return a structured decision. If the gate fails, the training model remains eager. This keeps production correctness ahead of speed claims.

## Rejected Approaches

Whole-model `torch.compile` remains rejected for conservative force training because it hits AOTAutograd double-backward limits and falls back to eager. Compiling arbitrary submodules such as readouts or radial embeddings is also rejected unless the force-loss equivalence gate passes, because earlier probes showed those boundaries can fail under second-order autograd. Direct-force heads are out of scope because they change the MACE objective.

## API Boundary

The helper should expose a small data structure such as `EdgeForceCompileConfig` with fields for `enabled`, `tracing_mode`, `strip_detach`, `compile_graph`, `compile_mode`, `compile_dynamic`, `allow_fallback`, `atol`, and `rtol`. A gate result should include `enabled`, `accepted`, `fallback_reason`, `detach_nodes_before`, `detach_nodes_after`, `node_count`, `comparison`, and `compile_kwargs`.

The candidate should support only `modules.ScaleShiftMACE` initially. It should explicitly reject virials, stress, displacement, hessian, edge forces, atomic stresses, and non-force training modes. This is conservative but aligned with the successful RECIO evidence. Support can widen only after separate equivalence gates.

## Cueq Compatibility

The integration must not hide the cueq Linear issue. Any SAI job that enables the compiled edge-vector force path with cueq must use the cueq-minus-linear profile until a new gate proves optimized Linear works. The code should document this constraint and the benchmark scripts should keep granular cueq flags visible.

## Testing

Tests should be TDD-first and small. Unit tests should verify config parsing, unsupported-output rejection, and that a gate result rejects when comparison fails. Probe-level tests should verify the helper is used by `probe_edge_vector_force_equivalence.py`. Focused regression must include `tests/test_force_compile.py`, `tests/test_edge_vector_force_equivalence.py`, `tests/test_force_backward_compile_ops.py`, and `tests/test_training_compile_probe.py`.

## Acceptance For This Slice

This slice is complete when the reusable gate object exists, the edge-vector probe uses it, and CPU focused tests pass. It does not claim end-to-end training speedup. The training-loop wrapper and SAI sbatch validation remain separate acceptance gates that require their own RECIO evidence.
