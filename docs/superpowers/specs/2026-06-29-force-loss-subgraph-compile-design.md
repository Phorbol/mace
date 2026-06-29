# Force-Loss Subgraph Compile Design

## Objective

Build a correctness gate for DPA4-inspired MACE training compile work before enabling any compiled subgraph in force-loss training. The gate must prove that a candidate compiled path preserves MACE's conservative-force objective: forces remain `-dE/dR`, and force-loss backpropagation produces the same parameter gradients as eager execution.

## Current State

Whole-model training compile is safe but falls back to eager for force-loss MACE training because PyTorch AOTAutograd cannot currently handle the required double-backward path on the tested stack. Energy-only training with `compute_force=False` keeps compile enabled. RECIO SAI jobs `576776` and `576782` verified this boundary with cueq enabled.

## Design

Add a focused equivalence layer around the existing RECIO compile probe utilities. The first gate compares eager execution to a candidate execution on a small RECIO-like `ScaleShiftMACE` batch. The comparison must include energy, forces, scalar loss, and selected parameter gradients after force-loss backward. This gate is independent of training CLI flags; it is a precondition for adding any future `--train_compile_subgraphs` behavior.

The initial candidate can be an identity/eager candidate, then a later compiled subgraph candidate. This keeps the public training path unchanged while establishing the invariant that every acceleration candidate must satisfy.

## Acceptance Criteria

- A new test module verifies energy, force, and parameter-gradient equivalence for the force-loss objective.
- The equivalence helper returns structured mismatch data so future probes can report failures without hiding them.
- Existing training compile fallback behavior remains unchanged.
- Focused tests pass in `mace_env`.

## Non-Goals

- Do not enable subgraph compile in production training yet.
- Do not change MACE model architecture or direct-force semantics.
- Do not claim force-loss compile acceleration until a non-fallback compiled candidate passes the equivalence gate and a RECIO sbatch smoke.
