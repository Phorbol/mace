# MACE Training Stability Guards Design

## Scope

This design is an incremental phase of the DPA4-inspired MACE training
acceleration work. It adds opt-in stability guards around the existing MACE
training loop before wiring the compiled edge-vector force path into real
training.

The target is ordinary small-to-medium MACE training on the RECIO 8k benchmark:

- Dataset root: `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k`
- Environment: `mace_env`
- Primary training loop: `mace/tools/train.py`
- Existing acceleration evidence: `docs/acceleration/recio8k_training_acceleration.md`

This phase does not change the MACE model architecture, force definition, loss
semantics, HybridMuon routing, D3 implementation, or scheduler policy. Those are
separate phases. The guards introduced here must be compatible with later
force-compile, bf16 AMP, HybridMuon, and nvalchemi D3 work.

## Motivation

The current `take_step()` path computes loss, immediately calls
`loss.backward()`, applies `torch.nn.utils.clip_grad_norm_`, then always runs
`optimizer.step()`. This is simple, but it leaves three DPA4/TACE-identified
failure modes insufficiently guarded:

1. A non-finite loss can enter backward and optimizer state.
2. A large loss spike from a bad batch can permanently perturb Adam/Muon state.
3. A non-finite gradient norm can be clipped or checkpointed without a precise
   divergence report.

This matters more once DPA4-style force backward compile and bf16 AMP are added:
both increase the need for explicit numerical guardrails. The guards should
preserve current eager training behavior when disabled.

## Reference Ideas

Use the references as design inputs, not copied implementations:

- TACE `LossSkipController`: dynamic EMA threshold plus optional manual
  threshold; skip bad batches before optimizer state is updated.
- DeepMD-kit `NonFiniteGradGuard`: accumulate non-finite gradient-norm state on
  device and check before writing checkpoints.
- DeepMD-kit stable gradient clipping: max-magnitude normalization plus float64
  norm reduction to avoid overflow in reduced-precision training.

## User-Facing Behavior

Add opt-in controls with conservative defaults:

- `--loss_skip`: default `False`.
- `--loss_skip_nan`: default `True` when `--loss_skip` is enabled.
- `--loss_skip_large`: default `True` when `--loss_skip` is enabled.
- `--loss_skip_ema_window`: default `100`.
- `--loss_skip_multiplier`: default `3.0`.
- `--loss_skip_start_step`: default `1000`.
- `--loss_skip_threshold`: optional absolute threshold.
- `--stable_grad_clip`: default `False` for backward compatibility.
- `--nonfinite_grad_guard`: default `False`.

When all flags are disabled, generated logs, checkpoint timing, optimizer
updates, and training metrics should remain unchanged except for negligible
function-call overhead.

## Component Design

### TrainingGuardConfig

Create a small dataclass for guard options. It should be constructed in
`run_train.py` from CLI arguments and passed into `train()` and `train_one_epoch()`.
Keeping this as one object avoids expanding every training function with many
boolean and numeric parameters.

### LossSkipController

Create a MACE-owned controller in a focused module such as
`mace/tools/training_guards.py`.

Behavior:

- Check the scalar loss after forward/loss computation and before backward.
- Skip immediately for non-finite loss if `skip_nan=True`.
- After `start_step`, skip if loss exceeds the effective large-loss threshold.
- Effective threshold is the minimum of the optional absolute threshold and
  `multiplier * EMA(loss)`, ignoring unavailable thresholds.
- Update the EMA only on globally accepted finite losses.
- In distributed training, synchronize skip decisions with an integer
  `all_reduce(MAX)` so every rank either skips or trains the same step.

Skip handling:

- Call `optimizer.zero_grad(set_to_none=True)`.
- Do not call `loss.backward()`.
- Do not call `optimizer.step()`.
- Do not update EMA model weights.
- Return metrics containing `skipped=1`, `skip_reason`, and threshold/EMA
  values where available.

This intentionally differs from TACE's Lightning implementation because MACE
has a simpler naked PyTorch training loop. The MACE version should not depend on
Lightning or Hydra.

### Stable Gradient Clipping

Implement a local helper that returns the pre-clip total norm:

- If no gradients exist, return a CPU scalar zero.
- If `stable=False`, delegate to the current PyTorch norm behavior.
- If `stable=True`, use foreach max-norm scaling and float64 norm reduction.
- Apply `torch.nn.utils.clip_grads_with_norm_` with the computed norm.

The helper must not call `.item()` on every step. Returning a tensor keeps the
path compatible with GPU execution and future compile-adjacent code.

### NonFiniteGradGuard

Create a guard that records whether any total grad norm since the previous
checkpoint was non-finite.

Behavior:

- `update(total_norm)` accumulates the non-finite condition as a tensor.
- `raise_if_nonfinite(model.named_parameters)` reads the accumulated flag only
  before checkpoint writes.
- On failure, raise a clear `RuntimeError` listing current parameters whose
  gradient norm is non-finite. If current individual gradients are finite, say
  the non-finite norm was recorded earlier in the checkpoint interval.

Checkpoint integration should happen immediately before
`checkpoint_handler.save(...)` in `train()`, including both best-checkpoint and
`save_all_checkpoints` paths. This prevents writing a checkpoint after a
divergent interval.

## Training Loop Integration

Modify `take_step()` first; leave LBFGS unsupported in the first guard phase.
LBFGS uses closure semantics and whole-loader stepping, so applying skip logic
there needs separate design.

The new `take_step()` flow:

1. Move batch to device as today.
2. Zero gradients.
3. Forward under the existing precision context.
4. Compute scalar loss.
5. Run loss-skip check.
6. If skipped, return metrics and avoid backward, optimizer step, and EMA update.
7. Backward.
8. Clip gradients with either stable or existing behavior.
9. Update `NonFiniteGradGuard` if enabled.
10. Optimizer step.
11. EMA update.
12. Return metrics including loss, time, skipped flag, and grad norm when
    available.

The compile fallback retry around `disable_compile_fallback()` must stay intact.
If the first closure fails due to an accepted compile fallback, retry in eager
and only then evaluate skip/backward behavior.

## Metrics and Logging

Each training step should continue logging the existing `loss` and `time`
fields. New optional fields:

- `loss_skipped`: `0` or `1`
- `loss_skip_reason`: stable string such as `none`, `nonfinite`, or `large`
- `loss_skip_threshold`
- `loss_skip_ema`
- `grad_norm`
- `grad_norm_nonfinite`: `0` or `1`

These names are deliberately plain so the existing metrics parser can ignore
them until benchmark parsing is extended.

## Test Strategy

Use TDD for implementation.

Unit tests:

- `LossSkipController` accepts finite losses, skips NaN/Inf, skips large losses
  after warmup, updates EMA only on accepted losses, and reports stable reason
  strings.
- Stable clipping returns finite norms for large finite gradients and matches
  PyTorch clipping for ordinary gradients within tolerance.
- `NonFiniteGradGuard` raises after a non-finite norm and resets after checking.
- `take_step()` skips optimizer and EMA updates when loss-skip rejects a batch.
- `take_step()` still behaves as before when guard config is disabled.

Integration tests:

- Existing acceleration tests must keep passing.
- A CPU mini training-step smoke should verify guard metrics are present when
  enabled.
- RECIO/8k SAI smoke should run with guards enabled before force-compile
  training integration.

## Acceptance Criteria

This phase is accepted when:

- Stability guards are opt-in and disabled by default.
- Enabling loss-skip prevents optimizer and EMA updates on skipped batches.
- Stable clipping reports a tensor grad norm without per-step host sync.
- Non-finite grad guard blocks checkpoint saving after a recorded non-finite
  norm.
- Existing HybridMuon, bf16, D3, and force-compile probe tests still pass.
- A RECIO/8k SAI smoke with guards enabled completes without NaNs and logs guard
  metrics.

## Out of Scope

- WSD/warmup scheduler changes.
- Uncertainty loss.
- Full torchmetrics/DDP metrics rewrite.
- LMDB data pipeline.
- GPU D3 numerical validation.
- Force-compiled training integration.
- Any broadening of HybridMuon routing into equivariant tensor parameters.

These remain important, but they should not be coupled to the first stability
guard implementation.
