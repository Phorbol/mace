# Edge-Force Compile Cache and Bucketing Design

## Goal

Make the DPA4-style edge-force compiled training path useful for real MACE training, not only for fixed-batch profilers. The path must remain compatible with cueq-minus-linear and HybridMuon, preserve MACE's conservative energy-to-force training objective, and fail closed when it cannot amortize compile cost.

Current evidence shows the compiled edge-force path is correct for RECIO/8k real training steps: `edge_force_compile=true` and `edge_force_gate_accepted=true` on strict SAI job `581285`. It is not yet an end-to-end speedup because RECIO batches have changing atom and edge counts, causing `edge_force_cache_hit=false` and repeated 18-19 second compile steps.

## Non-Goals

This design does not change MACE's equivariant architecture, tensor-product algebra, or conservative force definition. It does not expand HybridMuon routing into equivariant tensor/path parameters yet. It does not implement NVIDIA D3 acceleration. Those remain separate workstreams after force-compile training is a stable speedup.

## Constraints

- Forces must remain `-dE/dR` through the model energy.
- The force loss must backpropagate to parameters through the force derivative.
- cueq optimized Linear is excluded from the compiled force graph until its zero-force lowering issue is fixed; cueq tensor products, channelwise, FCTP, symmetric contraction, and conv fusion can remain enabled.
- V100 does not prove native bf16 acceleration, so bf16 claims require a bf16-capable SAI partition later.
- A compile path that is correct but slower must disable itself or be reported as a fallback/negative-speedup case.

## Architecture

The implementation keeps the existing `EdgeForceCompiledLossModule` as the training-loop integration point and adds a cache policy layer above its current shape cache.

### 1. Compile-Aware Cache Policy

Add an `EdgeForceCachePolicy` with three modes:

- `shape`: current behavior. Cache key includes exact runtime tensor shapes.
- `repeat_only`: compile only after the same shape has appeared at least `N` times. This prevents one-off shapes from triggering expensive compilation.
- `bucket`: pad runtime batches to a small set of fixed atom/edge buckets, then compile/cache by bucket shape.

Default remains conservative: edge-force compile is opt-in, and the policy initially defaults to `repeat_only` when full graph compile is enabled. This avoids the current RECIO failure mode where every new shape triggers a compile.

### 2. Negative-Speedup Guard

Track per-shape or per-bucket counters:

- `seen_count`
- `compile_count`
- `cache_hit_count`
- `compile_setup_seconds`
- `compiled_step_seconds_ema`
- `eager_step_seconds_ema` when sampled

If a cache entry has no hit after setup, or if compiled steady-state is not faster after a configurable warmup, mark that shape/bucket as `compile_disabled`. Training then uses eager force loss for that shape instead of repeatedly compiling.

### 3. Bucketed Padding

Use `mace.data.padding_tools.build_fake_padding_graph` as the starting point, but do not wire it blindly into training. The bucketed path must pass an equivalence gate first:

- real structures are batched normally;
- a fake padding graph extends the batch to the bucket's `(num_atoms, num_edges)`;
- fake edges have cutoff-zero contributions;
- fake atoms/graphs do not contribute to supervised energy or force loss;
- output energy/forces/loss/selected parameter gradients match the unpadded eager baseline.

The first implementation should support single-head energy/force training only, matching the current edge-force compile restriction. Virials, stress, Hessian, edge forces, and atomic stresses remain unsupported and force eager fallback.

### 4. Training DataLoader Interaction

The existing `--shuffle` flag now controls training DataLoader shuffle. This enables deterministic repeated-batch cache experiments with `--shuffle False`, but it is not the final speed solution. For normal shuffled training, bucket mode should group similar graph sizes without changing labels or model semantics.

The first implementation can avoid a custom sampler by padding on demand inside `EdgeForceCompiledLossModule`. A later sampler can improve bucket locality, but correctness should not depend on DataLoader order.

## Data Flow

For a supported training batch:

1. `take_step` calls `compiled_force_training_loss`.
2. The wrapper computes a policy decision from current atom/edge counts.
3. If policy says eager, it returns the ordinary eager force loss with metrics.
4. If policy says compile, it optionally pads to a bucket shape.
5. The compiled closure takes edge vectors plus non-label batch tensors as explicit FX inputs.
6. The gate compares energy, forces, loss, and selected parameter gradients for new cache entries; for graph-compiled callables, the gate must not consume compiled outer backward before the optimizer step.
7. The optimizer backward runs once through the compiled loss.
8. Metrics record cache policy, bucket shape, cache hit, compile setup time, and fallback/disable reasons.

## CLI

Add opt-in flags:

- `--edge_force_compile_cache_policy {shape,repeat_only,bucket}`
- `--edge_force_compile_min_repeats N`
- `--edge_force_compile_disable_negative_speedup`
- `--edge_force_compile_bucket_atoms`
- `--edge_force_compile_bucket_edges`
- `--edge_force_compile_bucket_margin`

The exact bucket interface can start simple: comma-separated integer bucket sizes for atoms and edges. If no bucket fits, the path falls back eager for that batch and logs `edge_force_compile_disabled_reason=no_bucket`.

## Testing

Required local tests:

- cache policy does not compile first-seen shapes in `repeat_only`;
- repeated exact shape compiles once and then hits cache;
- negative-speedup guard disables a shape after configured miss/slow thresholds;
- bucket padding preserves energy, forces, scalar loss, and selected parameter gradients on a tiny CPU MACE batch;
- unsupported outputs still bypass edge-force compile.

Required SAI gates:

- RECIO tiny smoke with cueq-minus-linear and HybridMuon, strict no-fallback, confirms a compiled bucket cache hit;
- RECIO 8k short run comparing baseline eager, HybridMuon eager, and compiled bucket/repeat policy;
- benchmark parser must report cache hit rate, compile setup time, steady-state step time, validation force/energy metrics, and whether any fallback occurred.

## Acceptance

This work is accepted only when a real RECIO/8k multi-epoch run shows:

- no forced fallback for supported batches;
- cache hit rate high enough to amortize setup cost;
- wall-clock training speed improvement over ordinary MACE under comparable settings;
- validation energy/force trend comparable to the eager baseline;
- HybridMuon and cueq-minus-linear both active in the accelerated run;
- metrics make negative-speedup cases visible rather than silently counting them as acceleration.

