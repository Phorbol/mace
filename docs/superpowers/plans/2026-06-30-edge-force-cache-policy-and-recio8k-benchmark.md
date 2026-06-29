# Edge Force Cache Policy and RECIO8k Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make MACE's DPA4-style edge-force backward compile path produce a measurable speedup on real RECIO/8k training while remaining compatible with cueq-minus-linear and HybridMuon.

**Architecture:** Keep the existing `EdgeForceCompiledLossModule` as the only training-loop integration point. Add an explicit cache policy layer that avoids compiling one-off shapes, records cache/negative-speed metrics, and only then add bucket padding behind equivalence tests. Treat TACE/DPA4 HybridMuon routing expansion as a later controlled optimizer experiment after compile speed is measurable.

**Tech Stack:** PyTorch, `torch.fx.experimental.proxy_tensor.make_fx`, Inductor, MACE `ScaleShiftMACE`, cuEquivariance cueq-minus-linear, `pytest`, Slurm/SAI sbatch with `mace_env`.

---

## Context and Constraints

The strict RECIO/8k job `581285` proved that real tracing with graph compile can run supported MACE force training batches with `edge_force_compile=true` and `edge_force_gate_accepted=true`, including HybridMuon and cueq-minus-linear. It also showed the current shape cache is not useful for shuffled RECIO training because every batch has a new `(num_atoms, num_edges)` key and `edge_force_cache_hit=false`.

Symbolic tracing is currently blocked by e3nn spherical harmonics custom/TorchScript code raising `Cannot access data pointer of Tensor` under FakeTensor. Do not make symbolic tracing the main path in this plan.

The current priority is compile-cache usefulness and real training benchmark evidence. DPA4/TACE HybridMuon improvements are important, but must not be mixed into this cache benchmark because changing optimizer routing at the same time would make speed and accuracy attribution unclear.

## File Responsibilities

- `mace/tools/training_compile.py`: cache policy dataclasses, policy decision logic, negative-speed guard state, integration into `EdgeForceCompiledLossModule`, metrics.
- `mace/tools/arg_parser.py`: new CLI flags for cache policy, repeat threshold, negative-speed guard, and bucket lists.
- `mace/cli/run_train.py`: forward new CLI fields into `EdgeForceCompileConfig`.
- `tests/test_compile.py`: unit tests for parser, policy behavior, integration decisions, cache hit behavior, and metrics.
- `scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh`: Slurm launcher for tiny and 8k RECIO benchmark runs.
- `scripts/benchmarks/recio8k_accel/parse_edge_force_cache_policy.py`: summarize wall time, cache hit rate, compile setup time, fallback count, and validation metrics from logs/results.
- `docs/acceleration/recio8k_training_acceleration.md`: append benchmark results and explicitly state whether speedup was achieved.

## Design Decisions

- Default cache policy stays conservative. If `--edge_force_compile` is enabled and no cache policy is provided, use `repeat_only` for graph compile so one-off RECIO shapes do not trigger repeated expensive compiles.
- `shape` preserves current behavior for fixed-batch and profiler experiments.
- `repeat_only` means "run eager until this exact shape has appeared `N` times, then compile it." It prevents negative speed on low-repeat shape distributions.
- `bucket` is not allowed to silently change labels or losses. It must first pass energy, force, scalar loss, and selected parameter-gradient equivalence on a tiny CPU batch before it can be used in training.
- Cache-policy eager fallback is normal operation and must not permanently disable compile. Runtime compile errors still respect `allow_fallback`; strict runs with `--no-edge_force_compile_allow_fallback` must fail loudly.
- HybridMuon in the current benchmark uses the existing MACE route. TACE-style `muon_mode=slice` and DPA4 optimizer internals become a separate follow-up plan after cache speed is demonstrated.

---

### Task 1: Add Cache Policy Types and Pure Decision Logic

**Files:**
- Modify: `mace/tools/training_compile.py`
- Test: `tests/test_compile.py`

- [ ] **Step 1: Write failing tests for repeat-only policy decisions**

Append these tests near the existing edge-force compile cache tests in `tests/test_compile.py`:

```python
def test_edge_force_cache_policy_repeat_only_skips_first_seen_shape():
    from mace.tools.training_compile import (
        EdgeForceCachePolicyState,
        edge_force_compile_shape_cache_key,
    )

    cache_key = edge_force_compile_shape_cache_key(
        num_atoms=4,
        num_edges=8,
        input_shapes={
            "positions": (4, 3),
            "edge_index": (2, 8),
            "node_attrs": (4, 2),
            "batch": (4,),
            "ptr": (2,),
        },
    )
    state = EdgeForceCachePolicyState()

    decision = state.record_and_decide(
        cache_key,
        policy="repeat_only",
        min_repeats=2,
    )

    assert decision.compile_allowed is False
    assert decision.reason == "min_repeats"
    assert decision.seen_count == 1
    assert decision.cache_policy == "repeat_only"


def test_edge_force_cache_policy_repeat_only_allows_repeated_shape():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    cache_key = ("shape", 4, 8)
    state = EdgeForceCachePolicyState()

    first = state.record_and_decide(cache_key, policy="repeat_only", min_repeats=2)
    second = state.record_and_decide(cache_key, policy="repeat_only", min_repeats=2)

    assert first.compile_allowed is False
    assert second.compile_allowed is True
    assert second.reason is None
    assert second.seen_count == 2
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_edge_force_cache_policy_repeat_only_skips_first_seen_shape tests/test_compile.py::test_edge_force_cache_policy_repeat_only_allows_repeated_shape -q
```

Expected: both tests fail with `ImportError` or missing `EdgeForceCachePolicyState`.

- [ ] **Step 3: Add dataclasses and decision state**

In `mace/tools/training_compile.py`, extend `EdgeForceCompileConfig` and add these classes below `EdgeForceCompileGateResult`:

```python
@dataclasses.dataclass(frozen=True)
class EdgeForceCachePolicyDecision:
    cache_policy: str
    compile_allowed: bool
    reason: str | None
    seen_count: int
    compile_count: int
    cache_hit_count: int
    disabled: bool = False


@dataclasses.dataclass
class EdgeForceCacheEntryStats:
    seen_count: int = 0
    compile_count: int = 0
    cache_hit_count: int = 0
    disabled_reason: str | None = None
    compile_setup_seconds: float = 0.0
    compiled_step_seconds_ema: float | None = None
    eager_step_seconds_ema: float | None = None


@dataclasses.dataclass
class EdgeForceCachePolicyState:
    entries: dict[tuple, EdgeForceCacheEntryStats] = dataclasses.field(
        default_factory=dict
    )

    def stats_for(self, cache_key: tuple) -> EdgeForceCacheEntryStats:
        return self.entries.setdefault(cache_key, EdgeForceCacheEntryStats())

    def record_and_decide(
        self,
        cache_key: tuple,
        *,
        policy: str,
        min_repeats: int,
        cache_hit: bool = False,
    ) -> EdgeForceCachePolicyDecision:
        stats = self.stats_for(cache_key)
        stats.seen_count += 1
        if cache_hit:
            stats.cache_hit_count += 1
        if stats.disabled_reason is not None:
            return EdgeForceCachePolicyDecision(
                cache_policy=policy,
                compile_allowed=False,
                reason=stats.disabled_reason,
                seen_count=stats.seen_count,
                compile_count=stats.compile_count,
                cache_hit_count=stats.cache_hit_count,
                disabled=True,
            )
        if policy == "shape":
            allowed = True
            reason = None
        elif policy == "repeat_only":
            allowed = stats.seen_count >= max(1, int(min_repeats))
            reason = None if allowed else "min_repeats"
        elif policy == "bucket":
            allowed = True
            reason = None
        else:
            raise ValueError(f"unknown edge-force cache policy: {policy}")
        return EdgeForceCachePolicyDecision(
            cache_policy=policy,
            compile_allowed=allowed,
            reason=reason,
            seen_count=stats.seen_count,
            compile_count=stats.compile_count,
            cache_hit_count=stats.cache_hit_count,
            disabled=False,
        )

    def record_compile(self, cache_key: tuple, *, setup_seconds: float) -> None:
        stats = self.stats_for(cache_key)
        stats.compile_count += 1
        stats.compile_setup_seconds += float(setup_seconds)

    def disable(self, cache_key: tuple, reason: str) -> None:
        self.stats_for(cache_key).disabled_reason = reason
```

Add fields to `EdgeForceCompileConfig`:

```python
    cache_policy: str = "repeat_only"
    min_repeats: int = 2
    disable_negative_speedup: bool = True
    negative_speedup_min_steps: int = 4
```

- [ ] **Step 4: Run the policy tests**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_edge_force_cache_policy_repeat_only_skips_first_seen_shape tests/test_compile.py::test_edge_force_cache_policy_repeat_only_allows_repeated_shape -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: add edge-force cache policy state"
```

---

### Task 2: Integrate Repeat-Only Policy Into Training Loss

**Files:**
- Modify: `mace/tools/training_compile.py`
- Test: `tests/test_compile.py`

- [ ] **Step 1: Write failing integration test for no compile before repeat threshold**

Append this test near `test_edge_force_compiled_loss_gates_shape_cache_hits`:

```python
def test_edge_force_compiled_loss_repeat_only_uses_eager_before_threshold(monkeypatch):
    from mace.tools import training_compile

    class DummyBatch:
        positions = torch.zeros(4, 3)
        edge_index = torch.zeros(2, 8, dtype=torch.long)

        def to_dict(self):
            return {
                "positions": self.positions,
                "edge_index": self.edge_index,
                "node_attrs": torch.zeros(4, 2),
                "batch": torch.zeros(4, dtype=torch.long),
                "ptr": torch.tensor([0, 4], dtype=torch.long),
            }

    class DummyModel(torch.nn.Module):
        def forward(self, data, **kwargs):
            return {
                "energy": torch.zeros(1, 1, requires_grad=True),
                "forces": torch.zeros(4, 3, requires_grad=True),
            }

    def loss_fn(*, pred, ref):
        return pred["energy"].sum() + pred["forces"].sum()

    wrapper = training_compile.EdgeForceCompiledLossModule(
        DummyModel(),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            cache_policy="repeat_only",
            min_repeats=2,
            allow_fallback=False,
        ),
    )

    def fail_compile(**kwargs):
        raise AssertionError("_compile_step should not run before min_repeats")

    monkeypatch.setattr(wrapper, "_compile_step", fail_compile)

    loss, metrics = wrapper.compiled_force_training_loss(
        batch=DummyBatch(),
        loss_fn=loss_fn,
        output_args={},
    )

    assert loss.item() == 0.0
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "min_repeats"
    assert metrics["edge_force_compile_cache_policy"] == "repeat_only"
    assert metrics["edge_force_cache_seen_count"] == 1
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_edge_force_compiled_loss_repeat_only_uses_eager_before_threshold -q
```

Expected: fail because `EdgeForceCompiledLossModule` compiles immediately.

- [ ] **Step 3: Implement policy-aware eager metrics**

Change `_eager_force_loss` signature in `mace/tools/training_compile.py` to:

```python
    def _eager_force_loss(
        self,
        *,
        batch,
        loss_fn,
        disabled_reason: str = "disabled",
        policy_decision: EdgeForceCachePolicyDecision | None = None,
    ):
        output = self.model(
            batch.to_dict(),
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        metrics = {
            "edge_force_compile": False,
            "edge_force_compile_disabled": True,
            "edge_force_compile_disabled_reason": disabled_reason,
        }
        if policy_decision is not None:
            metrics.update(
                {
                    "edge_force_compile_cache_policy": policy_decision.cache_policy,
                    "edge_force_cache_seen_count": policy_decision.seen_count,
                    "edge_force_cache_compile_count": policy_decision.compile_count,
                    "edge_force_cache_hit_count": policy_decision.cache_hit_count,
                }
            )
        return loss_fn(pred=output, ref=batch), metrics
```

In `__init__`, add:

```python
        self.cache_policy_state = EdgeForceCachePolicyState()
```

In `compiled_force_training_loss`, after computing `compiled` and `cache_hit`, add:

```python
            policy_decision = self.cache_policy_state.record_and_decide(
                cache_key,
                policy=self.config.cache_policy,
                min_repeats=self.config.min_repeats,
                cache_hit=cache_hit,
            )
            if not cache_hit and not policy_decision.compile_allowed:
                return self._eager_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    disabled_reason=policy_decision.reason or "policy_disabled",
                    policy_decision=policy_decision,
                )
```

- [ ] **Step 4: Record compile counts and policy metrics**

At the top of `mace/tools/training_compile.py`, add:

```python
import time
```

Around the `_compile_step` call in `compiled_force_training_loss`, replace:

```python
                compiled = self._compile_step(
                    batch=batch,
                    loss_fn=loss_fn,
                    cache_key=cache_key,
                )
```

with:

```python
                setup_start = time.perf_counter()
                compiled = self._compile_step(
                    batch=batch,
                    loss_fn=loss_fn,
                    cache_key=cache_key,
                )
                setup_seconds = time.perf_counter() - setup_start
                self.cache_policy_state.record_compile(
                    cache_key,
                    setup_seconds=setup_seconds,
                )
```

Extend compiled metrics:

```python
                "edge_force_compile_cache_policy": policy_decision.cache_policy,
                "edge_force_cache_seen_count": policy_decision.seen_count,
                "edge_force_cache_compile_count": self.cache_policy_state.stats_for(
                    cache_key
                ).compile_count,
                "edge_force_cache_hit_count": self.cache_policy_state.stats_for(
                    cache_key
                ).cache_hit_count,
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_edge_force_compiled_loss_repeat_only_uses_eager_before_threshold tests/test_compile.py::test_edge_force_compiled_loss_gates_shape_cache_hits -q
```

Expected: both pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: gate edge-force compile by cache policy"
```

---

### Task 3: Add CLI Flags and Run-Train Forwarding

**Files:**
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Test: `tests/test_compile.py`

- [ ] **Step 1: Update parser test first**

Extend `test_arg_parser_accepts_edge_force_compile_flags`:

```python
            "--edge_force_compile_cache_policy",
            "repeat_only",
            "--edge_force_compile_min_repeats",
            "3",
            "--no-edge_force_compile_disable_negative_speedup",
```

Add assertions:

```python
    assert args.edge_force_compile_cache_policy == "repeat_only"
    assert args.edge_force_compile_min_repeats == 3
    assert args.edge_force_compile_disable_negative_speedup is False
```

- [ ] **Step 2: Run parser test and verify it fails**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_arg_parser_accepts_edge_force_compile_flags -q
```

Expected: fail because flags do not exist.

- [ ] **Step 3: Add parser arguments**

In `mace/tools/arg_parser.py`, after `--edge_force_compile_cache_hit_gate`, add:

```python
    parser.add_argument(
        "--edge_force_compile_cache_policy",
        help="Cache policy for edge-force compile: shape compiles immediately, repeat_only waits for repeated shapes, bucket uses padded bucket shapes",
        type=str,
        choices=["shape", "repeat_only", "bucket"],
        default="repeat_only",
    )
    parser.add_argument(
        "--edge_force_compile_min_repeats",
        help="Minimum times an exact shape must be seen before repeat_only compiles it",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--edge_force_compile_disable_negative_speedup",
        help="Disable compile for a shape or bucket when policy metrics show negative speedup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
```

- [ ] **Step 4: Forward flags into config**

In `mace/cli/run_train.py`, extend `EdgeForceCompileConfig(...)`:

```python
                cache_policy=args.edge_force_compile_cache_policy,
                min_repeats=args.edge_force_compile_min_repeats,
                disable_negative_speedup=args.edge_force_compile_disable_negative_speedup,
```

- [ ] **Step 5: Run parser test**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_arg_parser_accepts_edge_force_compile_flags -q
```

Expected: pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add mace/tools/arg_parser.py mace/cli/run_train.py tests/test_compile.py
git commit -m "feat: expose edge-force cache policy flags"
```

---

### Task 4: Add Minimal Negative-Speedup Guard Metrics

**Files:**
- Modify: `mace/tools/training_compile.py`
- Test: `tests/test_compile.py`

- [ ] **Step 1: Write failing guard-state test**

Append:

```python
def test_edge_force_cache_policy_can_disable_negative_speedup_shape():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    state = EdgeForceCachePolicyState()
    cache_key = ("shape", 4, 8)
    state.disable(cache_key, "negative_speedup")

    decision = state.record_and_decide(
        cache_key,
        policy="shape",
        min_repeats=1,
    )

    assert decision.compile_allowed is False
    assert decision.disabled is True
    assert decision.reason == "negative_speedup"
```

- [ ] **Step 2: Run guard test**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_edge_force_cache_policy_can_disable_negative_speedup_shape -q
```

Expected: pass if Task 1 state implementation already includes `disable`; if it fails, implement the exact `disable` method from Task 1.

- [ ] **Step 3: Add timing helper**

Add to `EdgeForceCachePolicyState`:

```python
    def record_step_time(
        self,
        cache_key: tuple,
        *,
        compiled: bool,
        seconds: float,
        ema_decay: float = 0.9,
    ) -> None:
        stats = self.stats_for(cache_key)
        value = float(seconds)
        if compiled:
            old = stats.compiled_step_seconds_ema
            stats.compiled_step_seconds_ema = (
                value if old is None else ema_decay * old + (1.0 - ema_decay) * value
            )
        else:
            old = stats.eager_step_seconds_ema
            stats.eager_step_seconds_ema = (
                value if old is None else ema_decay * old + (1.0 - ema_decay) * value
            )
```

Wrap eager policy fallback in `compiled_force_training_loss` with `time.perf_counter()` and record eager step time. Wrap compiled executable/loss construction with `time.perf_counter()` and record compiled step time. Do not include optimizer backward in this helper because this wrapper only owns loss construction.

- [ ] **Step 4: Add metrics for step-time EMAs**

Add these metrics when returning eager policy fallback or compiled loss:

```python
            "edge_force_compile_setup_seconds": stats.compile_setup_seconds,
            "edge_force_compiled_step_seconds_ema": stats.compiled_step_seconds_ema,
            "edge_force_eager_step_seconds_ema": stats.eager_step_seconds_ema,
```

Use local `stats = self.cache_policy_state.stats_for(cache_key)`.

- [ ] **Step 5: Keep disabling conservative**

Do not automatically disable on the first slow compiled step. Only mark `negative_speedup` when both EMAs are present, `stats.cache_hit_count >= self.config.negative_speedup_min_steps`, and `compiled_ema >= eager_ema`. This keeps the guard from disabling before it has meaningful cache-hit data:

```python
            if (
                self.config.disable_negative_speedup
                and stats.cache_hit_count >= self.config.negative_speedup_min_steps
                and stats.compiled_step_seconds_ema is not None
                and stats.eager_step_seconds_ema is not None
                and stats.compiled_step_seconds_ema >= stats.eager_step_seconds_ema
            ):
                self.cache_policy_state.disable(cache_key, "negative_speedup")
```

- [ ] **Step 6: Run compile tests**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py -q
```

Expected: all compile tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add mace/tools/training_compile.py tests/test_compile.py
git commit -m "feat: report edge-force compile cache timing"
```

---

### Task 5: Add Bucket CLI Parsing Without Enabling Training Padding

**Files:**
- Modify: `mace/tools/training_compile.py`
- Modify: `mace/tools/arg_parser.py`
- Modify: `mace/cli/run_train.py`
- Test: `tests/test_compile.py`

- [ ] **Step 1: Write parser test for buckets**

Extend `test_arg_parser_accepts_edge_force_compile_flags` with:

```python
            "--edge_force_compile_bucket_atoms",
            "256,512",
            "--edge_force_compile_bucket_edges",
            "2048,4096",
            "--edge_force_compile_bucket_margin",
            "1.15",
```

Add assertions:

```python
    assert args.edge_force_compile_bucket_atoms == "256,512"
    assert args.edge_force_compile_bucket_edges == "2048,4096"
    assert args.edge_force_compile_bucket_margin == 1.15
```

- [ ] **Step 2: Add bucket fields**

Extend `EdgeForceCompileConfig`:

```python
    bucket_atoms: tuple[int, ...] = ()
    bucket_edges: tuple[int, ...] = ()
    bucket_margin: float = 1.0
```

Add helper:

```python
def parse_edge_force_bucket_sizes(value: str | None) -> tuple[int, ...]:
    if value is None or value == "":
        return ()
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(size <= 0 for size in sizes):
        raise ValueError("edge-force bucket sizes must be positive integers")
    return tuple(sorted(set(sizes)))
```

- [ ] **Step 3: Add parser flags**

In `mace/tools/arg_parser.py`, add:

```python
    parser.add_argument(
        "--edge_force_compile_bucket_atoms",
        help="Comma-separated atom-count buckets for edge-force compile bucket mode",
        type=str,
        default="",
    )
    parser.add_argument(
        "--edge_force_compile_bucket_edges",
        help="Comma-separated edge-count buckets for edge-force compile bucket mode",
        type=str,
        default="",
    )
    parser.add_argument(
        "--edge_force_compile_bucket_margin",
        help="Maximum bucket/input size ratio allowed for bucket mode",
        type=float,
        default=1.0,
    )
```

- [ ] **Step 4: Forward bucket fields**

Import `parse_edge_force_bucket_sizes` in `mace/cli/run_train.py` alongside `EdgeForceCompileConfig`, then pass:

```python
                bucket_atoms=parse_edge_force_bucket_sizes(
                    args.edge_force_compile_bucket_atoms
                ),
                bucket_edges=parse_edge_force_bucket_sizes(
                    args.edge_force_compile_bucket_edges
                ),
                bucket_margin=args.edge_force_compile_bucket_margin,
```

- [ ] **Step 5: Add no-bucket policy behavior**

In `compiled_force_training_loss`, if `self.config.cache_policy == "bucket"` and either bucket tuple is empty, return eager with reason `no_bucket`:

```python
            if (
                self.config.cache_policy == "bucket"
                and (not self.config.bucket_atoms or not self.config.bucket_edges)
            ):
                return self._eager_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    disabled_reason="no_bucket",
                )
```

This task intentionally does not pad training batches. It only creates the safe CLI/config surface and an explicit fallback reason.

- [ ] **Step 6: Run tests**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py::test_arg_parser_accepts_edge_force_compile_flags -q
pytest tests/test_compile.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add mace/tools/training_compile.py mace/tools/arg_parser.py mace/cli/run_train.py tests/test_compile.py
git commit -m "feat: add edge-force bucket policy interface"
```

---

### Task 6: Add RECIO Cache-Policy Slurm Smoke Launcher

**Files:**
- Create: `scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh`

- [ ] **Step 1: Create Slurm script**

Create:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mace-edge-cache
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=runs/recio8k_edge_cache/%x-%j.out
#SBATCH --error=runs/recio8k_edge_cache/%x-%j.err

set -euo pipefail

source /opt/envs/anaconda3.env
conda activate mace_env

cd /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace

RUN_ROOT="${RUN_ROOT:-runs/recio8k_edge_cache}"
TRAIN_FILE="${TRAIN_FILE:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}"
mkdir -p "${RUN_ROOT}"

python -m mace.cli.run_train \
  --name="${NAME:-recio8k_edge_cache_repeat_smoke}" \
  --train_file="${TRAIN_FILE}" \
  --valid_fraction=0.05 \
  --test_file="${TRAIN_FILE}" \
  --E0s=average \
  --model=ScaleShiftMACE \
  --num_interactions=2 \
  --num_channels="${NUM_CHANNELS:-64}" \
  --max_L="${MAX_L:-1}" \
  --correlation=3 \
  --r_max=5.0 \
  --batch_size="${BATCH_SIZE:-32}" \
  --valid_batch_size="${VALID_BATCH_SIZE:-32}" \
  --max_num_epochs="${MAX_NUM_EPOCHS:-2}" \
  --patience=999 \
  --eval_interval=1 \
  --error_table=PerAtomMAE \
  --default_dtype=float32 \
  --device=cuda \
  --seed=123 \
  --shuffle="${SHUFFLE:-False}" \
  --enable_cueq=True \
  --cueq_config=cueq_minus_linear \
  --optimizer=hybrid_muon \
  --edge_force_compile \
  --edge_force_compile_tracing_mode=real \
  --edge_force_compile_cache_policy="${EDGE_FORCE_CACHE_POLICY:-repeat_only}" \
  --edge_force_compile_min_repeats="${EDGE_FORCE_MIN_REPEATS:-2}" \
  --edge_force_compile_mode=default \
  --edge_force_compile_dynamic=True \
  --edge_force_compile_graph=True \
  --no-edge_force_compile_allow_fallback \
  --work_dir="${RUN_ROOT}" \
  --log_dir="${RUN_ROOT}/logs" \
  --model_dir="${RUN_ROOT}/models" \
  --checkpoints_dir="${RUN_ROOT}/checkpoints" \
  --results_dir="${RUN_ROOT}/results"
```

- [ ] **Step 2: Make it executable**

Run:

```bash
chmod +x scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

- [ ] **Step 3: Commit**

Run:

```bash
git add scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
git commit -m "chore: add RECIO edge-force cache smoke launcher"
```

---

### Task 7: Add Benchmark Log Parser

**Files:**
- Create: `scripts/benchmarks/recio8k_accel/parse_edge_force_cache_policy.py`

- [ ] **Step 1: Create parser script**

Create:

```python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_bool_count(text: str, key: str, value: str) -> int:
    return len(re.findall(rf"{re.escape(key)}[=: ]+{value}", text))


def summarize_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    compile_true = _parse_bool_count(text, "edge_force_compile", "True") + _parse_bool_count(
        text, "edge_force_compile", "true"
    )
    cache_hit_true = _parse_bool_count(text, "edge_force_cache_hit", "True") + _parse_bool_count(
        text, "edge_force_cache_hit", "true"
    )
    disabled_reasons = sorted(
        set(re.findall(r"edge_force_compile_disabled_reason[=: ]+([A-Za-z0-9_\\-]+)", text))
    )
    fallback_mentions = len(re.findall(r"fallback|failed; disabling compiled force loss", text, re.I))
    return {
        "path": str(path),
        "edge_force_compile_true": compile_true,
        "edge_force_cache_hit_true": cache_hit_true,
        "fallback_mentions": fallback_mentions,
        "disabled_reasons": disabled_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize_log(path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run parser against existing run logs**

Run:

```bash
python scripts/benchmarks/recio8k_accel/parse_edge_force_cache_policy.py runs/recio8k_accel_value_gate_strict/*.out
```

Expected: prints JSON with at least `path`, `edge_force_compile_true`, `edge_force_cache_hit_true`, and `fallback_mentions`.

- [ ] **Step 3: Commit**

Run:

```bash
git add scripts/benchmarks/recio8k_accel/parse_edge_force_cache_policy.py
git commit -m "chore: add edge-force cache benchmark parser"
```

---

### Task 8: Run Local Verification

**Files:**
- No source changes unless tests expose regressions.

- [ ] **Step 1: Run targeted compile tests**

Run:

```bash
source /opt/envs/anaconda3.env
conda activate mace_env
pytest tests/test_compile.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Check git status**

Run:

```bash
git status --short
```

Expected: only expected untracked run outputs or review artifacts remain. No unintended tracked modifications.

---

### Task 9: Run SAI Smoke and RECIO/8k Benchmark

**Files:**
- Modify: `docs/acceleration/recio8k_training_acceleration.md`

- [ ] **Step 1: Submit strict repeat-only smoke**

Run:

```bash
mkdir -p runs/recio8k_edge_cache
sbatch scripts/benchmarks/recio8k_accel/run_edge_force_cache_policy_sai.sh
```

Expected: Slurm returns a job id. The run should either compile after a repeated shape or use eager with `edge_force_compile_disabled_reason=min_repeats`; it must not silently fallback from a compile error because `--no-edge_force_compile_allow_fallback` is set.

- [ ] **Step 2: Monitor the job**

Run:

```bash
squeue -u "$USER"
```

After completion, inspect:

```bash
python scripts/benchmarks/recio8k_accel/parse_edge_force_cache_policy.py runs/recio8k_edge_cache/*.out
```

Expected: parser output shows whether cache hits occurred and whether fallbacks appeared.

- [ ] **Step 3: Run baseline eager HybridMuon + cueq**

Submit an eager control using the same script but with edge-force compile disabled by passing the normal `run_train` command from the script without these flags:

```bash
  --edge_force_compile
  --edge_force_compile_tracing_mode=real
  --edge_force_compile_cache_policy=repeat_only
  --edge_force_compile_min_repeats=2
  --edge_force_compile_mode=default
  --edge_force_compile_dynamic=True
  --edge_force_compile_graph=True
  --no-edge_force_compile_allow_fallback
```

Use the same `NUM_CHANNELS=64`, `MAX_L=1`, `correlation=3`, `batch_size=32`, `max_num_epochs=2`, `shuffle=False`, `cueq_minus_linear`, and `optimizer=hybrid_muon`.

- [ ] **Step 4: Decide whether bucket implementation is required immediately**

If repeat-only has low cache hits on the real RECIO/8k split, record that result and do not claim speedup. Move to the separate bucket-padding implementation plan. If repeat-only produces repeated shapes under `--shuffle False`, compare steady-state step times and confirm no fallback.

- [ ] **Step 5: Document evidence**

Append a section to `docs/acceleration/recio8k_training_acceleration.md`:

```markdown
## Edge-Force Cache Policy Benchmark

- Dataset: `/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz`
- Model: `ScaleShiftMACE`, `num_interactions=2`, `num_channels=64`, `max_L=1`, `correlation=3`
- Accelerator: cueq-minus-linear
- Optimizer: HybridMuon
- Policy: repeat_only, `min_repeats=2`
- Baseline job:
- Compile job:
- Cache hit rate:
- Fallback count:
- Wall time per epoch:
- Validation energy/force metric:
- Conclusion:
```

Fill in the job IDs and measured values from the parser and logs.

- [ ] **Step 6: Commit benchmark docs**

Run:

```bash
git add docs/acceleration/recio8k_training_acceleration.md
git commit -m "docs: record edge-force cache policy benchmark"
```

---

### Task 10: Follow-Up Plan Boundary for TACE/DPA4 HybridMuon

**Files:**
- Create: `docs/superpowers/specs/2026-06-30-tace-dpa4-hybrid-muon-routing-design.md`

- [ ] **Step 1: Create follow-up design note**

Create a short design note with these exact requirements:

```markdown
# TACE/DPA4 HybridMuon Routing for MACE

## Goal

Evaluate whether MACE should adopt DPA4/TACE HybridMuon internals and TACE-style `muon_mode=slice` routing after edge-force compile speed is measurable.

## Constraints

- Do not change optimizer routing in the edge-force compile benchmark.
- Keep the DPA4/TACE optimizer body separate from the MACE parameter-routing decision.
- Muon may only act on legal multiplicity/channel matrices.
- Do not flatten across irreps, species, parity, tensor-product path, correlation order, or physical branch semantics.
- Every expanded route must have an ablation against current MACE HybridMuon and AdamW.

## Candidate Experiments

1. Replace MACE Newton-Schulz body with DPA4/TACE two-stage NS and optional Magma-lite damping while keeping current MACE routing.
2. Add `muon_mode={2d,slice,flat}` but default to current conservative MACE behavior.
3. Test per-slice symmetric-contraction routing only as an opt-in experiment, with RECIO validation and equivariance checks.
4. Log per-block update RMS, gradient norm, momentum-gradient cosine, and update-to-weight norm.

## Acceptance

The expanded route is accepted only if real RECIO/8k multi-epoch training improves speed or accuracy without worse validation force/energy trend, without NaN, and without violating equivariance/structure diagnostics.
```

- [ ] **Step 2: Commit follow-up boundary**

Run:

```bash
git add docs/superpowers/specs/2026-06-30-tace-dpa4-hybrid-muon-routing-design.md
git commit -m "docs: scope TACE DPA4 HybridMuon follow-up"
```

---

## Self-Review Checklist

- Spec coverage: cache policy, repeat threshold, negative-speed metrics, bucket CLI surface, RECIO/8k benchmark, cueq-minus-linear, HybridMuon compatibility, and TACE/DPA4 Muon follow-up are covered.
- No symbolic tracing dependency: plan keeps real tracing as the main path because symbolic currently fails in e3nn spherical harmonics.
- Attribution: optimizer routing is not changed before compile-cache benchmark evidence exists.
- Physics constraint: bucket padding is not enabled until equivalence tests prove fake atoms/edges do not affect energy, forces, scalar loss, or selected parameter gradients.
- Benchmark constraint: acceptance requires real RECIO/8k evidence, not a toy task.
