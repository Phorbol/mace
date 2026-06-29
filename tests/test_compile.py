import os
from functools import wraps
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3
from torch.testing import assert_close

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric

table = tools.AtomicNumberTable([6])
atomic_energies = np.array([1.0], dtype=float)
cutoff = 5.0


def setup_cueq(enable: bool, device: str):
    if not enable:
        return None

    return CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=(device == "cuda"),
    )


def create_tiny_mace(
    device: str, seed: int = 1702, scale: float = 1.0, shift: float = 0.0
):
    torch_geometric.seed_everything(seed)
    model_config = {
        "r_max": cutoff,
        "num_bessel": 4,
        "num_polynomial_cutoff": 4,
        "max_ell": 1,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 1,
        "num_elements": 1,
        "hidden_irreps": o3.Irreps("8x0e + 8x1o"),
        "MLP_irreps": o3.Irreps("8x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 1,
        "radial_type": "bessel",
        "atomic_inter_scale": scale,
        "atomic_inter_shift": shift,
        "cueq_config": None,
    }
    model = modules.ScaleShiftMACE(**model_config)
    return model.to(device)


def create_mace(device: str, seed: int = 1702, enable_cueq: bool = False):
    torch_geometric.seed_everything(seed)

    model_config = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 1,
        "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": setup_cueq(enable_cueq, device),
    }
    model = modules.ScaleShiftMACE(**model_config)
    return model.to(device)


class _BatchDictAdapter:
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def to(self, device, **kwargs):
        self.data_dict = {
            key: value.to(device, **kwargs) if hasattr(value, "to") else value
            for key, value in self.data_dict.items()
        }
        return self

    def to_dict(self):
        return dict(self.data_dict)

    def __getattr__(self, name):
        try:
            return self.data_dict[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _EnergyForcesMiniLoss(torch.nn.Module):
    def forward(self, pred, ref):
        del ref
        return pred["energy"].sum() + pred["forces"].square().mean()


def create_batch(device: str):
    from ase import build

    size = 2
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = batch.to_dict()
    return batch


def time_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch._inductor.cudagraph_mark_step_begin()  # pylint: disable=W0212
        outputs = func(*args, **kwargs)
        torch.cuda.synchronize()
        return outputs

    return wrapper


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def default_dtype(request):
    with tools.torch_tools.default_dtype(request.param):
        yield torch.get_default_dtype()


# skip if on windows
@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace(device, default_dtype):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_defaults = create_mace(device)
    tmp_model = mace_compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default", fullgraph=True)

    batch1 = create_batch(device)
    output1 = model_defaults(batch1, training=True)

    batch2 = create_batch(device)
    batch2["positions"].requires_grad_(True)
    output2 = model_compiled(batch2, training=True)
    assert_close(output1["energy"], output2["energy"])
    assert_close(output1["forces"], output2["forces"])


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace_compile_stress(device, default_dtype):  # pylint: disable=W0621
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_eager = create_mace(device)
    batch_eager = create_batch(device)
    output_eager = model_eager(batch_eager, compute_stress=True)

    tmp_model = mace_compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default", fullgraph=True)
    batch_compiled = create_batch(device)
    batch_compiled["positions"].requires_grad_(True)
    output_compiled = model_compiled(batch_compiled, training=True, compute_stress=True)

    assert_close(output_eager["energy"], output_compiled["energy"])
    assert_close(output_eager["forces"], output_compiled["forces"])
    assert output_eager["stress"] is not None
    assert output_compiled["stress"] is not None
    assert_close(output_eager["stress"], output_compiled["stress"])


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("enable_cueq", [True, False])
def test_eager_benchmark(
    benchmark, default_dtype, enable_cueq
):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    batch = create_batch("cuda")
    model = create_mace("cuda", enable_cueq=enable_cueq)
    model = time_func(model)
    benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("enable_amp", [False, True], ids=["fp32", "mixed"])
@pytest.mark.parametrize("enable_cueq", [False, True])
def test_compile_benchmark(benchmark, compile_mode, enable_amp, enable_cueq):
    _torch_version = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    if (
        enable_cueq
        and compile_mode in ("reduce-overhead", "max-autotune")
        and _torch_version < (2, 10)
    ):
        pytest.skip("cueq + CUDA graphs requires PyTorch >= 2.10")
    with tools.torch_tools.default_dtype(torch.float32):
        torch.compiler.reset()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        batch = create_batch("cuda")
        batch["positions"].requires_grad_(True)
        model = mace_compile.prepare(create_mace)("cuda", enable_cueq=enable_cueq)
        model = torch.compile(model, mode=compile_mode)
        model = time_func(model)

        with torch.autocast("cuda", enabled=enable_amp):
            benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_graph_breaks(device):
    import torch._dynamo as dynamo

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    batch = create_batch(device)
    batch["positions"].requires_grad_(True)
    model = mace_compile.prepare(create_mace)(device)
    explanation = dynamo.explain(model)(batch, training=False)

    # these clutter the output but might be useful for investigating graph breaks
    explanation.ops_per_graph = None
    explanation.out_guards = None
    print(explanation)
    assert explanation.graph_break_count == 0


def test_training_compile_helper_noop_cpu():
    from mace.tools.training_compile import prepare_model_for_training_compile

    model = create_mace("cpu")
    compiled = prepare_model_for_training_compile(
        model,
        enabled=False,
        mode="default",
        fullgraph=False,
        allow_fallback=True,
    )

    assert compiled is model


def test_compile_fx_graph_module_disables_donated_buffer_during_compile(monkeypatch):
    import torch._functorch.config as functorch_config

    from mace.tools.force_compile import compile_fx_graph_module

    graph_module = torch.fx.symbolic_trace(torch.nn.Identity())
    previous = functorch_config.donated_buffer
    compile_seen_donated_buffer = []

    def fake_compile(module, **kwargs):
        compile_seen_donated_buffer.append(functorch_config.donated_buffer)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    functorch_config.donated_buffer = True
    try:
        executable, compile_kwargs = compile_fx_graph_module(
            graph_module,
            compile_graph=True,
            compile_mode="default",
            compile_dynamic=True,
        )
    finally:
        functorch_config.donated_buffer = previous

    assert executable is graph_module
    assert compile_kwargs["backend"] == "inductor"
    assert compile_kwargs["dynamic"] is True
    assert isinstance(compile_kwargs["options"], dict)
    if "triton.cudagraphs" in compile_kwargs["options"]:
        assert compile_kwargs["options"]["triton.cudagraphs"] is False
    if "triton.persistent_reductions" in compile_kwargs["options"]:
        assert compile_kwargs["options"]["triton.persistent_reductions"] is False
    assert compile_seen_donated_buffer == [False]
    assert functorch_config.donated_buffer is True


def test_edge_force_compile_config_defaults_to_disabled():
    from mace.tools.training_compile import EdgeForceCompileConfig

    config = EdgeForceCompileConfig()

    assert config.enabled is False
    assert config.tracing_mode == "real"
    assert config.strip_detach is True
    assert config.compile_graph is True
    assert config.compile_mode == "default"
    assert config.compile_dynamic is True
    assert config.allow_fallback is True
    assert config.atol == 1.0e-5
    assert config.rtol == 1.0e-4


def test_edge_force_compile_gate_rejects_disabled_config():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=False),
    )

    assert result.enabled is False
    assert result.accepted is False
    assert result.fallback_reason == "disabled"
    assert result.detach_nodes_before is None
    assert result.detach_nodes_after is None
    assert result.node_count is None


def test_edge_force_compile_gate_rejects_unsupported_outputs():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=True),
        compute_virials=True,
    )

    assert result.enabled is True
    assert result.accepted is False
    assert result.fallback_reason == "unsupported_outputs"


def test_edge_force_compile_result_from_trace_records_gate_metadata():
    from mace.tools.force_compile import trace_force_closure
    from mace.tools.training_compile import edge_force_compile_result_from_trace

    def fn(x):
        y = x + x.detach().detach()
        return y

    x = torch.tensor([1.0], requires_grad=True)
    trace_result = trace_force_closure(
        fn,
        (x,),
        tracing_mode="real",
        strip_detach=True,
    )
    comparison = {"ok": True, "failed_checks": []}

    result = edge_force_compile_result_from_trace(
        trace_result=trace_result,
        comparison=comparison,
        compile_kwargs={"backend": "inductor", "dynamic": True},
    )

    assert result.enabled is True
    assert result.accepted is True
    assert result.fallback_reason is None
    assert result.detach_nodes_before >= 2
    assert result.detach_nodes_after == 0
    assert result.node_count == len(list(trace_result.graph_module.graph.nodes))
    assert result.comparison == comparison
    assert result.compile_kwargs == {"backend": "inductor", "dynamic": True}


def test_edge_force_compile_input_names_exclude_labels_and_geometry():
    from mace.tools.training_compile import edge_force_compile_input_names

    names = edge_force_compile_input_names(
        {
            "positions",
            "edge_index",
            "node_attrs",
            "batch",
            "ptr",
            "head",
            "shifts",
            "energy",
            "forces",
            "stress",
            "virials",
        }
    )

    assert names == ("positions", "edge_index", "node_attrs", "batch", "ptr", "head")


def test_edge_force_compile_shape_cache_key_ignores_batch_identity():
    from mace.tools.training_compile import edge_force_compile_shape_cache_key

    left = edge_force_compile_shape_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={"positions": (286, 3), "node_attrs": (286, 4)},
    )
    right = edge_force_compile_shape_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={"positions": (286, 3), "node_attrs": (286, 4)},
    )

    assert left == right
    assert left[0] == "shape"


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


def test_edge_force_cache_policy_records_step_time_emas():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    cache_key = ("shape", 4, 8)
    state = EdgeForceCachePolicyState()

    state.record_step_time(cache_key, compiled=False, seconds=10.0, ema_decay=0.5)
    state.record_step_time(cache_key, compiled=False, seconds=6.0, ema_decay=0.5)
    state.record_step_time(cache_key, compiled=True, seconds=4.0, ema_decay=0.5)
    state.record_step_time(cache_key, compiled=True, seconds=2.0, ema_decay=0.5)

    stats = state.stats_for(cache_key)
    assert stats.eager_step_seconds_ema == 8.0
    assert stats.compiled_step_seconds_ema == 3.0


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


def test_parse_edge_force_bucket_sizes_sorts_and_validates_values():
    from mace.tools.training_compile import parse_edge_force_bucket_sizes

    assert parse_edge_force_bucket_sizes("") == ()
    assert parse_edge_force_bucket_sizes(None) == ()
    assert parse_edge_force_bucket_sizes("512,256,512") == (256, 512)
    with pytest.raises(ValueError, match="positive integers"):
        parse_edge_force_bucket_sizes("128,0")


def test_edge_force_compile_cache_hit_gate_result_records_failure():
    from mace.tools.training_compile import edge_force_cache_hit_gate_result

    comparison = {"ok": False, "failed_checks": ["forces"]}
    result = edge_force_cache_hit_gate_result(
        comparison=comparison,
        cache_key=("shape", 2, 4),
    )

    assert result.enabled is True
    assert result.accepted is False
    assert result.fallback_reason == "cache_hit_equivalence_failed"
    assert result.comparison == comparison
    assert result.cache_hit is True
    assert result.cache_key == ["shape", 2, 4]


def test_prepare_edge_force_compiled_loss_disabled_returns_model():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = _TrainingOnlyModel()

    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(enabled=False),
    )

    assert prepared is model



def test_edge_force_compiled_loss_disables_functorch_donated_buffer_for_graph_compile():
    import torch._functorch.config as functorch_config

    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    previous = functorch_config.donated_buffer
    try:
        functorch_config.donated_buffer = True
        eager_only = prepare_edge_force_compiled_loss(
            create_tiny_mace("cpu"),
            config=EdgeForceCompileConfig(enabled=True, compile_graph=False),
        )
        assert eager_only.functorch_donated_buffer_disabled is False
        assert functorch_config.donated_buffer is True

        graph_compiled = prepare_edge_force_compiled_loss(
            create_tiny_mace("cpu"),
            config=EdgeForceCompileConfig(enabled=True, compile_graph=True),
        )
        assert graph_compiled.functorch_donated_buffer_disabled is True
        assert functorch_config.donated_buffer is False
    finally:
        functorch_config.donated_buffer = previous


def test_edge_force_compiled_loss_bucket_policy_without_buckets_uses_eager(monkeypatch):
    from mace.tools import training_compile

    model = create_tiny_mace("cpu")
    prepared = training_compile.prepare_edge_force_compiled_loss(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_policy="bucket",
            bucket_atoms=(),
            bucket_edges=(),
            allow_fallback=False,
        ),
    )

    def fail_compile(**kwargs):
        raise AssertionError("_compile_step should not run without buckets")

    monkeypatch.setattr(prepared, "_compile_step", fail_compile)

    loss, metrics = prepared.compiled_force_training_loss(
        batch=_BatchDictAdapter(create_batch("cpu")),
        loss_fn=_EnergyForcesMiniLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert torch.isfinite(loss)
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "no_bucket"


def test_edge_force_compiled_loss_repeat_only_uses_eager_before_threshold(monkeypatch):
    from mace.tools import training_compile

    model = create_tiny_mace("cpu")
    prepared = training_compile.prepare_edge_force_compiled_loss(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_policy="repeat_only",
            min_repeats=2,
            allow_fallback=False,
        ),
    )

    def fail_compile(**kwargs):
        raise AssertionError("_compile_step should not run before min_repeats")

    monkeypatch.setattr(prepared, "_compile_step", fail_compile)

    loss, metrics = prepared.compiled_force_training_loss(
        batch=_BatchDictAdapter(create_batch("cpu")),
        loss_fn=_EnergyForcesMiniLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert torch.isfinite(loss)
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "min_repeats"
    assert metrics["edge_force_compile_cache_policy"] == "repeat_only"
    assert metrics["edge_force_cache_seen_count"] == 1


def test_edge_force_compiled_loss_gates_shape_cache_hits():
    from mace.tools.train import take_step
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = create_tiny_mace("cpu")
    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_hit_gate=True,
            cache_policy="shape",
        ),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)

    take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=_BatchDictAdapter(create_batch("cpu")),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )
    _, metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=_BatchDictAdapter(create_batch("cpu")),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert metrics["edge_force_cache_hit"] is True
    assert metrics["edge_force_cache_hit_gate_accepted"] is True


def test_edge_force_compiled_loss_handles_nonidentity_scaleshift():
    from mace.tools.train import take_step
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = create_tiny_mace("cpu", scale=2.0, shift=0.25)
    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_policy="shape",
        ),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)

    _, metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=_BatchDictAdapter(create_batch("cpu")),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_gate_accepted"] is True


def test_prepare_edge_force_compiled_loss_wraps_scaleshiftmace_for_take_step():
    from mace.tools.train import take_step
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = create_tiny_mace("cpu")
    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_policy="shape",
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)

    loss, metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert prepared is not model
    assert hasattr(prepared, "compiled_force_training_loss")
    assert loss.requires_grad is True
    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_cache_hit"] is False
    assert metrics["edge_force_gate_accepted"] is True


def test_edge_force_compile_step_uses_fresh_executable_after_gate(monkeypatch):
    import types

    from mace.tools import training_compile

    compiled_executables = []
    gate_executables = []

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        executable = object()
        compiled_executables.append(executable)
        return executable, {"compile_graph": kwargs["compile_graph"]}

    def snapshot_payload():
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {},
        }

    def fake_snapshot(*, executable, **kwargs):
        gate_executables.append(executable)
        return snapshot_payload()

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        lambda **kwargs: snapshot_payload(),
    )
    monkeypatch.setattr(
        training_compile, "_edge_force_value_snapshot_from_executable", fake_snapshot
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected grad gate")),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_snapshot_from_executable",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected grad gate")),
    )

    wrapper = training_compile.EdgeForceCompiledLossModule(
        create_mace("cpu"),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
        ),
    )

    class BatchWrapper:
        def __init__(self, data):
            self._data = data
            self.positions = data["positions"]
            self.edge_index = data["edge_index"]

        def to_dict(self):
            return self._data

    compiled = wrapper._compile_step(
        batch=BatchWrapper(create_batch("cpu")),
        loss_fn=torch.nn.MSELoss(),
        cache_key=("shape",),
    )

    assert len(compiled_executables) == 2
    assert gate_executables == [compiled_executables[0]]
    assert compiled.executable is compiled_executables[1]


def test_prepare_edge_force_compiled_loss_falls_back_for_unsupported_model():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = _TrainingOnlyModel()

    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(enabled=True, allow_fallback=True),
    )

    assert prepared is model


def test_arg_parser_accepts_edge_force_compile_flags():
    from mace.tools import build_default_arg_parser

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "edge-force-test",
            "--edge_force_compile",
            "--edge_force_compile_mode",
            "reduce-overhead",
            "--edge_force_compile_tracing_mode",
            "symbolic",
            "--no-edge_force_compile_graph",
            "--no-edge_force_compile_dynamic",
            "--no-edge_force_compile_cache_hit_gate",
            "--edge_force_compile_cache_policy",
            "repeat_only",
            "--edge_force_compile_min_repeats",
            "3",
            "--edge_force_compile_bucket_atoms",
            "256,512",
            "--edge_force_compile_bucket_edges",
            "2048,4096",
            "--edge_force_compile_bucket_margin",
            "1.15",
            "--no-edge_force_compile_disable_negative_speedup",
            "--no-edge_force_compile_allow_fallback",
        ]
    )

    assert args.edge_force_compile is True
    assert args.edge_force_compile_mode == "reduce-overhead"
    assert args.edge_force_compile_tracing_mode == "symbolic"
    assert args.edge_force_compile_graph is False
    assert args.edge_force_compile_dynamic is False
    assert args.edge_force_compile_cache_hit_gate is False
    assert args.edge_force_compile_cache_policy == "repeat_only"
    assert args.edge_force_compile_min_repeats == 3
    assert args.edge_force_compile_bucket_atoms == "256,512"
    assert args.edge_force_compile_bucket_edges == "2048,4096"
    assert args.edge_force_compile_bucket_margin == 1.15
    assert args.edge_force_compile_disable_negative_speedup is False
    assert args.edge_force_compile_allow_fallback is False


def test_arg_parser_accepts_training_shuffle_flag():
    from mace.tools import build_default_arg_parser

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "shuffle-test",
            "--shuffle",
            "False",
        ]
    )

    assert args.shuffle is False


def test_arg_parser_accepts_compile_compatible_cueq_flags():
    from mace.tools import build_default_arg_parser

    args = build_default_arg_parser().parse_args(
        [
            "--name=test",
            "--enable_cueq=True",
            "--cueq_layout=mul_ir",
            "--no-cueq-optimize-all",
            "--no-cueq-optimize-linear",
            "--cueq-optimize-channelwise",
            "--cueq-optimize-symmetric",
            "--cueq-optimize-fctp",
            "--no-cueq-conv-fusion",
        ]
    )

    assert args.enable_cueq is True
    assert args.cueq_layout == "mul_ir"
    assert args.cueq_optimize_all is False
    assert args.cueq_optimize_linear is False
    assert args.cueq_optimize_channelwise is True
    assert args.cueq_optimize_symmetric is True
    assert args.cueq_optimize_fctp is True
    assert args.cueq_conv_fusion is False


def test_e3nn_to_cueq_run_uses_granular_config(monkeypatch):
    import torch

    from mace.cli import convert_e3nn_cueq

    class HiddenIrreps:
        def slices(self):
            return [slice(0, 1), slice(1, 2)]

    class DummyModel(torch.nn.Module):
        def __init__(self, **config):
            super().__init__()
            self.config = config
            self.weight = torch.nn.Parameter(torch.ones(()))

    def fake_extract_config(_model):
        return {
            "hidden_irreps": HiddenIrreps(),
            "correlation": 3,
            "num_interactions": 1,
            "use_reduced_cg": True,
            "keep_last_layer_irreps": False,
        }

    monkeypatch.setattr(
        convert_e3nn_cueq, "extract_config_mace_model", fake_extract_config
    )
    monkeypatch.setattr(
        convert_e3nn_cueq, "transfer_weights", lambda *args, **kwargs: None
    )

    target = convert_e3nn_cueq.run(
        DummyModel(),
        device="cpu",
        layout="mul_ir",
        optimize_all=False,
        optimize_linear=False,
        optimize_channelwise=True,
        optimize_symmetric=True,
        optimize_fctp=True,
        conv_fusion=False,
    )

    cueq_config = target.config["cueq_config"]
    assert cueq_config.layout_str == "mul_ir"
    assert cueq_config.optimize_all is False
    assert cueq_config.optimize_linear is False
    assert cueq_config.optimize_channelwise is True
    assert cueq_config.optimize_symmetric is True
    assert cueq_config.optimize_fctp is True
    assert cueq_config.conv_fusion is False


def test_training_compile_allow_fallback_suppresses_dynamo_errors():
    import torch._dynamo.config as dynamo_config

    from mace.tools.training_compile import prepare_model_for_training_compile

    previous = dynamo_config.suppress_errors
    dynamo_config.suppress_errors = False
    try:
        model = torch.nn.Linear(1, 1)
        compiled = prepare_model_for_training_compile(
            model,
            enabled=True,
            mode="default",
            fullgraph=False,
            allow_fallback=True,
        )

        assert compiled is not model
        assert dynamo_config.suppress_errors is True
    finally:
        dynamo_config.suppress_errors = previous


class _MiniBatch:
    def __init__(self):
        self.x = torch.ones(1)

    def to(self, device, **kwargs):
        self.x = self.x.to(device, **kwargs)
        return self

    def to_dict(self):
        return {"x": self.x}


class _NoCallModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("base model should not be used for compiled training step")


class _TrainingOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, batch, **kwargs):
        self.calls += 1
        return {"value": batch["x"].sum() * self.weight}


class _MiniLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return pred["value"]


class _CompiledForceLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.forward_calls = 0
        self.compiled_loss_calls = 0

    def forward(self, batch, **kwargs):
        self.forward_calls += 1
        return {"value": batch["x"].sum() * self.weight}

    def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
        self.compiled_loss_calls += 1
        assert output_args == {"forces": True, "virials": False, "stress": False}
        loss = loss_fn(pred={"value": batch.x.sum() * self.weight}, ref=batch)
        return loss, {"edge_force_compile": True, "edge_force_cache_hit": False}


class _MiniLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics):
        self.records.append(metrics)


class _MiniEma:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


def test_train_one_epoch_uses_optional_training_model():
    from mace.tools.train import train_one_epoch

    base_model = _NoCallModel()
    training_model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(training_model.parameters(), lr=0.1)
    logger = _MiniLogger()

    train_one_epoch(
        model=base_model,
        training_model=training_model,
        loss_fn=_MiniLoss(),
        data_loader=[_MiniBatch()],
        optimizer=optimizer,
        epoch=0,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=logger,
        device=torch.device("cpu"),
        distributed=False,
    )

    assert training_model.calls == 1
    assert logger.records[0]["mode"] == "opt"


class _RaisesOnForward(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("compiled path failed")


def test_runtime_compile_fallback_retries_eager_model():
    from mace.tools.training_compile import RuntimeFallbackCompiledModule

    eager = torch.nn.Linear(1, 1)
    wrapper = RuntimeFallbackCompiledModule(
        eager_model=eager,
        compiled_model=_RaisesOnForward(),
        allow_fallback=True,
    )
    x = torch.ones(1, 1)

    output = wrapper(x)

    assert torch.allclose(output, eager(x))
    assert wrapper.disabled is True


def test_energy_only_compile_wrapper_preserves_force_outputs_cpu():
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    model = create_mace("cpu")
    wrapper = EnergyOnlyForceCompiledModule(
        eager_model=model,
        compiled_model=model,
        allow_fallback=False,
    )

    eager_output = model(
        create_batch("cpu"),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    wrapped_output = wrapper(
        create_batch("cpu"),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )

    assert_close(wrapped_output["energy"], eager_output["energy"])
    assert_close(wrapped_output["forces"], eager_output["forces"])


class _EnergyOnlyCompiledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, batch, **kwargs):
        self.calls += 1
        return {"energy": batch["x"].sum() * self.weight}


def test_energy_only_compile_wrapper_uses_compiled_model_without_forces():
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    eager = _NoCallModel()
    compiled = _EnergyOnlyCompiledModel()
    wrapper = EnergyOnlyForceCompiledModule(
        eager_model=eager,
        compiled_model=compiled,
        allow_fallback=False,
    )

    output = wrapper(
        {"x": torch.ones(2)},
        training=True,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
    )
    output["energy"].backward()

    assert compiled.calls == 1
    assert compiled.weight.grad is not None


class _BackwardFails(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("compiled backward failed")


class _BackwardFallbackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.disabled = False
        self.disable_calls = 0
        self.forward_calls = 0

    def forward(self, batch, **kwargs):
        self.forward_calls += 1
        value = batch["x"].sum() * self.weight
        if not self.disabled:
            value = _BackwardFails.apply(value)
        return {"value": value}

    def disable_compile_fallback(self, exc):
        self.disable_calls += 1
        self.disabled = True
        return True


def test_take_step_uses_compiled_force_training_loss_hook():
    from mace.tools.train import take_step

    model = _CompiledForceLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = _MiniEma()

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.forward_calls == 0
    assert model.compiled_loss_calls == 1
    assert model.weight.item() == pytest.approx(0.9)
    assert ema.updates == 1
    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_cache_hit"] is False


def test_take_step_honors_compiled_force_retain_graph_marker(monkeypatch):
    from mace.tools.train import take_step

    class RetainGraphCompiledForceLossModel(_CompiledForceLossModel):
        def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
            loss, metrics = super().compiled_force_training_loss(
                batch=batch, loss_fn=loss_fn, output_args=output_args
            )
            metrics["_retain_graph_for_backward"] = True
            return loss, metrics

    backward_retain_graph_values = []
    original_backward = torch.Tensor.backward

    def recording_backward(self, *args, **kwargs):
        backward_retain_graph_values.append(kwargs.get("retain_graph", False))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", recording_backward)

    model = RetainGraphCompiledForceLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert backward_retain_graph_values == [True]
    assert "_retain_graph_for_backward" not in metrics
    assert metrics["edge_force_compile"] is True


def test_take_step_retries_eager_after_compile_backward_failure():
    from mace.tools.train import take_step

    model = _BackwardFallbackModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert model.disabled is True
    assert model.disable_calls == 1
    assert model.forward_calls == 2
    assert metrics["loss"] == 1.0
    assert loss.requires_grad is True


class _TransferRecordingBatch:
    def __init__(self):
        self.to_calls = []

    def to(self, device, **kwargs):
        self.to_calls.append((device, kwargs))
        return self

    def to_dict(self):
        return {"value": torch.tensor([1.0])}


class _TransferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch_dict, **kwargs):
        return {"value": self.weight * batch_dict["value"]}


def test_take_step_uses_non_blocking_batch_transfer_when_requested():
    from mace.tools.train import take_step

    batch = _TransferRecordingBatch()
    model = _TransferModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        non_blocking_transfer=True,
    )

    assert batch.to_calls == [(torch.device("cpu"), {"non_blocking": True})]


def test_take_step_loss_skip_prevents_optimizer_and_ema_update():
    from mace.tools.train import take_step
    from mace.tools.training_guards import LossSkipController, TrainingGuardConfig

    model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = _MiniEma()
    controller = LossSkipController(
        manual_threshold=0.5,
        start_step=0,
        ema_window=2,
        multiplier=2.0,
        skip_nan=True,
        skip_large=True,
    )

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        guard_config=TrainingGuardConfig(loss_skip=True, loss_skip_threshold=0.5),
        loss_skip_controller=controller,
        global_step=0,
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(1.0)
    assert model.weight.grad is None
    assert ema.updates == 0
    assert metrics["loss_skipped"] is True
    assert metrics["loss_skip_reason"] == "large"


def test_take_step_default_guards_preserve_optimizer_and_ema_update():
    from mace.tools.train import take_step

    model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = _MiniEma()

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(0.9)
    assert model.weight.grad is not None
    assert ema.updates == 1
    assert metrics["loss_skipped"] is False


class _RecordingCheckpointHandler:
    def __init__(self):
        self.calls = []

    def save(self, **kwargs):
        self.calls.append(kwargs)


def test_save_checkpoint_after_guard_blocks_nonfinite_gradients():
    from mace.tools.train import _save_checkpoint_after_guard
    from mace.tools.training_guards import NonFiniteGradGuard

    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([float("inf")])
    guard = NonFiniteGradGuard()
    guard.update(torch.tensor(float("inf")))
    handler = _RecordingCheckpointHandler()

    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        _save_checkpoint_after_guard(
            checkpoint_handler=handler,
            state=object(),
            epochs=3,
            keep_last=True,
            grad_guard=guard,
            named_parameters=lambda: [("weight", param)],
        )

    assert handler.calls == []


def test_save_checkpoint_after_guard_saves_when_guard_is_clear():
    from mace.tools.train import _save_checkpoint_after_guard

    handler = _RecordingCheckpointHandler()
    state = object()

    _save_checkpoint_after_guard(
        checkpoint_handler=handler,
        state=state,
        epochs=4,
        keep_last=False,
        grad_guard=None,
        named_parameters=lambda: [],
    )

    assert handler.calls == [
        {"state": state, "epochs": 4, "keep_last": False}
    ]


def test_train_one_epoch_passes_non_blocking_transfer_to_take_step(monkeypatch):
    import importlib

    train_module = importlib.import_module("mace.tools.train")

    seen = {}

    def fake_take_step(**kwargs):
        seen["non_blocking_transfer"] = kwargs["non_blocking_transfer"]
        return torch.tensor(0.0), {"loss": 0.0}

    class FakeLogger:
        def log(self, metrics):
            seen["logged"] = metrics

    monkeypatch.setattr(train_module, "take_step", fake_take_step)

    train_module.train_one_epoch(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        data_loader=[object()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        epoch=3,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=FakeLogger(),
        device=torch.device("cpu"),
        distributed=False,
        non_blocking_transfer=True,
    )

    assert seen["non_blocking_transfer"] is True
    assert seen["logged"]["epoch"] == 3
