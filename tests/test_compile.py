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


def test_disable_e3nn_codegen_restores_setting_after_exception(monkeypatch):
    state = {"jit_script_fx": True}

    def fake_get_optimization_defaults():
        return dict(state)

    def fake_set_optimization_defaults(**kwargs):
        state.update(kwargs)

    monkeypatch.setattr(
        mace_compile, "get_optimization_defaults", fake_get_optimization_defaults
    )
    monkeypatch.setattr(
        mace_compile, "set_optimization_defaults", fake_set_optimization_defaults
    )

    try:
        with mace_compile.disable_e3nn_codegen():
            assert state["jit_script_fx"] is False
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected RuntimeError")

    assert state["jit_script_fx"] is True

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


def test_disabled_cueq_config_does_not_enable_conv_fusion():
    disabled_cueq = CuEquivarianceConfig(
        enabled=False,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_channelwise=True,
        conv_fusion=True,
    )

    model = create_tiny_mace("cpu", cueq_config=disabled_cueq)

    assert disabled_cueq.enabled is False
    assert all(not hasattr(interaction, "conv_fusion") for interaction in model.interactions)


def create_tiny_mace(
    device: str,
    seed: int = 1702,
    scale: float = 1.0,
    shift: float = 0.0,
    cueq_config=None,
    correlation: int = 1,
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
        "correlation": correlation,
        "radial_type": "bessel",
        "atomic_inter_scale": scale,
        "atomic_inter_shift": shift,
        "cueq_config": cueq_config,
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

    def __getitem__(self, name):
        return self.data_dict[name]


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


def test_build_force_compile_inductor_options_can_disable_shape_padding():
    from mace.tools.force_compile import build_force_compile_inductor_options

    options = build_force_compile_inductor_options(shape_padding=False)

    if "shape_padding" in options:
        assert options["shape_padding"] is False


def test_build_force_compile_inductor_options_can_set_max_fusion_size():
    from mace.tools.force_compile import build_force_compile_inductor_options

    options = build_force_compile_inductor_options(max_fusion_size=1)

    if "max_fusion_size" in options:
        assert options["max_fusion_size"] == 1


def test_edge_force_compile_config_defaults_to_disabled():
    from mace.tools.training_compile import EdgeForceCompileConfig

    config = EdgeForceCompileConfig()

    assert config.enabled is False
    assert config.tracing_mode == "real"
    assert config.strip_detach is False
    assert config.compile_graph is True
    assert config.compile_mode == "default"
    assert config.compile_dynamic is True
    assert config.compile_shape_padding is True
    assert config.compile_max_fusion_size == 8
    assert config.use_e3nn_spherical_harmonics is False
    assert config.allow_fallback is True
    assert config.atol == 1.0e-5
    assert config.rtol == 1.0e-4
    assert config.cache_hit_gate is False
    assert config.refresh_executable_each_step is False


def test_edge_force_symbolic_config_reuses_cached_executable_by_default():
    from mace.tools.training_compile import EdgeForceCompileConfig

    config = EdgeForceCompileConfig(tracing_mode="symbolic")

    assert config.refresh_executable_each_step is False


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


def test_aten_spherical_harmonics_matches_e3nn_component_lmax3():
    from mace.tools.training_compile import _edge_force_spherical_harmonics

    reference = o3.SphericalHarmonics(
        o3.Irreps.spherical_harmonics(3), normalize=True, normalization="component"
    )
    vectors = torch.randn(17, 3)

    actual = _edge_force_spherical_harmonics(reference, vectors)
    expected = reference(vectors)

    assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


def test_aten_spherical_harmonics_symbolic_trace_lmax3():
    from mace.tools.force_compile import trace_force_closure
    from mace.tools.training_compile import _edge_force_spherical_harmonics

    reference = o3.SphericalHarmonics(
        o3.Irreps.spherical_harmonics(3), normalize=True, normalization="component"
    )
    vectors = torch.randn(17, 3)

    result = trace_force_closure(
        lambda x: _edge_force_spherical_harmonics(reference, x),
        (vectors,),
        tracing_mode="symbolic",
        strip_detach=False,
    )

    assert len(list(result.graph_module.graph.nodes)) > 0
    assert_close(
        result.graph_module(vectors), reference(vectors), atol=1.0e-6, rtol=1.0e-6
    )


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


def test_edge_force_compile_loss_input_names_are_loss_specific():
    from mace.modules import (
        WeightedEnergyForcesL1L2Loss,
        WeightedEnergyForcesLoss,
        WeightedForcesLoss,
    )
    from mace.tools.training_compile import edge_force_compile_loss_input_names

    data_keys = {
        "energy",
        "forces",
        "weight",
        "energy_weight",
        "forces_weight",
    }

    assert edge_force_compile_loss_input_names(
        data_keys, loss_fn=WeightedEnergyForcesLoss()
    ) == ("energy", "forces", "weight", "energy_weight", "forces_weight")
    assert edge_force_compile_loss_input_names(
        data_keys, loss_fn=WeightedForcesLoss()
    ) == ("forces", "weight", "forces_weight")
    assert edge_force_compile_loss_input_names(
        data_keys, loss_fn=WeightedEnergyForcesL1L2Loss()
    ) == ("energy", "forces", "weight", "energy_weight")


def test_edge_force_compile_loss_input_names_reject_missing_loss_specific_keys():
    from mace.modules import WeightedForcesLoss
    from mace.tools.training_compile import edge_force_compile_loss_input_names

    assert (
        edge_force_compile_loss_input_names(
            {"energy", "weight", "forces_weight"},
            loss_fn=WeightedForcesLoss(),
        )
        == ()
    )


def test_position_force_compile_input_names_include_shifts():
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
        },
        force_gradient_mode="positions",
    )

    assert names == (
        "positions",
        "edge_index",
        "shifts",
        "node_attrs",
        "batch",
        "ptr",
        "head",
    )


def test_select_by_node_heads_matches_advanced_indexing():
    from mace.tools.training_compile import _select_by_node_heads

    values = torch.tensor(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]]
    )
    node_heads = torch.tensor([2, 0, 1])
    expected = values[torch.arange(values.shape[0]), node_heads]

    assert torch.equal(_select_by_node_heads(values, node_heads), expected)


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


def test_edge_force_compile_shape_cache_key_includes_abi_signature():
    from mace.tools.training_compile import edge_force_compile_shape_cache_key

    common = {
        "num_atoms": 4,
        "num_edges": 8,
        "input_shapes": {
            "positions": (4, 3),
            "edge_index": (2, 8),
        },
    }

    default = edge_force_compile_shape_cache_key(
        **common,
        abi_signature=("abi", ("compile_mode", "default")),
    )
    reduced = edge_force_compile_shape_cache_key(
        **common,
        abi_signature=("abi", ("compile_mode", "reduce-overhead")),
    )

    assert default != reduced
    assert default[-2] == ("abi", ("compile_mode", "default"))
    assert reduced[-2] == ("abi", ("compile_mode", "reduce-overhead"))


def test_edge_force_compiled_loss_cache_key_separates_compile_options():
    from mace.tools import training_compile

    batch = _BatchDictAdapter(create_batch("cpu"))
    data_dict = batch.to_dict()
    input_names = training_compile.edge_force_compile_input_names(data_dict.keys())

    default_wrapper = training_compile.EdgeForceCompiledLossModule(
        create_tiny_mace("cpu"),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            compile_mode="default",
            compile_dynamic=True,
        ),
    )
    reduced_wrapper = training_compile.EdgeForceCompiledLossModule(
        create_tiny_mace("cpu"),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            compile_mode="reduce-overhead",
            compile_dynamic=True,
        ),
    )

    default_key = default_wrapper._cache_key(
        batch=batch,
        input_names=input_names,
        data_dict=data_dict,
    )
    reduced_key = reduced_wrapper._cache_key(
        batch=batch,
        input_names=input_names,
        data_dict=data_dict,
    )

    assert default_key != reduced_key
    assert default_key[-2][0] == "abi"
    assert reduced_key[-2][0] == "abi"


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


def test_edge_force_cache_policy_break_even_waits_for_repeats():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    state = EdgeForceCachePolicyState()
    cache_key = ("shape", 4, 8)

    decision = state.record_and_decide(
        cache_key,
        policy="break_even",
        min_repeats=2,
        break_even_expected_remaining_hits=10,
    )

    assert decision.compile_allowed is False
    assert decision.reason == "min_repeats"
    assert decision.cache_policy == "break_even"


def test_edge_force_cache_policy_break_even_uses_setup_and_step_emas():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    state = EdgeForceCachePolicyState()
    cache_key = ("shape", 4, 8)
    stats = state.stats_for(cache_key)
    stats.seen_count = 2
    stats.compile_setup_seconds = 9.0
    stats.eager_step_seconds_ema = 5.0
    stats.compiled_step_seconds_ema = 2.0

    rejected = state.record_and_decide(
        cache_key,
        policy="break_even",
        min_repeats=2,
        break_even_expected_remaining_hits=2,
    )
    accepted = state.record_and_decide(
        cache_key,
        policy="break_even",
        min_repeats=2,
        break_even_expected_remaining_hits=3,
    )

    assert rejected.compile_allowed is False
    assert rejected.reason == "break_even"
    assert rejected.break_even_hits == pytest.approx(3.0)
    assert rejected.expected_remaining_hits == 2
    assert accepted.compile_allowed is True
    assert accepted.reason is None
    assert accepted.break_even_hits == pytest.approx(3.0)
    assert accepted.expected_remaining_hits == 3


def test_edge_force_cache_policy_break_even_rejects_non_positive_speedup():
    from mace.tools.training_compile import EdgeForceCachePolicyState

    state = EdgeForceCachePolicyState()
    cache_key = ("shape", 4, 8)
    stats = state.stats_for(cache_key)
    stats.seen_count = 2
    stats.compile_setup_seconds = 1.0
    stats.eager_step_seconds_ema = 2.0
    stats.compiled_step_seconds_ema = 2.5

    decision = state.record_and_decide(
        cache_key,
        policy="break_even",
        min_repeats=2,
        break_even_expected_remaining_hits=100,
    )

    assert decision.compile_allowed is False
    assert decision.reason == "break_even_no_speedup"


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


def test_pad_edge_force_data_to_bucket_adds_masks_and_fixed_shapes():
    from mace.tools.training_compile import _pad_edge_force_data_to_bucket

    batch = create_batch("cpu")
    num_atoms = batch["positions"].shape[0]
    num_edges = batch["edge_index"].shape[1]

    padded = _pad_edge_force_data_to_bucket(
        batch,
        atom_bucket=num_atoms + 3,
        edge_bucket=num_edges + 5,
        r_max=5.0,
    )

    assert padded["positions"].shape == (num_atoms + 3, 3)
    assert padded["node_attrs"].shape == (num_atoms + 3, batch["node_attrs"].shape[1])
    assert padded["batch"].shape == (num_atoms + 3,)
    assert padded["edge_index"].shape == (2, num_edges + 5)
    assert padded["shifts"].shape == (num_edges + 5, 3)
    assert padded["_node_mask"].shape == (num_atoms + 3,)
    assert padded["_edge_mask"].shape == (num_edges + 5,)
    assert padded["_real_num_atoms"].item() == num_atoms
    assert torch.all(padded["_node_mask"][:num_atoms] == 1)
    assert torch.all(padded["_node_mask"][num_atoms:] == 0)
    assert torch.all(padded["_edge_mask"][:num_edges] == 1)
    assert torch.all(padded["_edge_mask"][num_edges:] == 0)
    assert torch.all(padded["edge_index"][:, num_edges:] == num_atoms + 2)
    assert torch.all(padded["shifts"][num_edges:, 0] == 10.0)


def test_padded_edge_force_outputs_match_unpadded_real_atoms():
    from mace.modules.utils import get_edge_vectors_and_lengths
    from mace.tools.training_compile import (
        _edge_force_outputs,
        _edge_vector_inputs,
        _pad_edge_force_data_to_bucket,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    data_dict, positions, edge_index, vectors = _edge_vector_inputs(batch)
    ref_energy, ref_forces = _edge_force_outputs(
        model, data_dict, positions, edge_index, vectors
    )

    padded_data = _pad_edge_force_data_to_bucket(
        data_dict,
        atom_bucket=positions.shape[0] + 2,
        edge_bucket=edge_index.shape[1] + 4,
        r_max=5.0,
    )
    padded_vectors, _ = get_edge_vectors_and_lengths(
        positions=padded_data["positions"],
        edge_index=padded_data["edge_index"],
        shifts=padded_data["shifts"],
    )
    padded_vectors = padded_vectors.detach().clone().requires_grad_(True)
    padded_energy, padded_forces = _edge_force_outputs(
        model,
        padded_data,
        padded_data["positions"],
        padded_data["edge_index"],
        padded_vectors,
    )

    assert_close(padded_energy, ref_energy, atol=1e-6, rtol=1e-6)
    assert_close(padded_forces[: positions.shape[0]], ref_forces, atol=1e-6, rtol=1e-6)


def test_loss_from_energy_forces_slices_padded_forces_to_reference_atoms():
    from mace.tools.training_compile import _loss_from_energy_forces

    batch = _BatchDictAdapter(create_batch("cpu"))

    class ShapeCheckingLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["forces"].shape == ref.forces.shape
            return pred["forces"].sum() + pred["energy"].sum()

    padded_forces = torch.cat((batch.forces, torch.ones(2, 3)), dim=0)
    loss = _loss_from_energy_forces(
        batch=batch,
        loss_fn=ShapeCheckingLoss(),
        energy=batch.energy,
        forces=padded_forces,
    )

    assert torch.isfinite(loss)



def test_edge_force_weighted_energy_forces_tensor_loss_matches_loss_module():
    from mace.modules import WeightedEnergyForcesLoss
    from mace.tools.training_compile import (
        _atomic_forces_from_edge_grad,
        _edge_force_energy_and_edge_grad,
        _edge_force_weighted_energy_forces_loss,
        _edge_vector_inputs,
        _loss_from_energy_forces,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.3, forces_weight=7.0)
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    energy, edge_grad = _edge_force_energy_and_edge_grad(model, data_dict, vectors)
    forces = _atomic_forces_from_edge_grad(data_dict, edge_grad)

    reference = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    actual = _edge_force_weighted_energy_forces_loss(
        data_dict=data_dict,
        energy=energy,
        forces=forces,
        energy_loss_weight=loss_fn.energy_weight,
        forces_loss_weight=loss_fn.forces_weight,
    )

    assert_close(actual, reference)


def test_edge_force_weighted_forces_tensor_loss_matches_loss_module():
    from mace.modules import WeightedForcesLoss
    from mace.tools.training_compile import (
        _atomic_forces_from_edge_grad,
        _edge_force_energy_and_edge_grad,
        _edge_force_weighted_forces_loss,
        _edge_vector_inputs,
        _loss_from_energy_forces,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss_fn = WeightedForcesLoss(forces_weight=7.0)
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    energy, edge_grad = _edge_force_energy_and_edge_grad(model, data_dict, vectors)
    forces = _atomic_forces_from_edge_grad(data_dict, edge_grad)

    reference = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    actual = _edge_force_weighted_forces_loss(
        data_dict=data_dict,
        forces=forces,
        forces_loss_weight=loss_fn.forces_weight,
    )

    assert_close(actual, reference)


def test_edge_force_weighted_energy_forces_l1l2_tensor_loss_matches_loss_module():
    from mace.modules import WeightedEnergyForcesL1L2Loss
    from mace.tools.training_compile import (
        _atomic_forces_from_edge_grad,
        _edge_force_energy_and_edge_grad,
        _edge_force_weighted_energy_forces_l1l2_loss,
        _edge_vector_inputs,
        _loss_from_energy_forces,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss_fn = WeightedEnergyForcesL1L2Loss(energy_weight=1.3, forces_weight=7.0)
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    energy, edge_grad = _edge_force_energy_and_edge_grad(model, data_dict, vectors)
    forces = _atomic_forces_from_edge_grad(data_dict, edge_grad)

    reference = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    actual = _edge_force_weighted_energy_forces_l1l2_loss(
        data_dict=data_dict,
        energy=energy,
        forces=forces,
        energy_loss_weight=loss_fn.energy_weight,
        forces_loss_weight=loss_fn.forces_weight,
    )

    assert_close(actual, reference)


def test_position_force_weighted_energy_forces_tensor_loss_matches_loss_module():
    from mace.modules import WeightedEnergyForcesLoss
    from mace.tools.training_compile import (
        _edge_vector_inputs,
        _loss_from_energy_forces,
        _position_force_energy_and_forces,
        _position_force_weighted_energy_forces_loss,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.3, forces_weight=7.0)
    data_dict, positions, _, _ = _edge_vector_inputs(batch)
    energy, forces = _position_force_energy_and_forces(model, data_dict, positions)

    reference = _loss_from_energy_forces(
        batch=batch,
        loss_fn=loss_fn,
        energy=energy,
        forces=forces,
    )
    actual = _position_force_weighted_energy_forces_loss(
        data_dict=data_dict,
        energy=energy,
        forces=forces,
        energy_loss_weight=loss_fn.energy_weight,
        forces_loss_weight=loss_fn.forces_weight,
    )

    assert_close(forces, model(batch.to_dict(), training=True)["forces"])
    assert_close(actual, reference)


def test_edge_force_snapshot_scatters_compiled_edge_grad_outside_executable(monkeypatch):
    from mace.tools import training_compile
    from mace.tools.training_compile import (
        _edge_force_value_snapshot_from_executable,
        _edge_vector_inputs,
        edge_force_compile_input_names,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    data_dict, _, _, _ = _edge_vector_inputs(batch)
    input_names = edge_force_compile_input_names(data_dict.keys())
    expected_forces = torch.full_like(batch.forces, 2.0)
    calls = []

    def executable(vectors, *input_tensors):
        del input_tensors
        edge_grad = torch.ones_like(vectors)
        return torch.zeros_like(batch.energy), edge_grad

    def fake_edge_gradient_to_atomic_forces(edge_grad, *, edge_index, num_atoms):
        calls.append((edge_grad.shape, edge_index.shape, num_atoms))
        return expected_forces.clone()

    class ForceShapeLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["forces"].shape == ref.forces.shape
            return pred["forces"].sum() + pred["energy"].sum()

    monkeypatch.setattr(
        training_compile,
        "edge_gradient_to_atomic_forces",
        fake_edge_gradient_to_atomic_forces,
    )

    snapshot = _edge_force_value_snapshot_from_executable(
        model=model,
        batch=batch,
        loss_fn=ForceShapeLoss(),
        executable=executable,
        input_names=input_names,
    )

    assert calls == [
        (torch.Size((batch.edge_index.shape[1], 3)), batch.edge_index.shape, batch.positions.shape[0])
    ]
    assert_close(snapshot["forces"], expected_forces)


def test_edge_vector_inputs_leave_vectors_non_differentiable_for_core_leaf():
    from mace.tools.training_compile import _edge_vector_inputs

    data_dict, _, _, vectors = _edge_vector_inputs(_BatchDictAdapter(create_batch("cpu")))

    assert vectors.requires_grad is False
    assert data_dict["positions"].requires_grad is False


def test_edge_force_trace_strips_boundary_detach_for_training_backward():
    from mace.tools.force_compile import count_fx_detach_nodes, trace_force_closure
    from mace.tools.training_compile import (
        _edge_force_energy_and_edge_grad,
        _edge_vector_inputs,
        edge_force_compile_input_names,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    data_dict, _, _, vectors = _edge_vector_inputs(batch)
    input_names = edge_force_compile_input_names(data_dict.keys())
    example_inputs = tuple(data_dict[name] for name in input_names)

    def closure(vectors_arg, *data_tensors):
        current_data = dict(data_dict)
        current_data.update(zip(input_names, data_tensors, strict=True))
        return _edge_force_energy_and_edge_grad(model, current_data, vectors_arg)

    trace_result = trace_force_closure(
        closure,
        (vectors, *example_inputs),
        tracing_mode="real",
        strip_detach=True,
        strip_all_detach=True,
    )

    assert trace_result.detach_nodes_before > 0
    assert trace_result.detach_nodes_after == 0
    assert count_fx_detach_nodes(trace_result.graph_module) == 0


def test_edge_vector_inputs_can_pad_to_bucket_shape():
    from mace.tools.training_compile import _edge_vector_inputs

    batch = _BatchDictAdapter(create_batch("cpu"))
    atom_bucket = batch.positions.shape[0] + 2
    edge_bucket = batch.edge_index.shape[1] + 4

    data_dict, positions, edge_index, vectors = _edge_vector_inputs(
        batch, atom_bucket=atom_bucket, edge_bucket=edge_bucket, r_max=5.0
    )

    assert data_dict["positions"].shape[0] == atom_bucket
    assert data_dict["edge_index"].shape[1] == edge_bucket
    assert positions.shape[0] == atom_bucket
    assert edge_index.shape[1] == edge_bucket
    assert vectors.shape[0] == edge_bucket
    assert data_dict["_node_mask"].shape[0] == atom_bucket
    assert data_dict["_edge_mask"].shape[0] == edge_bucket


def test_edge_force_compile_dynamic_cache_key_groups_different_atom_edge_shapes():
    from mace.tools.training_compile import edge_force_compile_dynamic_cache_key

    left = edge_force_compile_dynamic_cache_key(
        input_shapes={
            "positions": (286, 3),
            "edge_index": (2, 4632),
            "shifts": (4632, 3),
            "node_attrs": (286, 4),
            "batch": (286,),
            "ptr": (33,),
        }
    )
    right = edge_force_compile_dynamic_cache_key(
        input_shapes={
            "positions": (441, 3),
            "edge_index": (2, 10274),
            "shifts": (10274, 3),
            "node_attrs": (441, 4),
            "batch": (441,),
            "ptr": (33,),
        }
    )

    assert left == right
    assert ("positions", (-1, 3)) in left[-1]
    assert ("edge_index", (2, -1)) in left[-1]
    assert ("shifts", (-1, 3)) in left[-1]
    assert ("ptr", (33,)) in left[-1]


def test_edge_force_compile_bucket_cache_key_groups_nearby_shapes():
    from mace.tools.training_compile import edge_force_compile_bucket_cache_key

    left = edge_force_compile_bucket_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={
            "positions": (286, 3),
            "edge_index": (2, 4632),
            "node_attrs": (286, 4),
            "batch": (286,),
            "ptr": (33,),
        },
        bucket_atoms=(320,),
        bucket_edges=(5000,),
        bucket_margin=2.0,
    )
    right = edge_force_compile_bucket_cache_key(
        num_atoms=290,
        num_edges=4700,
        input_shapes={
            "positions": (290, 3),
            "edge_index": (2, 4700),
            "node_attrs": (290, 4),
            "batch": (290,),
            "ptr": (33,),
        },
        bucket_atoms=(320,),
        bucket_edges=(5000,),
        bucket_margin=2.0,
    )

    assert left == right
    assert left is not None
    assert left[0] == "bucket"
    assert ("positions", (320, 3)) in left[-1]
    assert ("edge_index", (2, 5000)) in left[-1]


def test_edge_force_compile_bucket_cache_key_rejects_oversized_bucket():
    from mace.tools.training_compile import edge_force_compile_bucket_cache_key

    cache_key = edge_force_compile_bucket_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={
            "positions": (286, 3),
            "edge_index": (2, 4632),
        },
        bucket_atoms=(512,),
        bucket_edges=(8192,),
        bucket_margin=1.15,
    )

    assert cache_key is None


def test_edge_force_compile_bucket_cache_key_zero_margin_accepts_small_inputs():
    from mace.tools.training_compile import edge_force_compile_bucket_cache_key

    cache_key = edge_force_compile_bucket_cache_key(
        num_atoms=286,
        num_edges=4632,
        input_shapes={
            "positions": (286, 3),
            "edge_index": (2, 4632),
        },
        bucket_atoms=(768,),
        bucket_edges=(32768,),
        bucket_margin=0.0,
    )

    assert cache_key is not None
    assert cache_key[1:3] == (768, 32768)
    assert ("positions", (768, 3)) in cache_key[-1]
    assert ("edge_index", (2, 32768)) in cache_key[-1]


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



def test_edge_force_compiled_loss_does_not_mutate_functorch_donated_buffer_on_init():
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
        assert graph_compiled.functorch_donated_buffer_disabled is False
        assert functorch_config.donated_buffer is True
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

    batch = _BatchDictAdapter(create_batch("cpu"))
    loss, metrics = prepared.compiled_force_training_loss(
        batch=batch,
        loss_fn=_EnergyForcesMiniLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert torch.isfinite(loss)
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "min_repeats"
    assert metrics["edge_force_compile_cache_policy"] == "repeat_only"
    assert metrics["edge_force_cache_seen_count"] == 1
    assert metrics["edge_force_num_atoms"] == batch.positions.shape[0]
    assert metrics["edge_force_num_edges"] == batch.edge_index.shape[1]



def test_edge_force_compiled_loss_returns_tensor_loss_for_weighted_energy_forces():
    from mace.modules import WeightedEnergyForcesLoss
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
            cache_policy="dynamic",
            setup_gate="none",
            cache_hit_gate=False,
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss, metrics = prepared.compiled_force_training_loss(
        batch=batch,
        loss_fn=WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=100.0),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_compile_loss"] is True
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_edge_force_energy_force_output_loss_support_matrix():
    from mace.modules import (
        WeightedEnergyForcesL1L2Loss,
        WeightedEnergyForcesLoss,
        WeightedEnergyForcesStressLoss,
        WeightedEnergyForcesVirialsLoss,
        WeightedForcesLoss,
    )
    from mace.tools.training_compile import _edge_force_can_use_energy_force_outputs

    assert _edge_force_can_use_energy_force_outputs(WeightedEnergyForcesLoss())
    assert _edge_force_can_use_energy_force_outputs(WeightedForcesLoss())
    assert _edge_force_can_use_energy_force_outputs(WeightedEnergyForcesL1L2Loss())
    assert not _edge_force_can_use_energy_force_outputs(
        WeightedEnergyForcesStressLoss()
    )
    assert not _edge_force_can_use_energy_force_outputs(
        WeightedEnergyForcesVirialsLoss()
    )


def test_edge_force_loss_output_capability_registry():
    from mace.modules import (
        DipolePolarLoss,
        WeightedEnergyForcesLoss,
        WeightedEnergyForcesStressLoss,
        WeightedEnergyForcesVirialsLoss,
        WeightedForcesLoss,
    )
    from mace.tools.training_compile import edge_force_loss_output_capability

    assert edge_force_loss_output_capability(
        WeightedEnergyForcesLoss()
    ).required_outputs == ("energy", "forces")
    assert edge_force_loss_output_capability(
        WeightedForcesLoss()
    ).required_outputs == ("forces",)
    assert edge_force_loss_output_capability(
        WeightedEnergyForcesStressLoss()
    ).required_outputs == ("energy", "forces", "stress")
    assert edge_force_loss_output_capability(
        WeightedEnergyForcesVirialsLoss()
    ).required_outputs == ("energy", "forces", "virials")
    assert edge_force_loss_output_capability(
        DipolePolarLoss()
    ).required_outputs == ("dipole", "polarizability")


def test_edge_force_loss_output_capability_treats_unknown_loss_as_unsupported():
    from mace.tools.training_compile import edge_force_loss_output_capability

    class CustomLoss(torch.nn.Module):
        pass

    capability = edge_force_loss_output_capability(CustomLoss())

    assert capability.required_outputs == ()
    assert capability.edge_force_supported is False
    assert capability.unsupported_reason == "unknown_loss"


def test_edge_force_compiled_loss_falls_back_for_unknown_loss():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        EdgeForceCompiledLossModule,
    )

    class BatchWithGeometry:
        def __init__(self):
            self.x = torch.ones(())
            self.positions = torch.zeros(1, 3)
            self.edge_index = torch.empty(2, 0, dtype=torch.long)

        def to_dict(self):
            return {"x": self.x}

    class EagerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = 0

        def forward(self, batch, **kwargs):
            self.calls += 1
            assert kwargs["compute_force"] is True
            return {"value": batch["x"] * self.weight}

    class CustomLoss(torch.nn.Module):
        def forward(self, pred, ref):
            return pred["value"]

    model = EagerModel()
    compiled_loss = EdgeForceCompiledLossModule(
        model, config=EdgeForceCompileConfig(enabled=True, compile_graph=False)
    )

    loss, metrics = compiled_loss.compiled_force_training_loss(
        batch=BatchWithGeometry(),
        loss_fn=CustomLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.calls == 1
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "unknown_loss"


def test_edge_force_compiled_loss_supports_weighted_forces_loss():
    from mace.modules import WeightedForcesLoss
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
            cache_policy="dynamic",
            setup_gate="none",
            cache_hit_gate=False,
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss, metrics = prepared.compiled_force_training_loss(
        batch=batch,
        loss_fn=WeightedForcesLoss(forces_weight=100.0),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_compile_loss"] is True
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_edge_force_compiled_loss_supports_weighted_energy_forces_l1l2_loss():
    from mace.modules import WeightedEnergyForcesL1L2Loss
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
            cache_policy="dynamic",
            setup_gate="none",
            cache_hit_gate=False,
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss, metrics = prepared.compiled_force_training_loss(
        batch=batch,
        loss_fn=WeightedEnergyForcesL1L2Loss(energy_weight=1.0, forces_weight=100.0),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_compile_loss"] is True
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_edge_force_compiled_loss_eager_fallback_preserves_requested_outputs():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        EdgeForceCompiledLossModule,
    )

    class StressVirialsFallbackModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(
            self,
            batch_dict,
            *,
            training,
            compute_force,
            compute_virials,
            compute_stress,
        ):
            self.calls.append(
                {
                    "training": training,
                    "compute_force": compute_force,
                    "compute_virials": compute_virials,
                    "compute_stress": compute_stress,
                }
            )
            return {
                "energy": self.weight.reshape(1),
                "forces": torch.zeros_like(batch_dict["positions"]),
                "virials": (
                    self.weight.reshape(1, 1, 1).expand(1, 3, 3)
                    if compute_virials
                    else None
                ),
                "stress": (
                    self.weight.reshape(1, 1, 1).expand(1, 3, 3)
                    if compute_stress
                    else None
                ),
            }

    class OutputCheckingLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["stress"] is not None
            assert pred["virials"] is not None
            return pred["stress"].sum() + pred["virials"].sum()

    class MinimalBatch:
        positions = torch.zeros(1, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        def to_dict(self):
            return {"positions": self.positions, "edge_index": self.edge_index}

    model = StressVirialsFallbackModel()
    wrapper = EdgeForceCompiledLossModule(
        model,
        config=EdgeForceCompileConfig(enabled=True),
    )

    loss, metrics = wrapper.compiled_force_training_loss(
        batch=MinimalBatch(),
        loss_fn=OutputCheckingLoss(),
        output_args={"forces": True, "virials": True, "stress": True},
    )

    assert torch.isfinite(loss)
    assert model.calls == [
        {
            "training": True,
            "compute_force": True,
            "compute_virials": True,
            "compute_stress": True,
        }
    ]
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "unsupported_outputs"


def test_edge_force_compiled_loss_disabled_fallback_preserves_requested_outputs():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        EdgeForceCompiledLossModule,
    )

    class StressVirialsFallbackModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(
            self,
            batch_dict,
            *,
            training,
            compute_force,
            compute_virials,
            compute_stress,
        ):
            self.calls.append(
                {
                    "training": training,
                    "compute_force": compute_force,
                    "compute_virials": compute_virials,
                    "compute_stress": compute_stress,
                }
            )
            return {
                "energy": self.weight.reshape(1),
                "forces": torch.zeros_like(batch_dict["positions"]),
                "virials": (
                    self.weight.reshape(1, 1, 1).expand(1, 3, 3)
                    if compute_virials
                    else None
                ),
                "stress": (
                    self.weight.reshape(1, 1, 1).expand(1, 3, 3)
                    if compute_stress
                    else None
                ),
            }

    class OutputCheckingLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["stress"] is not None
            assert pred["virials"] is not None
            return pred["stress"].sum() + pred["virials"].sum()

    class MinimalBatch:
        positions = torch.zeros(1, 3)
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        def to_dict(self):
            return {"positions": self.positions, "edge_index": self.edge_index}

    model = StressVirialsFallbackModel()
    wrapper = EdgeForceCompiledLossModule(
        model,
        config=EdgeForceCompileConfig(enabled=True),
    )
    wrapper.disabled = True

    loss, metrics = wrapper.compiled_force_training_loss(
        batch=MinimalBatch(),
        loss_fn=OutputCheckingLoss(),
        output_args={"forces": True, "virials": True, "stress": True},
    )

    assert torch.isfinite(loss)
    assert model.calls == [
        {
            "training": True,
            "compute_force": True,
            "compute_virials": True,
            "compute_stress": True,
        }
    ]
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "disabled"


def test_edge_force_compiled_loss_can_use_position_gradient_mode():
    from mace.modules import WeightedEnergyForcesLoss
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
            cache_policy="dynamic",
            setup_gate="none",
            cache_hit_gate=False,
            force_gradient_mode="positions",
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    loss, metrics = prepared.compiled_force_training_loss(
        batch=batch,
        loss_fn=WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=100.0),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_compile_loss"] is True
    assert metrics["edge_force_gradient_mode"] == "positions"
    loss.backward()
    assert any(param.grad is not None for param in model.parameters())


def test_edge_force_compiled_loss_bucket_policy_compiles_padded_inputs():
    from mace.tools.train import take_step
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    model = create_tiny_mace("cpu")
    batch = _BatchDictAdapter(create_batch("cpu"))
    atom_bucket = batch.positions.shape[0] + 2
    edge_bucket = batch.edge_index.shape[1] + 4
    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_hit_gate=True,
            cache_policy="bucket",
            bucket_atoms=(atom_bucket,),
            bucket_edges=(edge_bucket,),
            bucket_margin=2.0,
            allow_fallback=False,
        ),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)

    _, metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert metrics["edge_force_compile"] is True
    assert metrics["edge_force_cache_hit"] is False
    compiled = next(iter(prepared.cache.values()))
    assert "_node_mask" in compiled.input_names
    assert "_edge_mask" in compiled.input_names
    assert ("positions", (atom_bucket, 3)) in compiled.cache_key[-1]
    assert ("edge_index", (2, edge_bucket)) in compiled.cache_key[-1]
    assert ("_node_mask", (atom_bucket,)) in compiled.cache_key[-1]
    assert ("_edge_mask", (edge_bucket,)) in compiled.cache_key[-1]

    _, hit_metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )
    assert hit_metrics["edge_force_cache_hit"] is True
    assert hit_metrics["edge_force_cache_hit_gate_accepted"] is True


def test_edge_force_compiled_loss_bucket_cache_hit_across_smaller_shape():
    from mace.tools.train import take_step
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        prepare_edge_force_compiled_loss,
    )

    def truncate_batch(batch_dict, keep_atoms):
        keep_edges = (batch_dict["edge_index"] < keep_atoms).all(dim=0)
        out = dict(batch_dict)
        for key in ("positions", "node_attrs", "forces", "batch", "charges", "density_coefficients"):
            if key in out and torch.is_tensor(out[key]) and out[key].shape[0] == batch_dict["positions"].shape[0]:
                out[key] = out[key][:keep_atoms].clone()
        out["edge_index"] = out["edge_index"][:, keep_edges].clone()
        for key in ("shifts", "unit_shifts"):
            if key in out and torch.is_tensor(out[key]) and out[key].shape[0] == keep_edges.shape[0]:
                out[key] = out[key][keep_edges].clone()
        out["ptr"] = torch.tensor([0, keep_atoms], dtype=batch_dict["ptr"].dtype)
        return out

    model = create_tiny_mace("cpu")
    first = _BatchDictAdapter(create_batch("cpu"))
    second = _BatchDictAdapter(truncate_batch(first.to_dict(), first.positions.shape[0] - 4))
    atom_bucket = first.positions.shape[0] + 2
    edge_bucket = first.edge_index.shape[1] + 4
    prepared = prepare_edge_force_compiled_loss(
        model,
        config=EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_hit_gate=True,
            cache_policy="bucket",
            bucket_atoms=(atom_bucket,),
            bucket_edges=(edge_bucket,),
            bucket_margin=2.0,
            allow_fallback=False,
        ),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)

    take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=first,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )
    _, metrics = take_step(
        model=prepared,
        loss_fn=_EnergyForcesMiniLoss(),
        batch=second,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert metrics["edge_force_cache_hit"] is True
    assert metrics["edge_force_cache_hit_gate_accepted"] is True


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


def test_edge_force_compile_step_promotes_trainable_parameters_as_inputs():
    from mace.tools import training_compile

    model = create_tiny_mace("cpu")
    wrapper = training_compile.EdgeForceCompiledLossModule(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=False,
            cache_hit_gate=False,
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))
    data_dict, _, _, _ = training_compile._edge_vector_inputs(batch)
    input_names = training_compile.edge_force_compile_input_names(data_dict.keys())
    cache_key = wrapper._cache_key(batch=batch, input_names=input_names, data_dict=data_dict)

    compiled = wrapper._compile_step(
        batch=batch,
        loss_fn=_EnergyForcesMiniLoss(),
        cache_key=cache_key,
    )

    expected = tuple(
        name for name, param in model.named_parameters() if param.requires_grad
    )
    assert compiled.param_names == expected
    assert compiled.param_names


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
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
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
            refresh_executable_each_step=True,
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


def test_edge_force_compile_metrics_report_setup_phase_breakdown(monkeypatch, caplog):
    import types

    from mace.tools import training_compile

    caplog.set_level("INFO")

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        def executable(vectors, *input_tensors):
            del input_tensors
            energy = torch.ones(
                1, dtype=vectors.dtype, device=vectors.device, requires_grad=True
            )
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    def snapshot_payload():
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {},
        }

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        lambda **kwargs: snapshot_payload(),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        lambda **kwargs: snapshot_payload(),
    )

    wrapper = training_compile.EdgeForceCompiledLossModule(
        create_tiny_mace("cpu"),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))

    _, metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    expected_phase_keys = {
        "edge_force_compile_trace_seconds",
        "edge_force_compile_gate_compile_seconds",
        "edge_force_compile_gate_reference_seconds",
        "edge_force_compile_gate_candidate_seconds",
        "edge_force_compile_training_compile_seconds",
    }
    assert expected_phase_keys <= set(metrics)
    assert all(metrics[key] >= 0.0 for key in expected_phase_keys)
    phase_total = sum(metrics[key] for key in expected_phase_keys)
    assert metrics["edge_force_compile_setup_seconds"] >= phase_total
    setup_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Edge-force compile setup phase input_prep" in setup_messages
    assert "Edge-force compile setup phase trace" in setup_messages
    assert "Edge-force compile setup phase gate_compile" in setup_messages



def test_edge_force_compile_setup_gate_none_skips_expensive_setup_snapshots(monkeypatch):
    import types

    from mace.tools import training_compile

    snapshot_calls = []

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        def executable(vectors, *input_tensors):
            del input_tensors
            energy = torch.ones(
                1, dtype=vectors.dtype, device=vectors.device, requires_grad=True
            )
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    def fail_snapshot(**kwargs):
        del kwargs
        snapshot_calls.append("called")
        raise AssertionError("setup gate snapshots should be skipped")

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        fail_snapshot,
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        fail_snapshot,
    )

    wrapper = training_compile.EdgeForceCompiledLossModule(
        create_tiny_mace("cpu"),
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
            setup_gate="none",
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))

    _, metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert snapshot_calls == []
    assert metrics["edge_force_setup_gate"] == "none"
    assert metrics["edge_force_gate_accepted"] is None
    assert metrics["edge_force_compile_gate_reference_seconds"] == 0.0
    assert metrics["edge_force_compile_gate_candidate_seconds"] == 0.0


def test_edge_force_cache_hit_refreshes_runtime_executable(monkeypatch):
    import types

    from mace.tools import training_compile

    compiled_labels = []
    used_labels = []
    compile_graph_modules = []
    traced_graph_module = types.SimpleNamespace(
        graph=types.SimpleNamespace(nodes=[object()])
    )
    rebuilt_graph_modules = []

    def fake_rebuild_fx_graph_module(graph_module):
        rebuilt = types.SimpleNamespace(
            graph=types.SimpleNamespace(nodes=[object()]),
            original=graph_module,
            label=f"rebuilt{len(rebuilt_graph_modules)}",
        )
        rebuilt_graph_modules.append(rebuilt)
        return rebuilt

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=traced_graph_module,
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(graph_module, *args, **kwargs):
        label = f"exec{len(compiled_labels)}"
        compiled_labels.append(label)
        compile_graph_modules.append(graph_module)

        def executable(vectors, *input_tensors):
            del input_tensors
            used_labels.append(label)
            energy = torch.ones(1, dtype=vectors.dtype, device=vectors.device, requires_grad=True)
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    def snapshot_payload():
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {},
        }

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

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
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        lambda **kwargs: snapshot_payload(),
    )
    monkeypatch.setattr(
        training_compile,
        "rebuild_fx_graph_module",
        fake_rebuild_fx_graph_module,
    )

    model = create_tiny_mace("cpu")
    wrapper = training_compile.EdgeForceCompiledLossModule(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
            refresh_executable_each_step=True,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))

    wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )
    used_labels.clear()

    loss, metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert loss.requires_grad is True
    assert metrics["edge_force_cache_hit"] is True
    assert len(compiled_labels) == 3
    assert used_labels == ["exec2"]
    assert all(module is not traced_graph_module for module in compile_graph_modules)
    assert len({id(module) for module in compile_graph_modules}) == len(compile_graph_modules)
    assert compile_graph_modules[2] is rebuilt_graph_modules[-1]
    assert metrics["edge_force_runtime_recompile"] is True
    assert next(iter(wrapper.cache.values())).executable is None


def test_edge_force_cache_hit_does_not_request_outer_retain_graph(monkeypatch):
    import types

    from mace.tools import training_compile

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        def executable(vectors, *input_tensors):
            del input_tensors
            energy = torch.ones(1, dtype=vectors.dtype, device=vectors.device, requires_grad=True)
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    def snapshot_payload():
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {},
        }

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        lambda **kwargs: snapshot_payload(),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        lambda **kwargs: snapshot_payload(),
    )

    model = create_tiny_mace("cpu")
    wrapper = training_compile.EdgeForceCompiledLossModule(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
            refresh_executable_each_step=False,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))

    _, first_metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )
    _, hit_metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert "_retain_graph_for_backward" not in first_metrics
    assert hit_metrics["edge_force_cache_hit"] is True
    assert "_retain_graph_for_backward" not in hit_metrics


def test_edge_force_compile_periodic_parity_check_records_gradient_diffs(monkeypatch):
    import types

    from mace.tools import training_compile

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        def executable(vectors, *input_tensors):
            first_param = input_tensors[0]
            energy = first_param.reshape(-1)[:1] * 0.0 + torch.ones(
                1, dtype=vectors.dtype, device=vectors.device
            )
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

    model = create_tiny_mace("cpu")
    snapshot_calls = []

    def snapshot_payload(label):
        snapshot_calls.append(label)
        first_param = next(model.parameters())
        first_param.grad = torch.ones_like(first_param)
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {"checked.weight": torch.zeros(1)},
        }

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_snapshot",
        lambda **kwargs: snapshot_payload("reference"),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_snapshot_from_executable",
        lambda **kwargs: snapshot_payload("candidate"),
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        lambda **kwargs: snapshot_payload("compile_reference"),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        lambda **kwargs: snapshot_payload("compile_candidate"),
    )

    wrapper = training_compile.EdgeForceCompiledLossModule(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
            refresh_executable_each_step=False,
            parity_check_interval=1,
        ),
    )
    batch = _BatchDictAdapter(create_batch("cpu"))

    wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )
    snapshot_calls.clear()
    loss, metrics = wrapper.compiled_force_training_loss(
        batch=batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert loss.requires_grad is True
    assert snapshot_calls == ["reference", "candidate"]
    assert metrics["edge_force_parity_check"] is True
    assert metrics["edge_force_parity_accepted"] is True
    assert metrics["edge_force_parity_failed_checks"] == 0
    assert metrics["edge_force_parity_loss_abs_diff"] == 0.0
    assert metrics["edge_force_parity_forces_max_abs_diff"] == 0.0
    assert metrics["edge_force_parity_param_grad_max_abs_diff"] == 0.0
    assert all(param.grad is None for param in model.parameters())



def test_edge_force_compile_fixed_probe_reuses_first_batch(monkeypatch):
    import types

    from mace.tools import training_compile

    def fake_trace_force_closure(*args, **kwargs):
        return types.SimpleNamespace(
            graph_module=types.SimpleNamespace(
                graph=types.SimpleNamespace(nodes=[object()])
            ),
            detach_nodes_before=0,
            detach_nodes_after=0,
        )

    def fake_compile_fx_graph_module(*args, **kwargs):
        def executable(vectors, *input_tensors):
            first_param = input_tensors[0]
            energy = first_param.reshape(-1)[:1] * 0.0 + torch.ones(
                1, dtype=vectors.dtype, device=vectors.device
            )
            edge_grad = torch.zeros_like(vectors)
            return energy, edge_grad

        return executable, {"compile_graph": kwargs["compile_graph"]}

    class EnergyOnlyLoss(torch.nn.Module):
        def forward(self, pred, ref):
            del ref
            return pred["energy"].sum()

    model = create_tiny_mace("cpu")
    snapshot_batches = []

    def snapshot_payload(label, batch, **kwargs):
        del kwargs
        snapshot_batches.append((label, int(batch._probe_marker.item())))
        return {
            "energy": torch.zeros(1),
            "forces": torch.zeros(1, 3),
            "loss": torch.zeros(()),
            "grads": {},
        }

    monkeypatch.setattr(
        training_compile, "trace_force_closure", fake_trace_force_closure
    )
    monkeypatch.setattr(
        training_compile, "compile_fx_graph_module", fake_compile_fx_graph_module
    )
    monkeypatch.setattr(
        training_compile, "rebuild_fx_graph_module", lambda graph_module: graph_module
    )
    monkeypatch.setattr(
        training_compile,
        "_position_force_value_snapshot",
        lambda **kwargs: snapshot_payload("reference", **kwargs),
    )
    monkeypatch.setattr(
        training_compile,
        "_edge_force_value_snapshot_from_executable",
        lambda **kwargs: snapshot_payload("candidate", **kwargs),
    )

    wrapper = training_compile.EdgeForceCompiledLossModule(
        model,
        config=training_compile.EdgeForceCompileConfig(
            enabled=True,
            compile_graph=True,
            cache_hit_gate=False,
            cache_policy="shape",
            allow_fallback=False,
            refresh_executable_each_step=False,
            fixed_probe_interval=1,
        ),
    )
    first_data = create_batch("cpu")
    first_data["_probe_marker"] = torch.tensor(1)
    second_data = create_batch("cpu")
    second_data["_probe_marker"] = torch.tensor(2)
    first_batch = _BatchDictAdapter(first_data)
    second_batch = _BatchDictAdapter(second_data)

    wrapper.compiled_force_training_loss(
        batch=first_batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )
    snapshot_batches.clear()
    _, metrics = wrapper.compiled_force_training_loss(
        batch=second_batch,
        loss_fn=EnergyOnlyLoss(),
        output_args={"forces": True, "virials": False, "stress": False},
    )

    assert snapshot_batches == [("reference", 1), ("candidate", 1)]
    assert metrics["edge_force_fixed_probe_check"] is True
    assert metrics["edge_force_fixed_probe_num_atoms"] == first_batch.positions.shape[0]
    assert metrics["edge_force_fixed_probe_forces_max_abs_diff"] == 0.0


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


def test_edge_force_symbolic_model_build_context_disables_e3nn_jit_fx():
    import argparse
    import e3nn

    from mace.cli.run_train import _model_build_context_from_args

    args = argparse.Namespace(
        edge_force_compile=True,
        edge_force_compile_tracing_mode="symbolic",
    )
    e3nn.set_optimization_defaults(jit_script_fx=True)

    with _model_build_context_from_args(args):
        assert e3nn.get_optimization_defaults()["jit_script_fx"] is False

    assert e3nn.get_optimization_defaults()["jit_script_fx"] is True


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
            "--train_tf32",
            "--no-edge_force_compile_graph",
            "--no-edge_force_compile_dynamic",
            "--no-edge_force_compile_shape_padding",
            "--edge_force_compile_max_fusion_size",
            "1",
            "--edge_force_compile_spherical_harmonics",
            "e3nn",
            "--edge_force_compile_force_gradient_mode",
            "positions",
            "--edge_force_compile_strip_detach",
            "--edge_force_compile_setup_gate",
            "none",
            "--no-edge_force_compile_cache_hit_gate",
            "--edge_force_compile_atol",
            "5e-5",
            "--edge_force_compile_rtol",
            "2e-4",
            "--edge_force_compile_cache_policy",
            "break_even",
            "--edge_force_compile_min_repeats",
            "3",
            "--edge_force_compile_break_even_expected_remaining_hits",
            "11",
            "--edge_force_compile_max_cache_entries",
            "7",
            "--edge_force_compile_bucket_atoms",
            "256,512",
            "--edge_force_compile_bucket_edges",
            "2048,4096",
            "--edge_force_compile_bucket_margin",
            "1.15",
            "--edge_force_compile_parity_check_interval",
            "17",
            "--no-edge_force_compile_parity_check_gradients",
            "--no-edge_force_compile_parity_check_strict",
            "--edge_force_compile_fixed_probe_interval",
            "19",
            "--edge_force_compile_fixed_probe_gradients",
            "--edge_force_compile_fixed_probe_strict",
            "--no-edge_force_compile_disable_negative_speedup",
            "--no-edge_force_compile_allow_fallback",
        ]
    )

    assert args.edge_force_compile is True
    assert args.edge_force_compile_mode == "reduce-overhead"
    assert args.edge_force_compile_tracing_mode == "symbolic"
    assert args.train_tf32 is True
    assert args.edge_force_compile_graph is False
    assert args.edge_force_compile_dynamic is False
    assert args.edge_force_compile_shape_padding is False
    assert args.edge_force_compile_max_fusion_size == 1
    assert args.edge_force_compile_spherical_harmonics == "e3nn"
    assert args.edge_force_compile_force_gradient_mode == "positions"
    assert args.edge_force_compile_strip_detach is True
    assert args.edge_force_compile_setup_gate == "none"
    assert args.edge_force_compile_cache_hit_gate is False
    assert args.edge_force_compile_atol == 5e-5
    assert args.edge_force_compile_rtol == 2e-4
    assert args.edge_force_compile_cache_policy == "break_even"
    assert args.edge_force_compile_min_repeats == 3
    assert args.edge_force_compile_break_even_expected_remaining_hits == 11
    assert args.edge_force_compile_max_cache_entries == 7
    assert args.edge_force_compile_bucket_atoms == "256,512"
    assert args.edge_force_compile_bucket_edges == "2048,4096"
    assert args.edge_force_compile_bucket_margin == 1.15
    assert args.edge_force_compile_parity_check_interval == 17
    assert args.edge_force_compile_parity_check_gradients is False
    assert args.edge_force_compile_parity_check_strict is False
    assert args.edge_force_compile_fixed_probe_interval == 19
    assert args.edge_force_compile_fixed_probe_gradients is True
    assert args.edge_force_compile_fixed_probe_strict is True
    assert args.edge_force_compile_disable_negative_speedup is False
    assert args.edge_force_compile_allow_fallback is False


def test_arg_parser_edge_force_diagnostic_flags_default_to_off():
    from mace.tools import build_default_arg_parser

    default_args = build_default_arg_parser().parse_args(["--name", "edge-force-default"])
    enabled_args = build_default_arg_parser().parse_args(
        [
            "--name",
            "edge-force-diagnostic",
            "--edge_force_compile_cache_hit_gate",
            "--edge_force_compile_strip_detach",
        ]
    )

    assert default_args.edge_force_compile_cache_hit_gate is False
    assert default_args.edge_force_compile_strip_detach is False
    assert enabled_args.edge_force_compile_cache_hit_gate is True
    assert enabled_args.edge_force_compile_strip_detach is True


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


def test_e3nn_to_cueq_skips_missing_symmetric_contraction_target_key():
    import torch
    from e3nn import o3

    from mace.cli.convert_e3nn_cueq import transfer_symmetric_contractions

    class SymmetricContractions:
        irreps_in = o3.Irreps("1x0e")
        irreps_out = o3.Irreps("1x0e")

    class Product:
        symmetric_contractions = SymmetricContractions()

    source_dict = {
        "products.0.symmetric_contractions.contractions.0.weights_max": torch.ones(2, 1, 4),
        "products.0.symmetric_contractions.contractions.0.weights.0": torch.ones(2, 1, 4),
        "products.0.symmetric_contractions.contractions.0.weights.1": torch.ones(2, 1, 4),
    }
    target_dict = {}

    transfer_symmetric_contractions(
        source_dict=source_dict,
        target_dict=target_dict,
        num_product_irreps=1,
        products=[Product()],
        correlation=3,
        num_layers=1,
        use_reduced_cg=False,
        keep_last_layer_irreps=False,
    )

    assert "products.0.symmetric_contractions.weight" not in target_dict



def test_e3nn_to_cueq_no_optimized_preserves_e3nn_layout_and_weights():
    from copy import deepcopy

    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.modules import wrapper_ops

    if not wrapper_ops.CUET_AVAILABLE:
        pytest.skip("cuequivariance_torch is not available")

    source = create_tiny_mace("cpu", seed=2027, correlation=3)
    target = run_e3nn_to_cueq(
        deepcopy(source),
        device="cpu",
        layout="ir_mul",
        optimize_all=False,
        optimize_linear=False,
        optimize_channelwise=False,
        optimize_symmetric=False,
        optimize_fctp=False,
        conv_fusion=False,
    )

    source_state = source.state_dict()
    target_state = target.state_dict()
    symmetric_keys = [
        key for key in source_state if "symmetric_contractions" in key and key in target_state
    ]
    assert symmetric_keys
    for key in symmetric_keys:
        assert_close(target_state[key], source_state[key])

    batch = create_batch("cpu")
    source_out = source(batch, training=False, compute_force=False)
    target_out = target(batch, training=False, compute_force=False)

    assert_close(target_out["energy"], source_out["energy"], atol=1.0e-6, rtol=1.0e-6)


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


class _RaisesCudaOOM(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("CUDA out of memory while running compiled path")


def test_runtime_compile_fallback_does_not_swallow_cuda_oom():
    from mace.tools.training_compile import RuntimeFallbackCompiledModule

    wrapper = RuntimeFallbackCompiledModule(
        eager_model=torch.nn.Linear(1, 1),
        compiled_model=_RaisesCudaOOM(),
        allow_fallback=True,
    )

    try:
        wrapper(torch.ones(1, 1))
    except RuntimeError as exc:
        assert "out of memory" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert wrapper.disabled is False


def test_edge_force_compile_fallback_rejects_nonfinite_errors():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        EdgeForceCompiledLossModule,
    )

    module = EdgeForceCompiledLossModule(
        torch.nn.Linear(1, 1),
        config=EdgeForceCompileConfig(enabled=True, allow_fallback=True),
    )

    assert (
        module.disable_compile_fallback(RuntimeError("Non-finite gradient norm"))
        is False
    )
    assert module.disabled is False


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


def test_energy_only_compile_wrapper_uses_compiled_model_for_stress_virials():
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    class StressVirialsCompiledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(self, batch, **kwargs):
            self.calls.append(dict(kwargs))
            return {
                "energy": batch["x"].sum() * self.weight,
                "forces": torch.zeros(1, 3),
                "virials": torch.ones(1, 3, 3) * self.weight,
                "stress": torch.ones(1, 3, 3) * self.weight,
            }

    eager = _NoCallModel()
    compiled = StressVirialsCompiledModel()
    wrapper = EnergyOnlyForceCompiledModule(
        eager_model=eager,
        compiled_model=compiled,
        allow_fallback=False,
    )

    output = wrapper(
        {"x": torch.ones(2)},
        training=True,
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )
    output["energy"].backward()

    assert compiled.calls == [
        {
            "training": True,
            "compute_force": True,
            "compute_virials": True,
            "compute_stress": True,
        }
    ]
    assert output["stress"] is not None
    assert output["virials"] is not None
    assert compiled.weight.grad is not None


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


def test_take_step_uses_training_compile_wrapper_for_stress_outputs():
    from mace.tools.train import take_step
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    class StressCompiledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(self, batch, **kwargs):
            self.calls.append(dict(kwargs))
            return {
                "value": batch["x"].sum() * self.weight,
                "stress": self.weight.reshape(1, 1, 1).expand(1, 3, 3),
            }

    class StressLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["stress"] is not None
            return pred["value"] + pred["stress"].sum() * 0.0

    compiled = StressCompiledModel()
    model = EnergyOnlyForceCompiledModule(
        eager_model=_NoCallModel(),
        compiled_model=compiled,
        allow_fallback=False,
    )
    optimizer = torch.optim.SGD(compiled.parameters(), lr=0.1)

    loss, _ = take_step(
        model=model,
        loss_fn=StressLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": True},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert compiled.calls == [
        {
            "training": True,
            "compute_force": True,
            "compute_virials": False,
            "compute_stress": True,
        }
    ]
    assert compiled.weight.item() == pytest.approx(0.9)


def test_take_step_uses_training_compile_wrapper_for_virial_outputs():
    from mace.tools.train import take_step
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    class VirialCompiledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.calls = []

        def forward(self, batch, **kwargs):
            self.calls.append(dict(kwargs))
            return {
                "value": batch["x"].sum() * self.weight,
                "virials": self.weight.reshape(1, 1, 1).expand(1, 3, 3),
            }

    class VirialLoss(torch.nn.Module):
        def forward(self, pred, ref):
            assert pred["virials"] is not None
            return pred["value"] + pred["virials"].sum() * 0.0

    compiled = VirialCompiledModel()
    model = EnergyOnlyForceCompiledModule(
        eager_model=_NoCallModel(),
        compiled_model=compiled,
        allow_fallback=False,
    )
    optimizer = torch.optim.SGD(compiled.parameters(), lr=0.1)

    loss, _ = take_step(
        model=model,
        loss_fn=VirialLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": True, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert compiled.calls == [
        {
            "training": True,
            "compute_force": True,
            "compute_virials": True,
            "compute_stress": False,
        }
    ]
    assert compiled.weight.item() == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("loss_name", "loss_fn", "output_args", "expected_flag"),
    [
        (
            "stress",
            modules.WeightedEnergyForcesStressLoss(
                energy_weight=1.0, forces_weight=1.0, stress_weight=1.0
            ),
            {"forces": True, "virials": False, "stress": True},
            "compute_stress",
        ),
        (
            "virials",
            modules.WeightedEnergyForcesVirialsLoss(
                energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
            ),
            {"forces": True, "virials": True, "stress": False},
            "compute_virials",
        ),
    ],
)
def test_take_step_training_compile_supports_builtin_stress_virials_losses(
    loss_name, loss_fn, output_args, expected_flag
):
    del loss_name
    from mace.tools.train import take_step
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    class RecordingCompiledModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.calls = []

        def forward(self, batch, **kwargs):
            self.calls.append(dict(kwargs))
            return self.model(batch, **kwargs)

    base_model = create_tiny_mace("cpu")
    compiled_model = RecordingCompiledModel(base_model)
    model = EnergyOnlyForceCompiledModule(
        eager_model=base_model,
        compiled_model=compiled_model,
        allow_fallback=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-4)
    batch = _BatchDictAdapter(create_batch("cpu"))

    loss, metrics = take_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args=output_args,
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert torch.isfinite(loss)
    assert compiled_model.calls
    assert compiled_model.calls[-1][expected_flag] is True
    assert "edge_force_compile" not in metrics


def test_take_step_keeps_compiled_force_loss_outside_autocast(monkeypatch):
    from mace.tools.precision import TrainingPrecisionConfig
    from mace.tools.train import take_step

    autocast_state = {"enabled": False}

    class FakeAutocast:
        def __enter__(self):
            autocast_state["enabled"] = True

        def __exit__(self, exc_type, exc, tb):
            autocast_state["enabled"] = False

    def fake_autocast(*, device_type, dtype):
        assert device_type == "cuda"
        assert dtype is torch.bfloat16
        return FakeAutocast()

    class RecordingCompiledModel(_CompiledForceLossModel):
        def __init__(self):
            super().__init__()
            self.compiled_autocast_states = []

        def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
            self.compiled_autocast_states.append(autocast_state["enabled"])
            return super().compiled_force_training_loss(
                batch=batch, loss_fn=loss_fn, output_args=output_args
            )

    class RecordingEagerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.forward_autocast_states = []

        def forward(self, batch, **kwargs):
            self.forward_autocast_states.append(autocast_state["enabled"])
            return {"value": batch["x"].sum() * self.weight}

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    precision_config = TrainingPrecisionConfig(enabled=True, dtype=torch.bfloat16)

    compiled_model = RecordingCompiledModel()
    take_step(
        model=compiled_model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=torch.optim.SGD(compiled_model.parameters(), lr=0.1),
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        precision_config=precision_config,
    )

    eager_model = RecordingEagerModel()
    take_step(
        model=eager_model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=torch.optim.SGD(eager_model.parameters(), lr=0.1),
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        precision_config=precision_config,
    )

    assert compiled_model.compiled_autocast_states == [False]
    assert eager_model.forward_autocast_states == [True]


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


def test_take_step_records_compile_disabled_reason_for_stress_outputs():
    from mace.tools.train import take_step

    model = _CompiledForceLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": True},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.forward_calls == 1
    assert model.compiled_loss_calls == 0
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "unsupported_outputs"


def test_take_step_records_compile_disabled_reason_for_virial_outputs():
    from mace.tools.train import take_step

    model = _CompiledForceLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": True, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert model.forward_calls == 1
    assert model.compiled_loss_calls == 0
    assert metrics["edge_force_compile"] is False
    assert metrics["edge_force_compile_disabled_reason"] == "unsupported_outputs"


def test_take_step_does_not_retain_outer_graph_for_compiled_force_loss(monkeypatch):
    from mace.tools.train import take_step

    class CompiledForceLossModel(_CompiledForceLossModel):
        def compiled_force_training_loss(self, *, batch, loss_fn, output_args):
            loss, metrics = super().compiled_force_training_loss(
                batch=batch, loss_fn=loss_fn, output_args=output_args
            )
            return loss, metrics

    backward_retain_graph_values = []
    original_backward = torch.Tensor.backward

    def recording_backward(self, *args, **kwargs):
        backward_retain_graph_values.append(kwargs.get("retain_graph", False))
        return original_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", recording_backward)

    model = CompiledForceLossModel()
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

    assert backward_retain_graph_values == [False]
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


def test_train_one_epoch_logs_edge_force_parity_summary(monkeypatch, caplog):
    import importlib
    import logging

    train_module = importlib.import_module("mace.tools.train")

    def fake_take_step(**kwargs):
        return torch.tensor(0.0), {
            "loss": 0.0,
            "time": 0.25,
            "edge_force_compile": True,
            "edge_force_cache_hit": True,
            "edge_force_parity_check": True,
            "edge_force_parity_accepted": False,
            "edge_force_parity_energy_max_abs_diff": 1.0e-6,
            "edge_force_parity_forces_max_abs_diff": 2.0e-5,
            "edge_force_parity_loss_abs_diff": 3.0e-4,
            "edge_force_parity_param_grad_max_abs_diff": 4.0e-4,
            "edge_force_parity_param_grad_worst": "layer.weight",
            "edge_force_parity_failed_check_names": "forces,param_grad:layer.weight",
            "edge_force_fixed_probe_check": True,
            "edge_force_fixed_probe_accepted": False,
            "edge_force_fixed_probe_num_atoms": 321,
            "edge_force_fixed_probe_num_edges": 6543,
            "edge_force_fixed_probe_energy_max_abs_diff": 5.0e-6,
            "edge_force_fixed_probe_forces_max_abs_diff": 6.0e-5,
            "edge_force_fixed_probe_loss_abs_diff": 7.0e-4,
            "edge_force_fixed_probe_param_grad_max_abs_diff": 8.0e-4,
            "edge_force_fixed_probe_param_grad_worst": "fixed.weight",
            "edge_force_fixed_probe_failed_check_names": "loss,param_grad:fixed.weight",
        }

    class FakeLogger:
        def log(self, metrics):
            pass

    monkeypatch.setattr(train_module, "take_step", fake_take_step)
    caplog.set_level(logging.INFO)

    train_module.train_one_epoch(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        data_loader=[object()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        epoch=7,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=FakeLogger(),
        device=torch.device("cpu"),
        distributed=False,
    )

    summary = "\n".join(record.getMessage() for record in caplog.records)
    assert "Edge-force compile epoch 7 summary" in summary
    assert "parity=checks=1 accepted=0 failed=1" in summary
    assert "max_forces=2.000e-05" in summary
    assert "max_grad=4.000e-04" in summary
    assert "worst_grad=layer.weight" in summary
    assert "fixed_probe=checks=1 accepted=0 failed=1" in summary
    assert "max_forces=6.000e-05" in summary
    assert "max_grad=8.000e-04" in summary
    assert "worst_grad=fixed.weight" in summary
    assert "shape=321x6543" in summary


def test_train_one_epoch_steps_batch_scheduler_only_after_successful_step(monkeypatch):
    import importlib

    train_module = importlib.import_module("mace.tools.train")
    calls = []

    def fake_take_step(**kwargs):
        calls.append(("take_step", kwargs["global_step"]))
        return torch.tensor(0.0), {
            "loss": 0.0,
            "loss_skipped": kwargs["global_step"] == 11,
        }

    class FakeScheduler:
        step_on_batch = True

        def __init__(self):
            self.steps = []

        def step_batch(self, global_step=None):
            self.steps.append(global_step)

        def get_last_lr(self):
            return [0.123]

    class FakeLogger:
        def __init__(self):
            self.records = []

        def log(self, metrics):
            self.records.append(metrics)

    scheduler = FakeScheduler()
    logger = FakeLogger()
    monkeypatch.setattr(train_module, "take_step", fake_take_step)

    train_module.train_one_epoch(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        data_loader=[object(), object()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        epoch=2,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=logger,
        device=torch.device("cpu"),
        distributed=False,
        global_step_start=10,
        lr_scheduler=scheduler,
    )

    assert calls == [("take_step", 10), ("take_step", 11)]
    assert scheduler.steps == [11]
    assert logger.records[0]["lr"] == pytest.approx(0.123)
    assert "lr" not in logger.records[1]


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
