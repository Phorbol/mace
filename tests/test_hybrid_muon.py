import argparse
import math

import pytest
import torch

from mace.tools.scripts_utils import get_optimizer

from mace.tools.hybrid_muon import (
    HybridMuon,
    OptimSpec,
    _orthogonalize_newton_schulz,
    _orthogonalize_newton_schulz_batched,
    build_hybrid_muon_param_groups,
    summarize_hybrid_muon_routes,
)


class TinyMaceLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.radial_embedding = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.SiLU(),
            torch.nn.Linear(8, 8),
        )
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(8, 1)])
        self.scale_shift = torch.nn.Parameter(torch.ones(1))
        self.atomic_energies_fn = torch.nn.Linear(5, 1, bias=False)
        self.products = torch.nn.Parameter(torch.randn(2, 3, 4, 5))

    def forward(self, x):
        return self.readouts[0](self.radial_embedding(x)).sum()


def _route_names(summary, route):
    return {entry["name"] for entry in summary if entry["route"] == route}


def test_hybrid_muon_routes_only_safe_dense_mace_weights():
    model = TinyMaceLike()
    groups, summary = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )

    muon_names = _route_names(summary, "muon")
    adam_names = _route_names(summary, "adam")

    assert "radial_embedding.0.weight" in muon_names
    assert "radial_embedding.2.weight" in muon_names
    assert "readouts.0.weight" in adam_names
    assert "radial_embedding.0.bias" in adam_names
    assert "readouts.0.bias" in adam_names
    assert "scale_shift" in adam_names
    assert "atomic_energies_fn.weight" in adam_names
    assert "products" in adam_names
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))
    muon_group = next(group for group in groups if group["route"] == "muon")
    assert muon_group["lr"] == 1.0e-4
    assert muon_group["hybrid_muon_base_lr"] == 1.0e-3
    assert muon_group["hybrid_muon_lr_factor"] == 0.1
    assert next(group for group in groups if group["route"] == "adam")["lr"] == 1.0e-3


def test_hybrid_muon_keeps_readout_matrices_on_adam_by_default():
    readout_matrix = torch.nn.Parameter(torch.ones(64, 64))

    _, summary = build_hybrid_muon_param_groups(
        [("readouts.0.hidden.weight", readout_matrix)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="mace",
    )

    assert summary == [
        {
            "name": "readouts.0.hidden.weight",
            "shape": (64, 64),
            "numel": 4096,
            "route": "adam",
            "reason": "sensitive-name",
        }
    ]


def test_hybrid_muon_rejects_duplicate_trainable_parameter_routes():
    param = torch.nn.Parameter(torch.ones(4, 4))

    try:
        build_hybrid_muon_param_groups(
            [
                ("radial_embedding.0.weight", param),
                ("radial_embedding.1.weight", param),
            ],
            lr=1.0e-3,
            weight_decay=1.0e-4,
            muon_weight_decay=0.0,
            muon_lr_factor=0.1,
        )
    except ValueError as exc:
        assert "appears more than once" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_muon_rejects_duplicate_parameter_names():
    first = torch.nn.Parameter(torch.ones(4, 4))
    second = torch.nn.Parameter(torch.ones(4, 4))

    try:
        build_hybrid_muon_param_groups(
            [
                ("radial_embedding.0.weight", first),
                ("radial_embedding.0.weight", second),
            ],
            lr=1.0e-3,
            weight_decay=1.0e-4,
            muon_weight_decay=0.0,
            muon_lr_factor=0.1,
        )
    except ValueError as exc:
        assert "Duplicate parameter name" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_muon_routes_singleton_matrix_views_to_adam():
    param = torch.nn.Parameter(torch.ones(1, 128))

    groups, summary = build_hybrid_muon_param_groups(
        [("readouts.0.linear.weight", param)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )

    assert summary == [
        {
            "name": "readouts.0.linear.weight",
            "shape": (1, 128),
            "numel": 128,
            "route": "adam",
            "reason": "sensitive-name",
        }
    ]
    assert len(groups) == 1
    assert groups[0]["route"] == "adam"


def test_hybrid_muon_routes_radial_tp_weight_mlps_to_muon():
    radial_tp_weight = torch.nn.Parameter(torch.ones(64, 256))
    contraction_weight = torch.nn.Parameter(torch.ones(2, 2, 128))

    _, summary = build_hybrid_muon_param_groups(
        [
            ("interactions.0.conv_tp_weights.layer3.weight", radial_tp_weight),
            (
                "products.0.symmetric_contractions.contractions.0.weights.0",
                contraction_weight,
            ),
        ],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert (
        by_name["interactions.0.conv_tp_weights.layer3.weight"]["route"] == "muon"
    )
    assert (
        by_name["interactions.0.conv_tp_weights.layer3.weight"]["reason"]
        == "radial-tp-weight-mlp"
    )
    assert (
        by_name["products.0.symmetric_contractions.contractions.0.weights.0"][
            "route"
        ]
        == "adam"
    )


def test_hybrid_muon_tace_routing_recovers_flattened_e3nn_linear_blocks(monkeypatch):
    from e3nn import o3

    linear = o3.Linear(
        o3.Irreps("4x0e + 4x1o"),
        o3.Irreps("4x0e + 4x1o"),
        internal_weights=True,
        shared_weights=True,
    )
    linear.weight.grad = torch.randn_like(linear.weight)

    groups, summary = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", linear.weight)],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": linear},
    )

    assert summary[0]["route"] == "muon"
    assert summary[0]["reason"] == "tace-e3nn-linear-muon"
    assert summary[0]["matrix_batch"] == 2
    assert summary[0]["matrix_shape"] == (4, 4)

    optimizer = HybridMuon(groups, lr=1.0e-3)
    manifest = optimizer.state_dict()["hybrid_muon_route_manifest"]
    matrix_views = manifest["parameters"]["interactions.0.linear.weight"][
        "matrix_views"
    ]
    assert matrix_views == [
        {
            "kind": "flat_spec",
            "offset": 0,
            "numel": 16,
            "shape": [4, 4],
            "path": [0, 0, 0],
        },
        {
            "kind": "flat_spec",
            "offset": 16,
            "numel": 16,
            "shape": [4, 4],
            "path": [1, 1, 1],
        },
    ]
    calls = []

    def fake_batched(updates, steps=None):
        calls.append(tuple(updates.shape))
        return torch.zeros_like(updates)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_batched,
    )

    optimizer.step()

    assert calls == [(2, 4, 4)]


def test_hybrid_muon_tace_routing_uses_cueq_module_slice_specs():
    class FakeCueqLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(1, 6))
            self.hybrid_muon_optim_specs = {
                "weight": {
                    "route": "muon",
                    "slice_specs": (
                        {
                            "offset": 0,
                            "numel": 6,
                            "matrix_view_shape": (1, 2, 3),
                        },
                    ),
                }
            }

    module = FakeCueqLinear()
    groups, summary = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", module.weight)],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="2d",
        routing="tace",
        module_map={"interactions.0.linear": module},
    )

    assert summary[0]["route"] == "muon"
    assert summary[0]["reason"] == "module-declared"
    assert summary[0]["matrix_batch"] == 1
    assert summary[0]["matrix_shape"] == (2, 3)
    muon_group = next(group for group in groups if group["route"] == "muon")
    assert muon_group["matrix_specs"] == {
        "interactions.0.linear.weight": [
            {
                "offset": 0,
                "numel": 6,
                "matrix_view_shape": (1, 2, 3),
            }
        ]
    }


def test_hybrid_muon_tace_module_include_filters_declared_specs():
    class FakeCueqLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(1, 6))
            self.hybrid_muon_optim_specs = {
                "weight": {
                    "route": "muon",
                    "slice_specs": (
                        {
                            "offset": 0,
                            "numel": 6,
                            "matrix_view_shape": (1, 2, 3),
                        },
                    ),
                }
            }

    interactions_linear = FakeCueqLinear()
    product_linear = FakeCueqLinear()
    groups, summary = build_hybrid_muon_param_groups(
        [
            ("interactions.0.linear.weight", interactions_linear.weight),
            ("products.0.linear.weight", product_linear.weight),
        ],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="2d",
        routing="tace",
        module_map={
            "interactions.0.linear": interactions_linear,
            "products.0.linear": product_linear,
        },
        tace_module_include=("interactions.*",),
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["interactions.0.linear.weight"]["route"] == "muon"
    assert by_name["interactions.0.linear.weight"]["reason"] == "module-declared"
    assert by_name["products.0.linear.weight"]["route"] == "adamw"
    assert by_name["products.0.linear.weight"]["reason"] == "module-spec-filtered"

    muon_group = next(group for group in groups if group["route"] == "muon")
    assert muon_group["param_names"] == ["interactions.0.linear.weight"]
    adam_group = next(group for group in groups if group["route"] == "adam")
    assert adam_group["param_names"] == ["products.0.linear.weight"]


def test_hybrid_muon_tace_module_lr_scale_only_scales_declared_specs():
    class FakeCueqLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(1, 6))
            self.hybrid_muon_optim_specs = {
                "weight": {
                    "route": "muon",
                    "slice_specs": (
                        {
                            "offset": 0,
                            "numel": 6,
                            "matrix_view_shape": (1, 2, 3),
                        },
                    ),
                }
            }

    class FakeDense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 8))

    declared = FakeCueqLinear()
    dense = FakeDense()
    groups, summary = build_hybrid_muon_param_groups(
        [
            ("interactions.0.linear.weight", declared.weight),
            ("interactions.0.conv_tp_weights.layer0.weight", dense.weight),
        ],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="2d",
        routing="tace",
        module_map={
            "interactions.0.linear": declared,
            "interactions.0.conv_tp_weights.layer0": dense,
        },
        tace_module_lr_scale=0.25,
    )

    muon_group = next(group for group in groups if group["route"] == "muon")
    assert muon_group["param_lr_scales"] == {"interactions.0.linear.weight": 0.25}
    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["interactions.0.linear.weight"]["lr_scale"] == 0.25
    assert "lr_scale" not in by_name[
        "interactions.0.conv_tp_weights.layer0.weight"
    ]


def test_hybrid_muon_rejects_nonpositive_tace_module_lr_scale():
    weight = torch.nn.Parameter(torch.ones(2, 2))

    try:
        build_hybrid_muon_param_groups(
            [("interactions.0.linear.weight", weight)],
            lr=1.0e-3,
            weight_decay=0.0,
            muon_weight_decay=0.0,
            muon_lr_factor=0.1,
            routing="tace",
            tace_module_lr_scale=0.0,
        )
    except ValueError as exc:
        assert "tace_module_lr_scale must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_muon_tace_flat_specs_survive_optimizer_resume(monkeypatch):
    class FakeInstruction:
        path_shape = (2, 3)

    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0))
            self.instructions = [FakeInstruction()]

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    first = FakeLinear()
    first.weight.grad = torch.ones_like(first.weight)
    first_groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", first.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": first},
    )
    first_optimizer = HybridMuon(first_groups, lr=1.0)
    first_optimizer.step()

    resumed = FakeLinear()
    resumed_groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", resumed.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": resumed},
    )
    resumed_optimizer = HybridMuon(resumed_groups, lr=1.0)
    resumed_optimizer.load_state_dict(first_optimizer.state_dict())

    before = resumed.weight.detach().clone()
    resumed.weight.grad = torch.ones_like(resumed.weight)
    resumed_optimizer.step()

    assert not torch.allclose(resumed.weight, before)
    state_dict = resumed_optimizer.state_dict()
    muon_group = next(
        group for group in state_dict["param_groups"] if group["route"] == "muon"
    )
    assert "matrix_specs" not in muon_group
    assert muon_group["param_names"] == ["interactions.0.linear.weight"]


def test_hybrid_muon_state_dict_does_not_remove_runtime_matrix_specs(monkeypatch):
    class FakeInstruction:
        path_shape = (2, 3)

    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0))
            self.instructions = [FakeInstruction()]

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    module = FakeLinear()
    groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", module.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": module},
    )
    optimizer = HybridMuon(groups, lr=1.0)

    state_dict = optimizer.state_dict()

    assert "matrix_specs" not in next(
        group for group in state_dict["param_groups"] if group["route"] == "muon"
    )
    runtime_group = next(
        group for group in optimizer.param_groups if group["route"] == "muon"
    )
    assert "matrix_specs" in runtime_group

    before = module.weight.detach().clone()
    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()

    assert not torch.allclose(module.weight, before)


def test_hybrid_muon_module_routing_accepts_ancestor_declarations():
    class BlockDeclaredModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
            self.hybrid_muon_optim_specs = {
                "linear.weight": OptimSpec(route="muon", matrix_axes=(0, 1)),
                "linear.bias": OptimSpec(route="adamw"),
            }

    module = BlockDeclaredModule()
    groups, summary = build_hybrid_muon_param_groups(
        [(f"block.{name}", param) for name, param in module.named_parameters()],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={
            "block" if name == "" else f"block.{name}": submodule
            for name, submodule in module.named_modules()
        },
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["block.linear.weight"]["route"] == "muon"
    assert by_name["block.linear.weight"]["matrix_shape"] == (3, 4)
    assert by_name["block.linear.bias"]["route"] == "adamw"

    muon_group = next(group for group in groups if group["route"] == "muon")
    adam_group = next(group for group in groups if group["route"] == "adam")
    assert set(muon_group["param_matrix_layouts"]) == {"block.linear.weight"}
    assert adam_group["param_names"] == ["block.linear.bias"]


def test_hybrid_muon_optim_spec_rejects_forbidden_semantic_matrix_axes():
    class BadSemanticModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 4, 4))
            self.hybrid_muon_optim_specs = {
                "weight": OptimSpec(
                    route="muon",
                    matrix_axes=(0, 2),
                    batch_axes=(1,),
                    semantic_axes=("degree", "path", "channel_out"),
                )
            }

    module = BadSemanticModule()

    with pytest.raises(ValueError, match="semantic axis 'degree'.*matrix axis"):
        build_hybrid_muon_param_groups(
            [("block.weight", module.weight)],
            lr=1.0e-3,
            weight_decay=1.0e-4,
            muon_weight_decay=0.0,
            muon_lr_factor=0.1,
            routing="module",
            module_map={"block": module},
        )


def test_hybrid_muon_optim_spec_rejects_unsupported_matrix_structures():
    for structure in ("complex", "shared_complex", "diagonal", "scalar_coeff"):
        class StructuredModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8, 8))
                self.hybrid_muon_optim_specs = {
                    "weight": OptimSpec(
                        route="muon",
                        matrix_axes=(0, 1),
                        matrix_structure=structure,
                    )
                }

        module = StructuredModule()

        with pytest.raises(ValueError, match=f"matrix_structure={structure!r}"):
            build_hybrid_muon_param_groups(
                [(f"block_{structure}.weight", module.weight)],
                lr=1.0e-3,
                weight_decay=1.0e-4,
                muon_weight_decay=0.0,
                muon_lr_factor=0.1,
                routing="module",
                module_map={f"block_{structure}": module},
            )


def test_hybrid_muon_optim_spec_matrix_size_gates_fallback_to_adamw():
    class SmallOrSkinnyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.small = torch.nn.Parameter(torch.ones(4, 4))
            self.skinny = torch.nn.Parameter(torch.ones(8, 40))
            self.hybrid_muon_optim_specs = {
                "small": OptimSpec(
                    route="muon",
                    matrix_axes=(0, 1),
                    min_matrix_dim=8,
                ),
                "skinny": OptimSpec(
                    route="muon",
                    matrix_axes=(0, 1),
                    max_aspect_ratio=4.0,
                ),
            }

    module = SmallOrSkinnyModule()
    groups, summary = build_hybrid_muon_param_groups(
        [(f"block.{name}", param) for name, param in module.named_parameters()],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={"block": module},
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["block.small"]["route"] == "adamw"
    assert by_name["block.small"]["reason"] == "module-matrix-dim-gate"
    assert by_name["block.skinny"]["route"] == "adamw"
    assert by_name["block.skinny"]["reason"] == "module-matrix-aspect-gate"
    assert {group["route"] for group in groups} == {"adam"}
    assert next(group for group in groups)["adam_variant"] == "adamw"


def test_hybrid_muon_module_routing_uses_radial_mlp_declarations():
    from mace.modules.radial import RadialMLP

    module = RadialMLP([4, 8, 3])
    groups, summary = build_hybrid_muon_param_groups(
        [(f"radial.{name}", param) for name, param in module.named_parameters()],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={
            f"radial.{name}": submodule
            for name, submodule in module.named_modules()
        },
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["radial.net.0.weight"]["route"] == "muon"
    assert by_name["radial.net.0.weight"]["matrix_shape"] == (8, 4)
    assert by_name["radial.net.3.weight"]["route"] == "muon"
    assert by_name["radial.net.3.weight"]["matrix_shape"] == (3, 8)
    assert by_name["radial.net.0.bias"]["route"] == "adamw"
    assert by_name["radial.net.1.weight"]["route"] == "adamw"
    assert by_name["radial.net.1.bias"]["route"] == "adamw"

    muon_group = next(group for group in groups if group["route"] == "muon")
    adam_group = next(group for group in groups if group["route"] == "adam")
    assert muon_group["param_matrix_layouts"].keys() == {
        "radial.net.0.weight",
        "radial.net.3.weight",
    }
    assert adam_group["adam_variant"] == "adamw"


def test_hybrid_muon_module_routing_defaults_undeclared_params_to_adamw():
    class PartiallyDeclaredModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.muon_linear = torch.nn.Linear(4, 3, bias=False)
            self.adam_linear = torch.nn.Linear(3, 2, bias=False)
            self.hybrid_muon_optim_specs = {
                "muon_linear.weight": OptimSpec(route="muon", matrix_axes=(0, 1)),
            }

    module = PartiallyDeclaredModule()
    groups, summary = build_hybrid_muon_param_groups(
        [(f"block.{name}", param) for name, param in module.named_parameters()],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={
            "block" if name == "" else f"block.{name}": submodule
            for name, submodule in module.named_modules()
        },
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["block.muon_linear.weight"]["route"] == "muon"
    assert by_name["block.muon_linear.weight"]["reason"] == "module-declared"
    assert by_name["block.adam_linear.weight"]["route"] == "adamw"
    assert by_name["block.adam_linear.weight"]["reason"] == "module-default-adamw"

    muon_group = next(group for group in groups if group["route"] == "muon")
    adam_group = next(group for group in groups if group["route"] == "adam")
    assert muon_group["param_names"] == ["block.muon_linear.weight"]
    assert adam_group["param_names"] == ["block.adam_linear.weight"]
    assert adam_group["adam_variant"] == "adamw"




def test_hybrid_muon_module_routing_defaults_cueq_radial_tp_weights_to_muon():
    class CueqRadialLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 64))

    module = CueqRadialLayer()
    groups, summary = build_hybrid_muon_param_groups(
        [("interactions.0.conv_tp_weights.layer0.weight", module.weight)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={"interactions.0.conv_tp_weights.layer0": module},
    )

    assert summary == [
        {
            "name": "interactions.0.conv_tp_weights.layer0.weight",
            "shape": (8, 64),
            "numel": 512,
            "route": "muon",
            "reason": "module-default-radial-tp-weight-mlp",
            "muon_mode": "2d",
            "matrix_shape": (8, 64),
            "matrix_batch": 1,
        }
    ]
    muon_group = next(group for group in groups if group["route"] == "muon")
    assert muon_group["param_names"] == [
        "interactions.0.conv_tp_weights.layer0.weight"
    ]


def test_hybrid_muon_module_routing_uses_declared_slice_specs(monkeypatch):
    class FlatSlicedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(10.0))
            self.hybrid_muon_optim_specs = {
                "weight": OptimSpec(
                    route="muon",
                    slice_specs=(
                        {
                            "offset": 0,
                            "numel": 6,
                            "matrix_view_shape": (1, 2, 3),
                        },
                        {
                            "offset": 6,
                            "numel": 4,
                            "matrix_view_shape": (1, 2, 2),
                        },
                    ),
                )
            }

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )
    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_orthogonalize,
    )

    module = FlatSlicedModule()
    groups, summary = build_hybrid_muon_param_groups(
        [("interactions.0.flat.weight", module.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        routing="module",
        module_map={"interactions.0.flat": module},
    )
    optimizer = HybridMuon(groups, lr=1.0)

    runtime_group = next(
        group for group in optimizer.param_groups if group["route"] == "muon"
    )
    assert runtime_group["matrix_specs"] == {
        "interactions.0.flat.weight": [
            {"offset": 0, "numel": 6, "matrix_view_shape": (1, 2, 3)},
            {"offset": 6, "numel": 4, "matrix_view_shape": (1, 2, 2)},
        ]
    }
    assert "matrix_specs" not in next(
        group for group in optimizer.state_dict()["param_groups"]
        if group["route"] == "muon"
    )
    assert summary == [
        {
            "name": "interactions.0.flat.weight",
            "shape": (10,),
            "numel": 10,
            "route": "muon",
            "reason": "module-declared",
            "muon_mode": "2d",
            "matrix_shape": None,
            "matrix_batch": 2,
        }
    ]

    before = module.weight.detach().clone()
    module.weight.grad = torch.ones_like(module.weight)
    optimizer.step()

    assert torch.allclose(module.weight, before - 1.0)


def test_hybrid_muon_state_dict_contains_route_manifest_hash():
    class DeclaredModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.hybrid_muon_optim_specs = {
                "weight": OptimSpec(route="muon", matrix_axes=(0, 1))
            }

    module = DeclaredModule()
    groups, _ = build_hybrid_muon_param_groups(
        [("block.weight", module.weight)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={"block": module},
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)

    state_dict = optimizer.state_dict()

    assert state_dict["hybrid_muon_route_manifest_hash"]
    manifest = state_dict["hybrid_muon_route_manifest"]
    assert manifest["spec_version"] == 1
    assert manifest["parameters"]["block.weight"]["shape"] == [4, 4]
    assert manifest["parameters"]["block.weight"]["route"] == "muon"
    assert manifest["parameters"]["block.weight"]["reason"] == "module-declared"
    assert manifest["parameters"]["block.weight"]["module_type"].endswith(
        "DeclaredModule"
    )
    assert manifest["parameters"]["block.weight"]["matrix_views"] == [
        {"kind": "layout", "shape": [4, 4]}
    ]


def test_hybrid_muon_mace_route_manifest_snapshot_records_safe_defaults():
    model = TinyMaceLike()
    groups, _ = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="mace",
        module_map=dict(model.named_modules()),
    )
    manifest = HybridMuon(groups, lr=1.0e-3).state_dict()[
        "hybrid_muon_route_manifest"
    ]
    parameters = manifest["parameters"]

    assert parameters["radial_embedding.0.weight"] == {
        "shape": [8, 4],
        "route": "muon",
        "group_route": "muon",
        "reason": "safe-dense-name",
        "module_type": "torch.nn.modules.linear.Linear",
        "matrix_views": [{"kind": "view", "shape": [8, 4]}],
        "structure": "real",
        "muon_mode": "2d",
    }
    assert parameters["readouts.0.weight"]["route"] == "adamw"
    assert parameters["readouts.0.weight"]["reason"] == "sensitive-name"
    assert parameters["products"]["route"] == "adamw"
    assert parameters["products"]["reason"] == "sensitive-name"


def test_hybrid_muon_load_state_rejects_route_manifest_drift():
    class SourceModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.hybrid_muon_optim_specs = {
                "weight": OptimSpec(route="muon", matrix_axes=(0, 1))
            }

    class DriftedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.hybrid_muon_optim_specs = {
                "weight": OptimSpec(route="adamw")
            }

    source = SourceModule()
    source_groups, _ = build_hybrid_muon_param_groups(
        [("block.weight", source.weight)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={"block": source},
    )
    checkpoint = HybridMuon(source_groups, lr=1.0e-3).state_dict()

    drifted = DriftedModule()
    drifted_groups, _ = build_hybrid_muon_param_groups(
        [("block.weight", drifted.weight)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        routing="module",
        module_map={"block": drifted},
    )
    optimizer = HybridMuon(drifted_groups, lr=1.0e-3)

    with pytest.raises(ValueError, match="HybridMuon route manifest hash mismatch"):
        optimizer.load_state_dict(checkpoint)


def test_hybrid_muon_load_state_ignores_serialized_matrix_specs(monkeypatch):
    class FakeInstruction:
        path_shape = (2, 3)

    class FakeLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0))
            self.instructions = [FakeInstruction()]

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    first = FakeLinear()
    first_groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", first.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": first},
    )
    first_optimizer = HybridMuon(first_groups, lr=1.0)
    checkpoint = first_optimizer.state_dict()
    muon_group = next(
        group for group in checkpoint["param_groups"] if group["route"] == "muon"
    )
    muon_group["matrix_specs"] = {
        "interactions.0.linear.weight": [
            {"offset": 0, "numel": 6, "matrix_view_shape": (1, 1, 6)}
        ]
    }

    resumed = FakeLinear()
    resumed_groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.linear.weight", resumed.weight)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.linear": resumed},
    )
    resumed_optimizer = HybridMuon(resumed_groups, lr=1.0)
    resumed_optimizer.load_state_dict(checkpoint)

    runtime_group = next(
        group for group in resumed_optimizer.param_groups if group["route"] == "muon"
    )
    assert runtime_group["matrix_specs"]["interactions.0.linear.weight"][0][
        "matrix_view_shape"
    ] == (1, 2, 3)

    before = resumed.weight.detach().clone()
    resumed.weight.grad = torch.ones_like(resumed.weight)
    resumed_optimizer.step()

    assert not torch.allclose(resumed.weight, before)


def test_hybrid_muon_routed_param_without_matrix_view_raises():
    param = torch.nn.Parameter(torch.ones(4))
    param.grad = torch.ones_like(param)
    optimizer = HybridMuon(
        [
            {
                "params": [param],
                "route": "muon",
                "lr": 1.0,
                "weight_decay": 0.0,
                "beta": 0.9,
                "muon_mode": "slice",
                "matrix_specs": {},
                "param_names": ["flat.weight"],
            }
        ],
        lr=1.0,
    )

    try:
        optimizer.step()
    except RuntimeError as exc:
        assert "flat.weight" in str(exc)
        assert "no valid MatrixSpec or matrix view" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_hybrid_muon_tace_routing_recovers_skip_tp_species_blocks(monkeypatch):
    from e3nn import o3

    skip_tp = o3.FullyConnectedTensorProduct(
        o3.Irreps("4x0e"),
        o3.Irreps("3x0e"),
        o3.Irreps("4x0e"),
        internal_weights=True,
        shared_weights=True,
    )
    skip_tp.weight.grad = torch.randn_like(skip_tp.weight)

    groups, summary = build_hybrid_muon_param_groups(
        [("interactions.0.skip_tp.weight", skip_tp.weight)],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="tace",
        module_map={"interactions.0.skip_tp": skip_tp},
    )

    assert summary[0]["route"] == "muon"
    assert summary[0]["reason"] == "tace-e3nn-skip-tp-species-muon"
    assert summary[0]["matrix_batch"] == 3
    assert summary[0]["matrix_shape"] == (4, 4)

    optimizer = HybridMuon(groups, lr=1.0e-3)
    calls = []

    def fake_batched(updates, steps=None):
        calls.append(tuple(updates.shape))
        return torch.zeros_like(updates)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_batched,
    )

    optimizer.step()

    assert calls == [(3, 4, 4)]


def test_hybrid_muon_tace_routing_defaults_unknown_matrices_to_adamw():
    ambiguous_matrix = torch.nn.Parameter(torch.ones(32, 64))
    equivariant_tensor = torch.nn.Parameter(torch.ones(3, 16, 64))
    embedding_matrix = torch.nn.Parameter(torch.ones(5, 64))
    atomic_head = torch.nn.Parameter(torch.ones(5, 1))
    affine_matrix = torch.nn.Parameter(torch.ones(64, 64))

    _, summary = build_hybrid_muon_param_groups(
        [
            ("interactions.0.linear_up.weight", ambiguous_matrix),
            ("interactions.0.skip_tp.weight", equivariant_tensor),
            ("node_embedding.linear.weight", embedding_matrix),
            ("atomic_energies_fn.weight", atomic_head),
            ("interactions.0.affine.weight", affine_matrix),
        ],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="tace",
    )

    by_name = {entry["name"]: entry for entry in summary}
    assert by_name["interactions.0.linear_up.weight"]["route"] == "adamw"
    assert by_name["interactions.0.linear_up.weight"]["reason"] == "tace-unknown-adamw"
    assert by_name["interactions.0.skip_tp.weight"]["route"] == "adamw"
    assert by_name["interactions.0.skip_tp.weight"]["reason"] == "tace-unknown-adamw"
    assert by_name["node_embedding.linear.weight"]["route"] == "adam"
    assert by_name["node_embedding.linear.weight"]["reason"] == "mace-sensitive-name"
    assert by_name["atomic_energies_fn.weight"]["route"] == "adam"
    assert by_name["atomic_energies_fn.weight"]["reason"] == "mace-sensitive-name"
    assert by_name["interactions.0.affine.weight"]["route"] == "adam"
    assert by_name["interactions.0.affine.weight"]["reason"] == "mace-sensitive-name"


def test_hybrid_muon_mace_slice_keeps_symmetric_contractions_on_adam():
    contraction_weight = torch.nn.Parameter(torch.ones(2, 4, 128))

    groups, summary = build_hybrid_muon_param_groups(
        [
            (
                "products.0.symmetric_contractions.contractions.0.weights.0",
                contraction_weight,
            ),
        ],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="mace",
    )

    assert summary == [
        {
            "name": "products.0.symmetric_contractions.contractions.0.weights.0",
            "shape": (2, 4, 128),
            "numel": 1024,
            "route": "adam",
            "reason": "sensitive-name",
        }
    ]
    assert len(groups) == 1
    assert groups[0]["route"] == "adam"


def test_hybrid_muon_tace_keeps_undeclared_rank3_tensors_on_adamw(monkeypatch):
    param = torch.nn.Parameter(torch.randn(2, 3, 4))
    param.grad = torch.randn_like(param)
    groups, summary = build_hybrid_muon_param_groups(
        [("products.0.symmetric_contractions.contractions.0.weights.0", param)],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        muon_mode="slice",
        routing="tace",
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)
    calls = []

    def fake_batched(updates, steps=5):
        calls.append(tuple(updates.shape))
        return torch.zeros_like(updates)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_batched,
    )

    optimizer.step()

    assert summary[0]["route"] == "adamw"
    assert summary[0]["reason"] == "tace-unknown-adamw"
    assert calls == []


def test_newton_schulz_two_stage_produces_tight_polar_factor():
    torch.manual_seed(71)
    update = torch.randn(16, 16)

    orthogonalized = _orthogonalize_newton_schulz(update)

    singular_values = torch.linalg.svdvals(orthogonalized.float())
    assert torch.allclose(
        singular_values, torch.ones_like(singular_values), atol=2.0e-3, rtol=0.0
    )
    assert orthogonalized.dtype == update.dtype


def test_batched_newton_schulz_matches_per_tensor_helper():
    torch.manual_seed(37)
    for shape in ((4, 64, 64), (3, 8, 64), (2, 80, 8)):
        updates = torch.randn(*shape)

        batched = _orthogonalize_newton_schulz_batched(updates)
        looped = torch.stack(
            [_orthogonalize_newton_schulz(update) for update in updates]
        )

        assert torch.allclose(batched, looped, atol=1.0e-6, rtol=1.0e-6)


def test_hybrid_muon_muon_lr_scale_modes(monkeypatch):
    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    cases = (
        ("original", 0.18, math.sqrt(4.0)),
        ("none", 0.18, 1.0),
        ("match_rms", 0.25, 0.25 * math.sqrt(8.0)),
    )
    for mode, coeff, expected_scale in cases:
        param = torch.nn.Parameter(torch.zeros(8, 2))
        param.grad = torch.ones_like(param)
        groups, _ = build_hybrid_muon_param_groups(
            [("interactions.0.conv_tp_weights.layer0.weight", param)],
            lr=1.0,
            weight_decay=0.0,
            muon_weight_decay=0.0,
            muon_lr_factor=1.0,
            muon_lr_scale_mode=mode,
            muon_match_rms_coeff=coeff,
        )
        optimizer = HybridMuon(groups, lr=1.0)

        optimizer.step()

        assert torch.allclose(param, torch.full_like(param, -expected_scale))
        muon_group = next(
            group for group in optimizer.param_groups if group["route"] == "muon"
        )
        assert muon_group["muon_lr_scale_mode"] == mode
        assert muon_group["muon_match_rms_coeff"] == coeff


def test_hybrid_muon_rejects_invalid_muon_lr_scale_options():
    param = torch.nn.Parameter(torch.zeros(8, 2))

    try:
        build_hybrid_muon_param_groups(
            [("interactions.0.conv_tp_weights.layer0.weight", param)],
            lr=1.0,
            weight_decay=0.0,
            muon_weight_decay=0.0,
            muon_lr_factor=1.0,
            muon_lr_scale_mode="bad",
        )
    except ValueError as exc:
        assert "hybrid_muon_lr_scale_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        build_hybrid_muon_param_groups(
            [("interactions.0.conv_tp_weights.layer0.weight", param)],
            lr=1.0,
            weight_decay=0.0,
            muon_weight_decay=0.0,
            muon_lr_factor=1.0,
            muon_match_rms_coeff=0.0,
        )
    except ValueError as exc:
        assert "muon_match_rms_coeff must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_muon_batches_same_shape_muon_updates(monkeypatch):
    params = [torch.nn.Parameter(torch.randn(4, 8)) for _ in range(3)]
    for param in params:
        param.grad = torch.randn_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [
            (f"interactions.0.conv_tp_weights.layer{i}.weight", param)
            for i, param in enumerate(params)
        ],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)
    calls = []

    def fake_batched(updates, steps=5):
        calls.append(tuple(updates.shape))
        return torch.zeros_like(updates)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_batched,
    )

    optimizer.step()

    assert calls == [(3, 4, 8)]


def test_hybrid_muon_batches_same_short_side_with_column_padding(monkeypatch):
    wide = torch.nn.Parameter(torch.randn(2, 4))
    tall = torch.nn.Parameter(torch.randn(6, 2))
    wide.grad = torch.randn_like(wide)
    tall.grad = torch.randn_like(tall)
    groups, _ = build_hybrid_muon_param_groups(
        [
            ("interactions.0.conv_tp_weights.wide.weight", wide),
            ("interactions.0.conv_tp_weights.tall.weight", tall),
        ],
        lr=1.0e-3,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)
    calls = []

    def fake_batched(updates, steps=None):
        calls.append(tuple(updates.shape))
        return torch.zeros_like(updates)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz_batched",
        fake_batched,
    )

    optimizer.step()

    assert calls == [(2, 2, 6)]


def test_hybrid_muon_keeps_momentum_buffer_separate_from_nesterov_update(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(4, 4))
    param.grad = torch.ones_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.conv_tp_weights.layer0.weight", param)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        beta=0.9,
    )
    optimizer = HybridMuon(groups, lr=1.0)

    def fake_orthogonalize(update, steps=None):
        return torch.zeros_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    optimizer.step()

    momentum = optimizer.state[param]["momentum"]
    assert torch.allclose(momentum, torch.full_like(param, 0.1))


def test_hybrid_muon_magma_lite_damps_misaligned_muon_update(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, 2))
    param.grad = torch.ones_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.conv_tp_weights.layer0.weight", param)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        beta=0.9,
        magma_lite=True,
    )
    optimizer = HybridMuon(groups, lr=1.0)
    optimizer.state[param]["momentum"] = -torch.ones_like(param)

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    optimizer.step()

    magma_score = optimizer.state[param]["magma_score"]
    assert magma_score.shape == (1,)
    assert magma_score.item() < 0.5
    expected_scale = 0.1 + 0.9 * magma_score.item()
    assert torch.allclose(param, torch.full_like(param, -expected_scale))

def test_hybrid_muon_magma_lite_initial_score_is_configurable(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, 2))
    param.grad = torch.ones_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.conv_tp_weights.layer0.weight", param)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        beta=0.9,
        magma_lite=True,
        magma_initial_score=0.0,
    )
    optimizer = HybridMuon(groups, lr=1.0)

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    optimizer.step()

    expected_score = 0.1
    expected_scale = 0.1 + 0.9 * expected_score
    assert torch.allclose(optimizer.state[param]["magma_score"], torch.tensor([expected_score]))
    assert torch.allclose(param, torch.full_like(param, -expected_scale))


def test_hybrid_muon_magma_lite_can_bypass_first_step(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2, 2))
    param.grad = torch.ones_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [("interactions.0.conv_tp_weights.layer0.weight", param)],
        lr=1.0,
        weight_decay=0.0,
        muon_weight_decay=0.0,
        muon_lr_factor=1.0,
        beta=0.9,
        magma_lite=True,
        magma_bypass_first_step=True,
    )
    optimizer = HybridMuon(groups, lr=1.0)

    def fake_orthogonalize(update, steps=None):
        return torch.ones_like(update)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon._orthogonalize_newton_schulz",
        fake_orthogonalize,
    )

    optimizer.step()

    assert "magma_score" in optimizer.state[param]
    assert torch.allclose(param, torch.full_like(param, -1.0))


def test_hybrid_muon_adam_route_uses_torch_functional_adam(monkeypatch):
    param = torch.nn.Parameter(torch.ones(1, 8))
    param.grad = torch.ones_like(param)
    calls = []

    def fake_adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, **kwargs
    ):
        calls.append(
            {
                "params": params,
                "grads": grads,
                "exp_avgs": exp_avgs,
                "exp_avg_sqs": exp_avg_sqs,
                "max_exp_avg_sqs": max_exp_avg_sqs,
                "state_steps": state_steps,
                "kwargs": kwargs,
            }
        )

    monkeypatch.setattr(
        "mace.tools.hybrid_muon.optim_functional.adam",
        fake_adam,
    )
    groups, _ = build_hybrid_muon_param_groups(
        [("readouts.0.linear.weight", param)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        adam_betas=(0.8, 0.97),
        eps=1.0e-7,
        amsgrad=True,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)

    optimizer.step()

    assert len(calls) == 1
    call = calls[0]
    assert call["params"] == [param]
    assert call["kwargs"]["foreach"] is True
    assert call["kwargs"]["amsgrad"] is True
    assert call["kwargs"]["beta1"] == 0.8
    assert call["kwargs"]["beta2"] == 0.97
    assert call["kwargs"]["lr"] == 1.0e-3
    assert call["kwargs"]["weight_decay"] == 1.0e-4
    assert call["kwargs"]["decoupled_weight_decay"] is True
    assert call["kwargs"]["eps"] == 1.0e-7


def test_hybrid_muon_adam_variant_can_use_coupled_adam(monkeypatch):
    param = torch.nn.Parameter(torch.ones(1, 8))
    param.grad = torch.ones_like(param)
    calls = []

    def fake_adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, **kwargs
    ):
        calls.append(kwargs)

    monkeypatch.setattr(
        "mace.tools.hybrid_muon.optim_functional.adam",
        fake_adam,
    )
    groups, _ = build_hybrid_muon_param_groups(
        [("readouts.0.linear.weight", param)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        adam_variant="adam",
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)

    optimizer.step()

    assert len(calls) == 1
    assert calls[0]["decoupled_weight_decay"] is False
    adam_group = next(
        group for group in optimizer.param_groups if group["route"] == "adam"
    )
    assert adam_group["adam_variant"] == "adam"


def test_hybrid_muon_adam_route_initializes_state_after_muon_route_switch():
    param = torch.nn.Parameter(torch.eye(8))
    groups = [
        {
            "params": [param],
            "route": "muon",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
            "beta": 0.9,
            "muon_mode": "2d",
        }
    ]
    optimizer = HybridMuon(groups, lr=1.0e-3)

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert "momentum" in optimizer.state[param]
    assert "step" not in optimizer.state[param]

    optimizer.param_groups[0].update(
        {
            "route": "adam",
            "adam_variant": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
        }
    )
    param.grad = torch.full_like(param, 0.5)

    optimizer.step()

    assert optimizer.state[param]["step"].item() == 1.0
    assert optimizer.state[param]["exp_avg"].shape == param.shape
    assert optimizer.state[param]["exp_avg_sq"].shape == param.shape


def test_hybrid_muon_adam_route_converts_legacy_integer_step_for_foreach():
    param = torch.nn.Parameter(torch.ones(1, 8))
    param.grad = torch.ones_like(param)
    groups, _ = build_hybrid_muon_param_groups(
        [("readouts.0.linear.weight", param)],
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)
    optimizer.state[param]["step"] = torch.tensor(3, device=param.device)
    optimizer.state[param]["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
    optimizer.state[param]["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

    optimizer.step()

    assert optimizer.state[param]["step"].device.type == "cpu"
    assert optimizer.state[param]["step"].dtype == torch.float32
    assert optimizer.state[param]["step"].item() == 4.0


def test_hybrid_muon_adam_route_matches_torch_adamw_with_amsgrad():
    torch.manual_seed(12)
    initial = torch.randn(1, 8)
    hybrid_param = torch.nn.Parameter(initial.clone())
    torch_param = torch.nn.Parameter(initial.clone())
    lr = 3.0e-3
    weight_decay = 2.0e-2
    betas = (0.8, 0.97)
    eps = 1.0e-7

    groups, summary = build_hybrid_muon_param_groups(
        [("readouts.0.linear.weight", hybrid_param)],
        lr=lr,
        weight_decay=weight_decay,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
        adam_betas=betas,
        eps=eps,
        amsgrad=True,
    )
    hybrid_optimizer = HybridMuon(groups, lr=lr)
    torch_optimizer = torch.optim.AdamW(
        [torch_param],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=True,
    )

    assert summary[0]["route"] == "adam"
    for _ in range(5):
        grad = torch.randn_like(initial)
        hybrid_param.grad = grad.clone()
        torch_param.grad = grad.clone()
        hybrid_optimizer.step()
        torch_optimizer.step()

    assert torch.allclose(hybrid_param, torch_param, atol=1.0e-7, rtol=1.0e-6)


def test_hybrid_muon_route_summary_is_loggable():
    model = TinyMaceLike()
    _, summary = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )

    text = summarize_hybrid_muon_routes(summary)

    assert "HybridMuon parameter routing" in text
    assert "Muon tensors:" in text
    assert "Adam tensors:" in text
    assert "radial_embedding.0.weight" in text


def test_hybrid_muon_route_summary_logs_per_parameter_lr_scale():
    text = summarize_hybrid_muon_routes(
        [
            {
                "name": "interactions.0.linear.weight",
                "shape": (1, 4096),
                "numel": 4096,
                "route": "muon",
                "reason": "module-declared",
                "muon_mode": "2d",
                "matrix_batch": 1,
                "matrix_shape": (64, 64),
                "lr_scale": 0.25,
            }
        ]
    )

    assert "lr_scale=0.25" in text


def test_hybrid_muon_step_updates_params_and_state_dict_reloads():
    torch.manual_seed(5)
    model = TinyMaceLike()
    groups, _ = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
        muon_lr_factor=0.1,
    )
    optimizer = HybridMuon(groups, lr=1.0e-3)

    before = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    loss = model(torch.randn(3, 4))
    loss.backward()
    optimizer.step()

    assert any(
        not torch.allclose(before[name], param)
        for name, param in model.named_parameters()
        if param.requires_grad
    )

    reloaded = HybridMuon(groups, lr=1.0e-3)
    reloaded.load_state_dict(optimizer.state_dict())
    assert reloaded.state_dict()["state"]


def test_hybrid_muon_rejects_nonpositive_lr_factor():
    model = TinyMaceLike()

    try:
        build_hybrid_muon_param_groups(
            model.named_parameters(),
            lr=1.0e-3,
            weight_decay=1.0e-4,
            muon_weight_decay=0.0,
            muon_lr_factor=0.0,
        )
    except ValueError as exc:
        assert "muon_lr_factor must be positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

def test_arg_parser_accepts_hybrid_muon_magma_lite_flag():
    from mace.tools import build_default_arg_parser

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "muon_magma_parser_test",
            "--optimizer",
            "hybrid_muon",
            "--hybrid_muon_magma_lite",
            "--hybrid_muon_adam_variant",
            "adam",
            "--hybrid_muon_lr_scale_mode",
            "match_rms",
            "--hybrid_muon_match_rms_coeff",
            "0.25",
            "--hybrid_muon_magma_initial_score",
            "0.25",
            "--hybrid_muon_magma_warmup_steps",
            "3",
            "--hybrid_muon_magma_bypass_first_step",
            "--hybrid_muon_stage_two_lr_factor",
            "0.25",
            "--hybrid_muon_stage_two_route",
            "adamw",
            "--hybrid_muon_tace_module_include",
            "interactions.*,products.*.linear.weight",
            "--hybrid_muon_tace_module_lr_scale",
            "0.25",
        ]
    )

    assert args.optimizer == "hybrid_muon"
    assert args.hybrid_muon_magma_lite is True
    assert args.hybrid_muon_adam_variant == "adam"
    assert args.hybrid_muon_lr_scale_mode == "match_rms"
    assert args.hybrid_muon_match_rms_coeff == 0.25
    assert args.hybrid_muon_magma_initial_score == 0.25
    assert args.hybrid_muon_magma_warmup_steps == 3
    assert args.hybrid_muon_magma_bypass_first_step is True
    assert args.hybrid_muon_stage_two_lr_factor == 0.25
    assert args.hybrid_muon_stage_two_route == "adamw"
    assert (
        args.hybrid_muon_tace_module_include
        == "interactions.*,products.*.linear.weight"
    )
    assert args.hybrid_muon_tace_module_lr_scale == 0.25


def test_get_optimizer_builds_hybrid_muon():
    model = TinyMaceLike()
    args = argparse.Namespace(
        optimizer="hybrid_muon",
        lr=1.0e-3,
        weight_decay=1.0e-4,
        hybrid_muon_weight_decay=0.0,
        hybrid_muon_lr_factor=0.1,
        hybrid_muon_mode="slice",
        hybrid_muon_routing="mace",
        hybrid_muon_magma_lite=True,
        beta=0.9,
        amsgrad=False,
    )
    param_options = {
        "params": [{"name": "all", "params": list(model.parameters()), "lr": args.lr}],
        "lr": args.lr,
        "amsgrad": args.amsgrad,
        "betas": (args.beta, 0.999),
    }

    optimizer = get_optimizer(
        args, param_options, named_parameters=model.named_parameters()
    )

    assert optimizer.__class__.__name__ == "HybridMuon"
    assert {group["route"] for group in optimizer.param_groups} == {"muon", "adam"}
    muon_group = next(group for group in optimizer.param_groups if group["route"] == "muon")
    assert muon_group["lr"] == 1.0e-4
    assert muon_group["muon_mode"] == "slice"
    assert muon_group["magma_lite"] is True
    assert next(group for group in optimizer.param_groups if group["route"] == "adam")["lr"] == 1.0e-3


def test_get_optimizer_preserves_mace_adam_fallback_weight_decay_groups():
    model = TinyMaceLike()
    args = argparse.Namespace(
        optimizer="hybrid_muon",
        lr=1.0e-3,
        weight_decay=1.0e-4,
        hybrid_muon_weight_decay=0.0,
        hybrid_muon_lr_factor=0.1,
        hybrid_muon_mode="slice",
        hybrid_muon_routing="mace",
        hybrid_muon_magma_lite=False,
        beta=0.9,
        amsgrad=False,
    )
    readout_weight = model.readouts[0].weight
    product_weight = model.products
    radial_weight = model.radial_embedding[0].weight
    param_options = {
        "params": [
            {
                "name": "readouts",
                "params": [readout_weight],
                "weight_decay": 0.0,
                "lr": args.lr,
            },
            {
                "name": "products",
                "params": [product_weight],
                "weight_decay": args.weight_decay,
                "lr": args.lr,
            },
            {
                "name": "radial",
                "params": [radial_weight],
                "weight_decay": args.weight_decay,
                "lr": args.lr,
            },
        ],
        "lr": args.lr,
        "amsgrad": args.amsgrad,
        "betas": (args.beta, 0.999),
    }

    optimizer = get_optimizer(
        args,
        param_options,
        named_parameters=[
            ("readouts.0.weight", readout_weight),
            ("products", product_weight),
            ("radial_embedding.0.weight", radial_weight),
        ],
    )

    adam_groups = [group for group in optimizer.param_groups if group["route"] == "adam"]
    no_decay_group = next(
        group
        for group in adam_groups
        if any(param is readout_weight for param in group["params"])
    )
    decay_group = next(
        group
        for group in adam_groups
        if any(param is product_weight for param in group["params"])
    )
    muon_group = next(group for group in optimizer.param_groups if group["route"] == "muon")

    assert no_decay_group["weight_decay"] == 0.0
    assert decay_group["weight_decay"] == args.weight_decay
    assert any(param is radial_weight for param in muon_group["params"])
