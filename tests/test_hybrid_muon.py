import argparse

import torch

from mace.tools.scripts_utils import get_optimizer

from mace.tools.hybrid_muon import (
    HybridMuon,
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
    assert next(group for group in groups if group["route"] == "muon")["lr"] == 1.0e-4
    assert next(group for group in groups if group["route"] == "adam")["lr"] == 1.0e-3


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
            "reason": "effective-rank<2",
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


def test_hybrid_muon_tace_routing_routes_eligible_mace_matrices_broadly():
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
    assert by_name["interactions.0.linear_up.weight"]["route"] == "muon"
    assert by_name["interactions.0.linear_up.weight"]["reason"] == "tace-matrix-muon"
    assert by_name["interactions.0.linear_up.weight"]["matrix_shape"] == (32, 64)
    assert by_name["interactions.0.skip_tp.weight"]["route"] == "muon"
    assert by_name["interactions.0.skip_tp.weight"]["reason"] == "tace-matrix-muon"
    assert by_name["interactions.0.skip_tp.weight"]["matrix_batch"] == 3
    assert by_name["interactions.0.skip_tp.weight"]["matrix_shape"] == (16, 64)
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


def test_hybrid_muon_tace_slice_step_updates_rank3_slices_without_flattening(monkeypatch):
    param = torch.nn.Parameter(torch.randn(2, 3, 4))
    param.grad = torch.randn_like(param)
    groups, _ = build_hybrid_muon_param_groups(
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

    assert calls == [(2, 3, 4)]


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
    assert call["kwargs"]["eps"] == 1.0e-7


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


def test_hybrid_muon_adam_route_matches_torch_adam_with_amsgrad():
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
    torch_optimizer = torch.optim.Adam(
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
        ]
    )

    assert args.optimizer == "hybrid_muon"
    assert args.hybrid_muon_magma_lite is True


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
