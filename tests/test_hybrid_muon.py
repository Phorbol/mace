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

def test_get_optimizer_builds_hybrid_muon():
    model = TinyMaceLike()
    args = argparse.Namespace(
        optimizer="hybrid_muon",
        lr=1.0e-3,
        weight_decay=1.0e-4,
        hybrid_muon_weight_decay=0.0,
        hybrid_muon_lr_factor=0.1,
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
    assert next(group for group in optimizer.param_groups if group["route"] == "muon")["lr"] == 1.0e-4
    assert next(group for group in optimizer.param_groups if group["route"] == "adam")["lr"] == 1.0e-3
