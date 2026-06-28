import argparse

import torch

from mace.tools.scripts_utils import get_optimizer

from mace.tools.hybrid_muon import (
    HybridMuon,
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
