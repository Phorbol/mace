import torch

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
    )

    muon_names = _route_names(summary, "muon")
    adam_names = _route_names(summary, "adam")

    assert "radial_embedding.0.weight" in muon_names
    assert "radial_embedding.2.weight" in muon_names
    assert "readouts.0.weight" in muon_names
    assert "radial_embedding.0.bias" in adam_names
    assert "readouts.0.bias" in adam_names
    assert "scale_shift" in adam_names
    assert "atomic_energies_fn.weight" in adam_names
    assert "products" in adam_names
    assert sum(len(group["params"]) for group in groups) == len(list(model.parameters()))


def test_hybrid_muon_route_summary_is_loggable():
    model = TinyMaceLike()
    _, summary = build_hybrid_muon_param_groups(
        model.named_parameters(),
        lr=1.0e-3,
        weight_decay=1.0e-4,
        muon_weight_decay=0.0,
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
