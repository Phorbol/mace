from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def load_probe():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "probe_edge_vector_force_equivalence.py"
    )
    spec = importlib.util.spec_from_file_location("edge_vector_force_equivalence", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_edge_gradient_to_atomic_forces_matches_position_gradient_for_pair_energy():
    probe = load_probe()
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]], requires_grad=True)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    vector = positions[edge_index[1]] - positions[edge_index[0]]
    energy = 0.5 * vector.square().sum()
    reference_force = -torch.autograd.grad(energy, positions)[0]

    edge_grad = vector.detach().clone()
    edge_force = probe.edge_gradient_to_atomic_forces(
        edge_grad, edge_index=edge_index, num_atoms=positions.shape[0]
    )

    assert torch.allclose(edge_force, reference_force)


def test_compare_snapshots_reports_energy_force_loss_and_parameter_gradients():
    probe = load_probe()
    left = {
        "energy": torch.tensor([1.0]),
        "forces": torch.tensor([[1.0, 0.0, -1.0]]),
        "loss": torch.tensor(2.0),
        "grads": {"weight": torch.tensor([[1.0, 2.0]])},
    }
    right = {
        "energy": torch.tensor([1.0]),
        "forces": torch.tensor([[1.0, 0.0, -1.0]]),
        "loss": torch.tensor(2.0),
        "grads": {"weight": torch.tensor([[1.0, 2.0]])},
    }

    comparison = probe.compare_snapshots(left, right, atol=1.0e-7, rtol=1.0e-6)

    assert comparison["ok"] is True
    assert comparison["failed_checks"] == []
    assert comparison["energy_max_abs_diff"] == 0.0
    assert comparison["forces_max_abs_diff"] == 0.0
    assert comparison["loss_abs_diff"] == 0.0
    assert comparison["param_grad_max_abs_diff"] == {"weight": 0.0}


def test_parse_indices_accepts_lists_and_ranges():
    probe = load_probe()

    assert probe.parse_indices("0,4,8") == [0, 4, 8]
    assert probe.parse_indices("0:5:2,9") == [0, 2, 4, 9]
