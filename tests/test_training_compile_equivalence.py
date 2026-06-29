from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_probe():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "probe_training_compile.py"
    )
    spec = importlib.util.spec_from_file_location("training_compile_probe", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_force_loss_equivalence_reports_energy_forces_loss_and_gradients():
    probe = load_probe()
    torch.manual_seed(123)
    device = torch.device("cpu")
    batch, z_table = probe._load_batch(
        Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"),
        [0],
        cutoff=5.0,
        device=device,
    )
    model = probe._create_model(
        z_table=z_table,
        cutoff=5.0,
        device=device,
        hidden_channels=8,
        max_ell=1,
        num_interactions=1,
        correlation=1,
        enable_cueq=False,
    )
    candidate = copy.deepcopy(model)

    result = probe.compare_force_loss_equivalence(
        model,
        candidate,
        batch,
        atol=1.0e-6,
        rtol=1.0e-5,
    )

    assert result["ok"] is True
    assert result["energy_max_abs_diff"] <= 1.0e-6
    assert result["forces_max_abs_diff"] <= 1.0e-6
    assert result["loss_abs_diff"] <= 1.0e-6
    assert result["param_grad_max_abs_diff"]
    assert all(diff <= 1.0e-6 for diff in result["param_grad_max_abs_diff"].values())
    assert result["failed_checks"] == []


def test_force_loss_equivalence_reports_mismatched_gradients():
    probe = load_probe()
    torch.manual_seed(456)
    device = torch.device("cpu")
    batch, z_table = probe._load_batch(
        Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"),
        [0],
        cutoff=5.0,
        device=device,
    )
    model = probe._create_model(
        z_table=z_table,
        cutoff=5.0,
        device=device,
        hidden_channels=8,
        max_ell=1,
        num_interactions=1,
        correlation=1,
        enable_cueq=False,
    )
    candidate = copy.deepcopy(model)
    with torch.no_grad():
        candidate.readouts[0].linear.weight.add_(0.1)

    result = probe.compare_force_loss_equivalence(
        model,
        candidate,
        batch,
        atol=1.0e-8,
        rtol=1.0e-8,
    )

    assert result["ok"] is False
    assert result["failed_checks"]

def test_force_loss_equivalence_matches_compiled_wrapper_parameter_names(monkeypatch):
    probe = load_probe()
    torch.manual_seed(789)
    device = torch.device("cpu")
    batch, z_table = probe._load_batch(
        Path("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz"),
        [0],
        cutoff=5.0,
        device=device,
    )
    model = probe._create_model(
        z_table=z_table,
        cutoff=5.0,
        device=device,
        hidden_channels=8,
        max_ell=1,
        num_interactions=1,
        correlation=1,
        enable_cueq=False,
    )

    class FakeCompiled(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self._orig_mod = wrapped

        def forward(self, *args, **kwargs):
            return self._orig_mod(*args, **kwargs)

    monkeypatch.setattr(
        probe.torch,
        "compile",
        lambda module, *, mode, fullgraph: FakeCompiled(module),
    )
    candidate = probe.build_equivalence_candidate_model(
        model,
        candidate="compile_readouts",
        compile_mode="default",
        compile_fullgraph=False,
    )

    result = probe.compare_force_loss_equivalence(
        model,
        candidate,
        batch,
        atol=1.0e-6,
        rtol=1.0e-5,
    )

    assert result["ok"] is True
    assert "readouts.0.linear.weight" in result["param_grad_max_abs_diff"]
    assert not any("_orig_mod" in name for name in result["param_grad_max_abs_diff"])
