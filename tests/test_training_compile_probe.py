from __future__ import annotations

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


def test_parse_indices_accepts_lists_and_ranges():
    probe = load_probe()

    assert probe.parse_indices("0,4,8") == [0, 4, 8]
    assert probe.parse_indices("0:8:4,10") == [0, 4, 10]


@pytest.mark.parametrize("value", ["", "0:", "0:1:2:3"])
def test_parse_indices_rejects_invalid_values(value):
    probe = load_probe()

    with pytest.raises(ValueError):
        probe.parse_indices(value)


def test_probe_modes_keep_energy_only_out_of_force_graph():
    probe = load_probe()

    modes = {mode.name: mode for mode in probe.get_probe_modes(None)}

    assert modes["compile_energy_only"].compiled is True
    assert modes["compile_energy_only"].compute_force is False
    assert modes["compile_energy_only"].loss_kind == "energy"
    assert modes["compile_force_loss"].compute_force is True
    assert modes["compile_force_loss"].loss_kind == "energy_forces"


def test_get_probe_modes_rejects_unknown_mode():
    probe = load_probe()

    with pytest.raises(ValueError, match="unknown probe mode"):
        probe.get_probe_modes(["missing"])


def test_run_probe_includes_force_loss_equivalence_when_enabled(monkeypatch):
    import argparse

    probe = load_probe()

    class FakeBatch:
        num_nodes = 3

    class FakeZTable:
        zs = [1]

    monkeypatch.setattr(
        probe,
        "_load_batch",
        lambda xyz, indices, cutoff, device: (FakeBatch(), FakeZTable()),
    )
    monkeypatch.setattr(
        probe,
        "_create_model",
        lambda **kwargs: probe.torch.nn.Linear(1, 1),
    )
    monkeypatch.setattr(
        probe,
        "compare_force_loss_equivalence",
        lambda *args, **kwargs: {"ok": True, "failed_checks": []},
    )

    payload = probe.run_probe(
        argparse.Namespace(
            device="cpu",
            dtype="float32",
            seed=123,
            indices="0",
            xyz="train.xyz",
            cutoff=5.0,
            hidden_channels=8,
            max_ell=1,
            num_interactions=1,
            correlation=1,
            enable_cueq=False,
            modes=[],
            compile_mode="default",
            compile_fullgraph=False,
            allow_fallback=True,
            warmup=0,
            repeats=1,
            equivalence_gate=True,
            equivalence_atol=1.0e-6,
            equivalence_rtol=1.0e-5,
        )
    )

    assert payload["force_loss_equivalence"]["ok"] is True
    assert payload["force_loss_equivalence"]["failed_checks"] == []
    assert payload["force_loss_equivalence"]["candidate"] == "eager_copy"
    assert payload["force_loss_equivalence"]["status"] == "ok"

def test_build_equivalence_candidate_compiles_readouts(monkeypatch):
    probe = load_probe()
    compiled = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.readouts = torch.nn.ModuleList([torch.nn.Linear(2, 1)])

    class FakeCompiled(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    def fake_compile(module, *, mode, fullgraph):
        compiled.append((module, mode, fullgraph))
        return FakeCompiled(module)

    monkeypatch.setattr(probe.torch, "compile", fake_compile)
    model = Tiny()

    candidate = probe.build_equivalence_candidate_model(
        model,
        candidate="compile_readouts",
        compile_mode="default",
        compile_fullgraph=False,
    )

    assert candidate is not model
    assert isinstance(candidate.readouts[0], FakeCompiled)
    assert len(compiled) == 1
    assert compiled[0][1:] == ("default", False)
    assert isinstance(model.readouts[0], torch.nn.Linear)


def test_run_probe_uses_selected_equivalence_candidate(monkeypatch):
    import argparse

    probe = load_probe()
    marker_candidate = object()
    seen = {}

    class FakeBatch:
        num_nodes = 3

    class FakeZTable:
        zs = [1]

    monkeypatch.setattr(
        probe,
        "_load_batch",
        lambda xyz, indices, cutoff, device: (FakeBatch(), FakeZTable()),
    )
    monkeypatch.setattr(
        probe,
        "_create_model",
        lambda **kwargs: probe.torch.nn.Linear(1, 1),
    )

    def fake_build(base_model, *, candidate, compile_mode, compile_fullgraph):
        seen["candidate"] = candidate
        seen["compile_mode"] = compile_mode
        seen["compile_fullgraph"] = compile_fullgraph
        return marker_candidate

    def fake_compare(reference_model, candidate_model, batch, *, atol, rtol):
        assert candidate_model is marker_candidate
        return {"ok": True, "failed_checks": []}

    monkeypatch.setattr(probe, "build_equivalence_candidate_model", fake_build)
    monkeypatch.setattr(probe, "compare_force_loss_equivalence", fake_compare)

    payload = probe.run_probe(
        argparse.Namespace(
            device="cpu",
            dtype="float32",
            seed=123,
            indices="0",
            xyz="train.xyz",
            cutoff=5.0,
            hidden_channels=8,
            max_ell=1,
            num_interactions=1,
            correlation=1,
            enable_cueq=False,
            modes=[],
            compile_mode="reduce-overhead",
            compile_fullgraph=True,
            allow_fallback=True,
            warmup=0,
            repeats=1,
            equivalence_gate=True,
            equivalence_candidate="compile_readouts",
            equivalence_atol=1.0e-6,
            equivalence_rtol=1.0e-5,
        )
    )

    assert seen == {
        "candidate": "compile_readouts",
        "compile_mode": "reduce-overhead",
        "compile_fullgraph": True,
    }
    assert payload["force_loss_equivalence"]["candidate"] == "compile_readouts"
    assert payload["force_loss_equivalence"]["ok"] is True

def test_build_equivalence_candidate_compiles_radial_embedding(monkeypatch):
    probe = load_probe()
    compiled = []

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.radial_embedding = torch.nn.Linear(2, 2)
            self.readouts = torch.nn.ModuleList([torch.nn.Linear(2, 1)])

    class FakeCompiled(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    def fake_compile(module, *, mode, fullgraph):
        compiled.append((module, mode, fullgraph))
        return FakeCompiled(module)

    monkeypatch.setattr(probe.torch, "compile", fake_compile)
    model = Tiny()

    candidate = probe.build_equivalence_candidate_model(
        model,
        candidate="compile_radial_embedding",
        compile_mode="default",
        compile_fullgraph=False,
    )

    assert candidate is not model
    assert isinstance(candidate.radial_embedding, FakeCompiled)
    assert len(compiled) == 1
    assert compiled[0][1:] == ("default", False)
    assert isinstance(model.radial_embedding, torch.nn.Linear)


def test_equivalence_candidate_choices_include_radial_embedding():
    probe = load_probe()

    assert "compile_radial_embedding" in probe.EQUIVALENCE_CANDIDATES
