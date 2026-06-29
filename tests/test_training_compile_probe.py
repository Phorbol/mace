from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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

    assert payload["force_loss_equivalence"] == {"ok": True, "failed_checks": []}
