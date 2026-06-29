from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def load_force_profile():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "profile_force_energy_modes.py"
    )
    spec = importlib.util.spec_from_file_location("force_energy_profile", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FallbackOnceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.disabled = False
        self.calls = 0

    def disable_compile_fallback(self, exc):
        self.disabled = True
        return True

    def forward(self, batch_dict, **kwargs):
        self.calls += 1
        if not self.disabled:
            raise RuntimeError("compiled force path failed")
        energy = self.weight * batch_dict["positions"].sum().reshape(1)
        forces = torch.ones_like(batch_dict["positions"]) * self.weight
        return {"energy": energy, "forces": forces}


class TinyBatch:
    def __init__(self):
        self.positions = torch.ones(2, 3)
        self.energy = torch.tensor([6.0])
        self.forces = torch.ones(2, 3)
        self.weight = torch.tensor([1.0])
        self.energy_weight = torch.tensor([1.0])
        self.forces_weight = torch.ones(2)

    def to_dict(self):
        return {"positions": self.positions.detach().clone().requires_grad_(True)}


def test_time_mode_records_compile_fallback_state():
    profile = load_force_profile()
    model = FallbackOnceModel()
    batch = TinyBatch()
    args = argparse.Namespace(
        warmup=0,
        repeats=1,
        energy_weight=1.0,
        forces_weight=1.0,
        max_grad_norm=0.0,
    )

    result = profile._time_mode(
        model,
        batch,
        mode="energy_forward",
        args=args,
        device=torch.device("cpu"),
    )

    assert result["compile_disabled"] is True
    assert result["mode"] == "energy_forward"
    assert model.calls == 2
    assert result["loss_first"] == 0.0


def test_prepare_profile_model_uses_training_compile_wrapper(monkeypatch):
    profile = load_force_profile()
    model = torch.nn.Linear(1, 1)
    wrapped = torch.nn.Sequential(model)
    seen = {}

    def fake_prepare_model_for_training_compile(
        model_arg, *, enabled, mode, fullgraph, allow_fallback
    ):
        seen["model"] = model_arg
        seen["enabled"] = enabled
        seen["mode"] = mode
        seen["fullgraph"] = fullgraph
        seen["allow_fallback"] = allow_fallback
        return wrapped

    monkeypatch.setattr(
        profile,
        "prepare_model_for_training_compile",
        fake_prepare_model_for_training_compile,
        raising=False,
    )
    args = argparse.Namespace(
        train_compile=True,
        train_compile_mode="reduce-overhead",
        train_compile_fullgraph=True,
        train_compile_allow_fallback=False,
    )

    result = profile._prepare_profile_model(model, args)

    assert result is wrapped
    assert seen == {
        "model": model,
        "enabled": True,
        "mode": "reduce-overhead",
        "fullgraph": True,
        "allow_fallback": False,
    }
