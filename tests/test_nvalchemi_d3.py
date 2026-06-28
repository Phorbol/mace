from __future__ import annotations

import builtins
import sys
import types

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.mixing import SumCalculator


def test_pbe_bj_d3_parameters_are_explicit():
    from mace.calculators.nvalchemi_d3 import get_d3_bj_parameters

    params = get_d3_bj_parameters(xc="pbe", damping="bj")

    assert params == {"a1": pytest.approx(0.4289), "a2": pytest.approx(4.4407), "s8": pytest.approx(0.7875)}


def test_unsupported_d3_mapping_raises_clear_error():
    from mace.calculators.nvalchemi_d3 import get_d3_bj_parameters

    with pytest.raises(NotImplementedError, match=r"PBE-D3\(BJ\)"):
        get_d3_bj_parameters(xc="r2scan", damping="bj")
    with pytest.raises(NotImplementedError, match=r"PBE-D3\(BJ\)"):
        get_d3_bj_parameters(xc="pbe", damping="zero")


def test_nvalchemi_d3_calculator_reports_missing_optional_dependency(monkeypatch):
    from mace.calculators.nvalchemi_d3 import NvalchemiDFTD3Calculator

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("nvalchemi"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="nvalchemi-toolkit"):
        NvalchemiDFTD3Calculator(device="cpu", auto_download=False)


def test_nvalchemi_d3_calculator_adapts_ase_atoms(monkeypatch):
    calls = {}

    class FakeAtomicData:
        @staticmethod
        def from_atoms(atoms):
            calls["atoms_numbers"] = atoms.numbers.tolist()
            return "atomic-data"

    class FakeBatch:
        @classmethod
        def from_data_list(cls, data_list, device=None):
            calls["data_list"] = data_list
            calls["device"] = device
            return "batch"

    class FakeDFTD3ModelWrapper:
        def __init__(self, **kwargs):
            calls["model_kwargs"] = kwargs
            self.model_config = types.SimpleNamespace(active_outputs={"energy", "forces"}, neighbor_config="neighbor-config")

        def to(self, *, device=None, dtype=None):
            calls["to"] = {"device": device, "dtype": dtype}
            return self

        def __call__(self, batch):
            calls["batch"] = batch
            return {
                "energy": torch.tensor([[1.25]], dtype=torch.float64),
                "forces": torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=torch.float64),
                "stress": torch.tensor([[[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]], dtype=torch.float64),
            }

    def fake_compute_neighbors(batch, *, config):
        calls["neighbor_batch"] = batch
        calls["neighbor_config"] = config

    dftd3_mod = types.ModuleType("nvalchemi.models.dftd3")
    dftd3_mod.DFTD3ModelWrapper = FakeDFTD3ModelWrapper
    data_mod = types.ModuleType("nvalchemi.data")
    data_mod.AtomicData = FakeAtomicData
    data_mod.Batch = FakeBatch
    neighbors_mod = types.ModuleType("nvalchemi.neighbors")
    neighbors_mod.compute_neighbors = fake_compute_neighbors
    monkeypatch.setitem(sys.modules, "nvalchemi", types.ModuleType("nvalchemi"))
    monkeypatch.setitem(sys.modules, "nvalchemi.models", types.ModuleType("nvalchemi.models"))
    monkeypatch.setitem(sys.modules, "nvalchemi.models.dftd3", dftd3_mod)
    monkeypatch.setitem(sys.modules, "nvalchemi.data", data_mod)
    monkeypatch.setitem(sys.modules, "nvalchemi.neighbors", neighbors_mod)

    from mace.calculators.nvalchemi_d3 import NvalchemiDFTD3Calculator

    calc = NvalchemiDFTD3Calculator(device="cuda", dtype=torch.float64, auto_download=False)
    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.75]])
    calc.calculate(atoms, properties=["energy", "forces", "stress"])

    assert calls["atoms_numbers"] == [1, 1]
    assert calls["data_list"] == ["atomic-data"]
    assert calls["device"] == "cuda"
    assert calls["neighbor_batch"] == "batch"
    assert calls["neighbor_config"] == "neighbor-config"
    assert calls["batch"] == "batch"
    assert calls["model_kwargs"]["a1"] == pytest.approx(0.4289)
    assert calls["model_kwargs"]["auto_download"] is False
    assert calc.results["energy"] == pytest.approx(1.25)
    np.testing.assert_allclose(calc.results["forces"], [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    np.testing.assert_allclose(calc.results["stress"], [1.0, 2.0, 3.0, 0.3, 0.2, 0.1])


def test_mace_mp_accepts_nvalchemi_dispersion_backend(monkeypatch):
    import mace.calculators.foundations_models as foundations

    class DummyMACECalculator:
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyD3Calculator:
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(foundations, "download_mace_mp_checkpoint", lambda model: "model.pt")
    monkeypatch.setattr(foundations, "MACECalculator", DummyMACECalculator)
    monkeypatch.setattr("mace.calculators.nvalchemi_d3.NvalchemiDFTD3Calculator", DummyD3Calculator)

    calc = foundations.mace_mp(
        model="small",
        device="cuda",
        dispersion=True,
        dispersion_backend="nvalchemi",
        dispersion_xc="pbe",
        damping="bj",
    )

    assert isinstance(calc, SumCalculator)
    assert isinstance(calc.mixer.calcs[0], DummyMACECalculator)
    assert isinstance(calc.mixer.calcs[1], DummyD3Calculator)
    assert calc.mixer.calcs[1].kwargs["device"] == "cuda"
    assert calc.mixer.calcs[1].kwargs["xc"] == "pbe"
