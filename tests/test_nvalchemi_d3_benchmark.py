from __future__ import annotations

import json
import sys

import pytest

from scripts.benchmarks.nvalchemi_d3_smoke import run_nvalchemi_d3_benchmark as bench


def test_parse_structure_indices_accepts_csv_and_slice_forms():
    assert bench.parse_structure_indices("0,2,5", total=10) == [0, 2, 5]
    assert bench.parse_structure_indices("1:6:2", total=10) == [1, 3, 5]
    assert bench.parse_structure_indices(None, total=5, max_structures=3) == [0, 1, 2]


def test_parse_structure_indices_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="out of range"):
        bench.parse_structure_indices("0,5", total=5)
    with pytest.raises(ValueError, match="at least one"):
        bench.parse_structure_indices("2:2", total=5)


def test_summarize_nvalchemi_records_reports_accuracy_and_timing():
    records = [
        {
            "index": 0,
            "natoms": 2,
            "energy_abs_diff_eV": 1.0e-8,
            "force_max_abs_diff_eV_A": 2.0e-7,
            "nvalchemi_cpu_seconds_per_eval": 0.010,
            "nvalchemi_cuda_seconds_per_eval": 0.004,
        },
        {
            "index": 1,
            "natoms": 4,
            "energy_abs_diff_eV": 3.0e-8,
            "force_max_abs_diff_eV_A": 5.0e-7,
            "nvalchemi_cpu_seconds_per_eval": 0.020,
            "nvalchemi_cuda_seconds_per_eval": 0.005,
        },
    ]

    summary = bench.summarize_records(records, torch_dftd_available=False)

    assert summary["num_structures"] == 2
    assert summary["total_atoms"] == 6
    assert summary["max_energy_abs_diff_eV"] == pytest.approx(3.0e-8)
    assert summary["max_force_abs_diff_eV_A"] == pytest.approx(5.0e-7)
    assert summary["mean_nvalchemi_cpu_seconds_per_eval"] == pytest.approx(0.015)
    assert summary["mean_nvalchemi_cuda_seconds_per_eval"] == pytest.approx(0.0045)
    assert summary["mean_nvalchemi_cuda_speedup_vs_cpu"] == pytest.approx(0.015 / 0.0045)
    assert summary["torch_dftd_comparison"] == "unavailable"


def test_json_payload_marks_missing_torch_dftd_without_failing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_dftd", None)
    payload = bench.build_json_payload(
        records=[],
        torch_dftd_available=False,
        metadata={"repeat": 3, "source": "dummy.xyz"},
    )

    parsed = json.loads(json.dumps(payload))
    assert parsed["metadata"]["repeat"] == 3
    assert parsed["summary"]["torch_dftd_comparison"] == "unavailable"
    assert parsed["records"] == []


def test_parse_and_apply_supercell_repeats_periodic_structure():
    from ase import Atoms

    atoms = Atoms(
        numbers=[46, 46],
        positions=[[0, 0, 0], [1, 1, 1]],
        cell=[3, 3, 3],
        pbc=True,
    )

    repeated = bench.apply_supercell(atoms, bench.parse_supercell("2,1,3"))

    assert len(repeated) == 12
    assert repeated.pbc.tolist() == [True, True, True]
    assert repeated.cell.lengths().tolist() == pytest.approx([6, 3, 9])


def test_parse_supercell_rejects_invalid_values():
    with pytest.raises(ValueError, match="three comma-separated"):
        bench.parse_supercell("2,2")
    with pytest.raises(ValueError, match="positive"):
        bench.parse_supercell("1,0,1")


def test_evaluate_atoms_clears_ase_calculator_cache_each_repeat():
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    calls = {"count": 0}

    class CountingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            calls["count"] += 1
            self.results["energy"] = float(calls["count"])
            self.results["forces"] = np.zeros((len(atoms), 3))

    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.75]])

    result = bench.evaluate_atoms(
        atoms,
        calculator_factory=CountingCalculator,
        repeat=3,
        device="cpu",
    )

    assert calls["count"] == 4
    assert result["energy_eV"] == 4.0


def test_smoke_evaluate_clears_ase_calculator_cache_each_repeat(monkeypatch):
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from scripts.benchmarks.nvalchemi_d3_smoke import run_nvalchemi_d3_smoke as smoke

    calls = {"count": 0}

    class CountingCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, *args, **kwargs):
            super().__init__()

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            calls["count"] += 1
            self.results["energy"] = float(calls["count"])
            self.results["forces"] = np.zeros((len(atoms), 3))

    monkeypatch.setattr(smoke, "NvalchemiDFTD3Calculator", CountingCalculator)

    result = smoke.evaluate("cpu", repeat=3, xyz=None, index=0)

    assert calls["count"] == 4
    assert result["energy"] == 4.0
