import numpy as np
import pytest
import torch
from ase.atoms import Atoms

from mace.calculators.batch_relaxer import BatchRelaxer, logger as batch_relax_logger
from mace.tools.utils import get_atomic_number_table_from_zs


class DummyMACEModel(torch.nn.Module):
    def forward(self, batch, compute_stress=False, training=False, **kwargs):
        positions = batch["positions"]
        batch_index = batch["batch"]
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0

        node_energy = (positions**2).sum(dim=1)
        energy = torch.zeros(num_graphs, device=positions.device, dtype=positions.dtype)
        energy = energy.index_add(0, batch_index, node_energy)

        forces = -2.0 * positions

        out = {"energy": energy, "forces": forces}
        if compute_stress:
            out["stress"] = torch.zeros(
                (num_graphs, 3, 3), device=positions.device, dtype=positions.dtype
            )
        else:
            out["stress"] = None
        return out


class DummyCalculator:
    def __init__(self):
        self.models = [DummyMACEModel()]
        self.z_table = get_atomic_number_table_from_zs([1, 8])
        self.r_max = 3.5
        self.available_heads = ["Default"]
        self.model_type = "MACE"
        self.use_compile = False
        self.energy_units_to_eV = 1.0
        self.length_units_to_A = 1.0


def _max_force_norm(atoms: Atoms) -> float:
    f = atoms.get_forces()
    return float(np.max(np.linalg.norm(f, axis=1)))


def test_batch_relaxer_runs_and_converges_cpu():
    calc = DummyCalculator()
    relaxer = BatchRelaxer(calc, max_edges_per_batch=100000, device="cpu")

    atoms_list = [
        Atoms(numbers=[8, 1, 1], positions=[[1.2, 0.0, 0.0], [0.0, -0.7, 0.0], [0.0, 0.8, 0.0]]),
        Atoms(numbers=[1, 1], positions=[[0.9, 0.0, 0.0], [-0.8, 0.0, 0.0]]),
    ]

    fmax = 0.05
    out = relaxer.relax(atoms_list, fmax=fmax, max_n_steps=200, inplace=True, verbose=False)

    assert len(out) == len(atoms_list)
    assert all(a is not None for a in out)
    assert all(a.calc is not None for a in out)
    assert all(_max_force_norm(a) <= fmax * 1.01 for a in out)


def test_batch_relaxer_compute_stress_flag():
    calc = DummyCalculator()
    relaxer = BatchRelaxer(calc, max_edges_per_batch=100000, device="cpu")

    atoms = Atoms(numbers=[1, 1], positions=[[0.6, 0.0, 0.0], [-0.4, 0.0, 0.0]])
    out = relaxer.relax([atoms], fmax=0.1, max_n_steps=50, compute_stress=False, verbose=False)

    assert out[0] is not None
    assert out[0].calc is not None
    assert ("stress" not in out[0].calc.results) or (out[0].calc.results["stress"] is None)


def test_batch_relaxer_does_not_leak_logger_handlers():
    initial = len(batch_relax_logger.handlers)
    calc = DummyCalculator()
    relaxer = BatchRelaxer(calc, max_edges_per_batch=100000, device="cpu")

    atoms = Atoms(numbers=[1, 1], positions=[[0.5, 0.0, 0.0], [-0.3, 0.0, 0.0]])
    relaxer.relax([atoms], fmax=0.1, max_n_steps=10, verbose=False)

    assert len(batch_relax_logger.handlers) == initial

