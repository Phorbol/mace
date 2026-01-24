
import sys
import os
from pathlib import Path

# Ensure root is in path for neb.py
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import FIRE
from ase.mep import NEB as BaseNEB

from mace.calculators.batch_neb import BatchNEB
from mace.calculators.batch_neb_scheduler import BatchNEBScheduler
from mace.tools.utils import get_atomic_number_table_from_zs

# --- Mocks ---

class DummyMACEModel(torch.nn.Module):
    def forward(self, batch, compute_stress=False, training=False, **kwargs):
        positions = batch["positions"]
        batch_index = batch["batch"]
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        
        # Simple potential: V = 0.5 * k * r^2 (Harmonic oscillator at origin)
        # Forces = -k * r
        # Let k=2.0 -> V = r^2, F = -2r
        node_energy = (positions**2).sum(dim=1)
        energy = torch.zeros(num_graphs, device=positions.device, dtype=positions.dtype)
        if batch_index.numel() > 0:
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

# --- Tests ---

def test_batch_neb_scheduler_runs_two_paths():
    calc = DummyCalculator()
    
    # Path 1: (-2,0,0) -> (-0.5, 0.5, 0) -> (2,0,0)
    img1_0 = Atoms(numbers=[1], positions=[[-2.0, 0.0, 0.0]])
    img1_1 = Atoms(numbers=[1], positions=[[-0.5, 0.5, 0.0]])
    img1_2 = Atoms(numbers=[1], positions=[[2.0, 0.0, 0.0]])
    path1 = [img1_0, img1_1, img1_2]
    
    # Path 2: (0,-2,0) -> (0.5, -0.5, 0) -> (0,2,0)
    img2_0 = Atoms(numbers=[1], positions=[[0.0, -2.0, 0.0]])
    img2_1 = Atoms(numbers=[1], positions=[[0.5, -0.5, 0.0]])
    img2_2 = Atoms(numbers=[1], positions=[[0.0, 2.0, 0.0]])
    path2 = [img2_0, img2_1, img2_2]
    
    # Use BatchNEB for path1 and BaseNEB for path2 to test mixed compatibility
    neb1 = BatchNEB(path1, calc, k=1.0, device='cpu')
    neb2 = BaseNEB(path2, k=1.0) # Standard NEB, no internal batching
    
    opt1 = FIRE(neb1, logfile=None)
    opt2 = FIRE(neb2, logfile=None)
    
    scheduler = BatchNEBScheduler(
        [(neb1, opt1), (neb2, opt2)],
        calc,
        device='cpu'
    )
    
    scheduler.run(fmax=0.05, steps=50)
    
    # Check convergence
    # Both middle images should move towards origin (0,0,0)
    pos1 = path1[1].get_positions()
    pos2 = path2[1].get_positions()
    
    print(f"Path 1 final pos: {pos1}")
    print(f"Path 2 final pos: {pos2}")
    
    assert np.allclose(pos1, [0, 0, 0], atol=0.2)
    assert np.allclose(pos2, [0, 0, 0], atol=0.2)
    
    # Verify BatchNEB flag was set
    assert neb1.external_batching is True
    
    # Verify NEB2 (standard) has results attached
    assert isinstance(path2[1].calc, SinglePointCalculator)

def test_scheduler_handles_endpoints():
    calc = DummyCalculator()
    # Path with no calc on endpoints
    path = [
        Atoms(numbers=[1], positions=[[-1,0,0]]),
        Atoms(numbers=[1], positions=[[0,0.1,0]]),
        Atoms(numbers=[1], positions=[[1,0,0]])
    ]
    neb = BaseNEB(path, k=1.0)
    opt = FIRE(neb, logfile=None)
    
    scheduler = BatchNEBScheduler([(neb, opt)], calc, device='cpu')
    
    # Run 1 step
    scheduler.run(fmax=10.0, steps=1)
    
    # Endpoints should have been computed because they started with None
    assert isinstance(path[0].calc, SinglePointCalculator)
    assert isinstance(path[-1].calc, SinglePointCalculator)
    assert path[0].calc.results['energy'] is not None

