
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
from ase.mep import NEB as BaseNEB

# Try to import BatchNEB
try:
    from mace.calculators.batch_neb import BatchNEB
except ImportError:
    pytest.skip("BatchNEB not importable", allow_module_level=True)

from mace.tools.utils import get_atomic_number_table_from_zs

# --- Mocks ---

class DummyMACEModel(torch.nn.Module):
    def forward(self, batch, compute_stress=False, training=False, **kwargs):
        positions = batch["positions"]
        batch_index = batch["batch"]
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        
        # Simple potential: V = sum(r^2)
        # Forces = -2r
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

def test_batch_neb_initialization():
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[0, 0, 0]]) for _ in range(3)]
    neb = BatchNEB(images, calc, k=0.1, device='cpu')
    assert neb.calculator is calc
    assert neb.parallel is False
    assert neb.allow_shared_calculator is True

def test_batch_neb_get_forces_runs_batch():
    calc = DummyCalculator()
    
    # Linear path: (-1,0,0) -> (0,0,0) -> (1,0,0)
    # Interior is (0,0,0). Force should be 0 from potential, but NEB springs will pull it.
    img0 = Atoms(numbers=[1], positions=[[-1.0, 0.0, 0.0]])
    img1 = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]])
    img2 = Atoms(numbers=[1], positions=[[1.0, 0.0, 0.0]])
    
    images = [img0, img1, img2]
    
    # Fix endpoints manually (standard NEB practice) via SinglePointCalculator
    # or let BatchNEB handle it if they have no calc.
    # In BatchNEB.get_forces, if endpoints have no calc, it computes them.
    
    neb = BatchNEB(images, calc, k=1.0, device='cpu')
    
    # First call: should compute all 3 images (endpoints have no calc)
    forces = neb.get_forces()
    
    # Check if SinglePointCalculator was attached
    assert isinstance(images[1].calc, SinglePointCalculator)
    assert images[1].calc.results['energy'] is not None
    assert images[1].calc.results['forces'] is not None
    
    # Potential force on img1 (0,0,0) is 0.
    # Spring force: img0 is at -1, img2 at 1. img1 is exactly in middle.
    # Tangent should be x-axis. Spring forces cancel out?
    # Let's check raw values from calc
    raw_f = images[1].calc.results['forces']
    assert np.allclose(raw_f, [0, 0, 0], atol=1e-5)
    
    # Check if forces array returned by get_forces has correct shape
    # ASE NEB returns ((nimages-2)*natoms, 3)
    assert forces.shape == ((3-2)*1, 3) # (1, 3)

def test_batch_neb_integration_with_optimizer():
    from ase.optimize import FIRE
    
    calc = DummyCalculator()
    # Path: (-2,0,0) -> (-0.5, 0.5, 0) -> (2,0,0)
    # Optimal path is straight line (-2,0,0) -> (2,0,0) through (0,0,0)
    
    img0 = Atoms(numbers=[1], positions=[[-2.0, 0.0, 0.0]])
    img1 = Atoms(numbers=[1], positions=[[-0.5, 0.5, 0.0]]) # Perturbed
    img2 = Atoms(numbers=[1], positions=[[2.0, 0.0, 0.0]])
    
    images = [img0, img1, img2]
    neb = BatchNEB(images, calc, k=1.0, device='cpu')
    
    opt = FIRE(neb, logfile=None)
    opt.run(fmax=0.05, steps=50)
    
    # Check convergence
    # Middle image should move towards (0,0,0) (potential minimum and straight line)
    pos = images[1].get_positions()
    assert np.allclose(pos, [0, 0, 0], atol=0.2) # approximate
