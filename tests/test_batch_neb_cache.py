
import sys
import os
import io
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
        
        node_energy = (positions**2).sum(dim=1)
        energy = torch.zeros(num_graphs, device=positions.device, dtype=positions.dtype)
        if batch_index.numel() > 0:
            energy = energy.index_add(0, batch_index, node_energy)
        
        forces = -2.0 * positions
        
        out = {"energy": energy, "forces": forces}
        out["stress"] = None
        return out

class DummyCalculator:
    def __init__(self):
        self.models = [DummyMACEModel()]
        self.z_table = get_atomic_number_table_from_zs([1])
        self.r_max = 3.5
        self.available_heads = ["Default"]
        self.model_type = "MACE"
        self.use_compile = False
        self.energy_units_to_eV = 1.0
        self.length_units_to_A = 1.0

# --- Tests ---

def test_batch_neb_avoids_recalculation():
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    
    # Capture logs to verify calls
    log_stream = io.StringIO()
    neb = BatchNEB(images, calc, k=1.0, device='cpu', batch_logfile=log_stream)
    
    # 1. First call: Should compute all (or at least interior)
    neb.get_forces()
    
    log_content = log_stream.getvalue()
    assert "# BatchNEB: Computing forces" in log_content
    # Count occurrences
    count_first = log_content.count("# BatchNEB: Computing forces")
    assert count_first == 1
    
    # 2. Second call: Position unchanged, should reuse SPC
    # The SPC attached by first call should still be valid
    neb.get_forces()
    
    log_content_2 = log_stream.getvalue()
    count_second = log_content_2.count("# BatchNEB: Computing forces")
    
    # Should NOT increase count
    assert count_second == count_first
    
    # 3. Modify position: Should trigger recalculation
    # We must be careful: modifying position directly on image triggers calc.reset()
    # But only if we use set_positions or similar.
    # Direct array modification might not trigger reset unless we notify atoms.
    
    images[1].set_positions([[0.1, 0.1, 0.1]])
    
    # Now get_forces should recompute
    neb.get_forces()
    
    log_content_3 = log_stream.getvalue()
    count_third = log_content_3.count("# BatchNEB: Computing forces")
    
    assert count_third == count_second + 1
