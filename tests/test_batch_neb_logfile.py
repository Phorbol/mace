
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

def test_batch_neb_writes_to_stream_logfile():
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    
    # Use StringIO as log file
    log_stream = io.StringIO()
    
    neb = BatchNEB(images, calc, k=1.0, device='cpu', batch_logfile=log_stream)
    
    # Trigger force calculation
    neb.get_forces()
    
    # Check log content
    content = log_stream.getvalue()
    assert "# BatchNEB: Computing forces" in content
    assert "3 images" in content

def test_batch_neb_writes_to_file_logfile(tmp_path):
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    
    log_file = tmp_path / "neb.log"
    
    # Initialize with filename string
    neb = BatchNEB(images, calc, k=1.0, device='cpu', batch_logfile=str(log_file))
    
    # Trigger force calculation
    neb.get_forces()
    
    # Check file content
    assert log_file.exists()
    content = log_file.read_text()
    assert "# BatchNEB: Computing forces" in content
    
    # Clean up (implicit via tmp_path, but object might keep file open)
    # The __del__ should handle it, or GC.

def test_batch_neb_stdout_logfile(capsys):
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    
    # Initialize with "-" for stdout
    neb = BatchNEB(images, calc, k=1.0, device='cpu', batch_logfile="-")
    
    # Trigger force calculation
    neb.get_forces()
    
    # Check stdout
    captured = capsys.readouterr()
    assert "# BatchNEB: Computing forces" in captured.out
