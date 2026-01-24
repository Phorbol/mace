
import sys
import os
import logging
from pathlib import Path

# Ensure root is in path for neb.py
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pytest
import torch
from ase.atoms import Atoms
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

def test_batch_neb_logs_batch_info(caplog):
    caplog.set_level(logging.INFO)
    
    calc = DummyCalculator()
    images = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    
    neb = BatchNEB(images, calc, k=1.0, device='cpu')
    
    # Trigger force calculation
    neb.get_forces()
    
    # Check logs
    # We expect 3 images (0,1,2) to be computed in first step
    # "BatchNEB: Computing forces for 3 images (Total 3 atoms) in a single batch."
    
    found_log = False
    for record in caplog.records:
        if "BatchNEB: Computing forces" in record.message:
            found_log = True
            assert "3 images" in record.message
            assert "Total 3 atoms" in record.message
            break
    
    assert found_log, "Did not find BatchNEB log message"

def test_batch_neb_scheduler_logs_batch_info(caplog):
    caplog.set_level(logging.INFO)
    
    calc = DummyCalculator()
    
    # Path 1: 3 images
    path1 = [Atoms(numbers=[1], positions=[[i, 0, 0]]) for i in range(3)]
    neb1 = BatchNEB(path1, calc, k=1.0, device='cpu')
    opt1 = FIRE(neb1, logfile=None)
    
    # Path 2: 3 images
    path2 = [Atoms(numbers=[1], positions=[[0, i, 0]]) for i in range(3)]
    neb2 = BatchNEB(path2, calc, k=1.0, device='cpu')
    opt2 = FIRE(neb2, logfile=None)
    
    scheduler = BatchNEBScheduler([(neb1, opt1), (neb2, opt2)], calc, device='cpu')
    
    # Run 1 step
    scheduler.run(fmax=10.0, steps=1)
    
    # Check logs
    # Expect 6 images total (3+3) computed in scheduler
    # "BatchNEBScheduler: Computing forces for 6 images (Total 6 atoms) in a single global batch."
    
    found_log = False
    for record in caplog.records:
        if "BatchNEBScheduler: Computing forces" in record.message:
            found_log = True
            # Might be 6 images or less depending on if endpoints are cached or logic.
            # In first step, endpoints have no cache, so all 6 should be computed.
            assert "6 images" in record.message
            assert "Total 6 atoms" in record.message
            break
            
    assert found_log, "Did not find BatchNEBScheduler log message"
