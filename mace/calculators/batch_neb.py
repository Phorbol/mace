
import logging
import sys
import os
import importlib.util
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from mace import data
from mace.calculators.batch_relaxer import _get_mace_config_and_data
from mace.tools import torch_geometric

logger = logging.getLogger("MACE_BatchNEB")

# Try fallback to standard ASE NEB
try:
    from ase.mep import NEB as BaseNEB
except ImportError:
    # If ASE is not installed (unlikely in this context), define a dummy or raise
    raise ImportError("ASE is required for BatchNEB but ase.mep.NEB could not be imported.")

class BatchNEB(BaseNEB):
    """
    MACE-accelerated NEB implementation that computes forces for all images
    in a single GPU batch, avoiding sequential calculator calls.
    """

    def __init__(
        self,
        images: List[Atoms],
        calculator,
        k: Union[float, List[float]] = 0.1,
        climb: bool = False,
        parallel: bool = False,
        remove_rotation_and_translation: bool = False,
        world=None,
        method: str = 'aseneb',
        allow_shared_calculator: bool = True, # We force this to True effectively
        precon=None,
        device: str = 'cuda',
        batch_logfile: Union[str, object, None] = None,
        **kwargs
    ):
        # Force allow_shared_calculator to True because we use one calculator for all
        if not allow_shared_calculator:
            logger.warning("BatchNEB requires shared calculator concept. Setting allow_shared_calculator=True.")
            allow_shared_calculator = True

        super().__init__(
            images,
            k=k,
            climb=climb,
            parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world,
            method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
            **kwargs
        )
        
        self.calculator = calculator
        self.device = device
        self.model = calculator.models[0]
        
        # Determine heads
        self.heads = getattr(calculator, "available_heads", ["Default"])
        if self.heads is None: self.heads = ["Default"]
        
        # Move model to device once
        self.model.to(self.device)
        
        # Flag to allow external schedulers to bypass internal batching
        self.external_batching = False
        
        # Handle batch_logfile
        self.batch_logfile = batch_logfile
        self._batch_log_handle = None
        
        if isinstance(batch_logfile, str):
            if batch_logfile == "-":
                self._batch_log_handle = sys.stdout
            else:
                # We open in append mode to play nice with other loggers
                # But user should manage file lifecycle if possible
                try:
                    self._batch_log_handle = open(batch_logfile, "a")
                except Exception as e:
                    logger.warning(f"Could not open batch_logfile {batch_logfile}: {e}")
        elif hasattr(batch_logfile, "write"):
            self._batch_log_handle = batch_logfile

    def __del__(self):
        # Attempt to close if we opened it and it's not stdout
        if getattr(self, "_batch_log_handle", None) and isinstance(self.batch_logfile, str) and self.batch_logfile != "-":
            try:
                self._batch_log_handle.close()
            except:
                pass

    def get_forces(self):
        """
        Override get_forces to batch-compute forces for images before delegating
        to the parent class for NEB projection logic.
        """
        # 0. Check for external batching control
        if getattr(self, "external_batching", False):
            # If controlled externally, we assume forces are already injected
            # into SinglePointCalculators on the images.
            # We just delegate to super().get_forces() which will read them.
            return super().get_forces()

        # 1. Identify which images need calculation
        # Standard NEB calculates forces for interior images [1:-1]
        # Endpoints [0] and [-1] are usually fixed, but some methods need their energy.
        # We'll check if endpoints have results; if not, include them.
        
        indices_to_compute = []
        
        # Check all images including interior and endpoints
        # Only compute if results are missing (i.e. position changed or not computed yet)
        
        # DEBUG LOGGING (Temporary for diagnosis)
        # logger.info(f"DEBUG: Checking images for calculation...")
        
        # Interior
        for i in range(1, self.nimages - 1):
            should_compute = False
            if self.images[i].calc is None:
                should_compute = True
            elif 'forces' not in self.images[i].calc.results:
                should_compute = True
            else:
                # Check if calculator state matches current atoms (e.g. positions changed)
                # SinglePointCalculator.check_state returns list of changed properties if mismatch
                if self.images[i].calc.check_state(self.images[i]):
                    should_compute = True
            
            if should_compute:
                indices_to_compute.append(i)
            
        # Endpoints (only if they lack results or we want to be safe)
        # For simplicity in Batch mode, if it's cheap, we can just compute them once
        # or check if 'energy' is in cache.
        for i in [0, self.nimages - 1]:
            # If calc is missing or results missing
            if self.images[i].calc is None or 'energy' not in self.images[i].calc.results:
                if i not in indices_to_compute:
                    indices_to_compute.append(i)
        
        indices_to_compute.sort()
        
        # DEBUG LOGGING (Temporary for diagnosis)
        # logger.debug(f"BatchNEB get_forces check:")
        # for i in range(self.nimages):
        #     has_calc = self.images[i].calc is not None
        #     has_forces = 'forces' in self.images[i].calc.results if has_calc else False
        #     logger.debug(f"  Image {i}: has_calc={has_calc}, has_forces={has_forces}")
        # logger.debug(f"  Indices to compute: {indices_to_compute}")
        
        if not indices_to_compute:
            return super().get_forces()

        # 2. Batch Construction
        atoms_list = [self.images[i] for i in indices_to_compute]
        
        # Reuse the helper from batch_relaxer (assuming it's available)
        # We need to handle potential errors if an image is bad
        data_list = []
        valid_indices = []
        
        # Parallelize graph building to avoid CPU bottleneck
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        t0 = time.time()
        
        def _build_data(args):
            idx, atoms = args
            try:
                # _get_mace_config_and_data is pure CPU work (numpy/matscipy)
                # It might release GIL during neighbour list calculation
                data_obj = _get_mace_config_and_data(atoms, self.calculator, heads=self.heads)
                return idx, data_obj, None
            except Exception as e:
                return idx, None, e

        # Use ThreadPoolExecutor
        # Max workers: default is usually min(32, os.cpu_count() + 4)
        # We process atoms_list in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_build_data, zip(indices_to_compute, atoms_list)))
        
        t1 = time.time()
        
        # Sort results to ensure order matches (executor.map preserves order, but good to be safe)
        # Actually executor.map preserves input order.
        
        for idx, data_obj, error in results:
            if error:
                logger.error(f"Failed to convert image {idx} to MACE input: {error}")
                raise error
            if data_obj:
                data_list.append(data_obj)
                valid_indices.append(idx)

        if not data_list:
             return super().get_forces()

        # 3. Batch Forward
        # Use torch_geometric.Batch (or similar)
        batch = torch_geometric.batch.Batch.from_data_list(data_list).to(self.device)
        
        t2 = time.time()
        
        # --- Batch Logging ---
        total_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else sum(len(a) for a in atoms_list)
        msg = (
            f"BatchNEB: Computing forces for {len(data_list)} images "
            f"(Total {total_atoms} atoms) in a single batch. "
            f"[GraphBuild: {t1-t0:.3f}s, Batch: {t2-t1:.3f}s]"
        )
        logger.info(msg)
        
        # Write to batch_logfile if available
        if self._batch_log_handle:
            try:
                # Prepend # to make it a comment in ASE logs
                self._batch_log_handle.write(f"# {msg}\n")
                if hasattr(self._batch_log_handle, "flush"):
                    self._batch_log_handle.flush()
            except Exception:
                pass
        # ---------------------
        
        batch["node_attrs"].requires_grad_(True)
        batch["positions"].requires_grad_(True)
        
        # MACE Forward
        # We assume single model for now (BatchRelaxer limitation too)
        compute_stress = False # NEB usually doesn't need stress
        use_compile = getattr(self.calculator, "use_compile", False)
        
        out = self.model(
            batch.to_dict(),
            compute_stress=compute_stress,
            training=use_compile
        )
        
        t3 = time.time()
        logger.debug(f"BatchNEB: Forward pass took {t3-t2:.3f}s")
        
        # 4. Distribute Results
        energies = out["energy"].detach().cpu().numpy()
        node_forces = out["forces"].detach().cpu().numpy()
        
        # MACE units conversion
        e_conv = self.calculator.energy_units_to_eV
        f_conv = self.calculator.energy_units_to_eV / self.calculator.length_units_to_A
        
        pointer = 0
        for i, idx in enumerate(valid_indices):
            atoms = self.images[idx]
            n_atoms = len(atoms)
            
            e = energies[i] * e_conv
            f = node_forces[pointer : pointer + n_atoms] * f_conv
            pointer += n_atoms
            
            # Attach SinglePointCalculator
            # This "caches" the results so when super().get_forces() calls 
            # atoms.get_forces(), it returns this immediately.
            atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)

        # 5. Delegate to ASE NEB for projections/springs
        return super().get_forces()
