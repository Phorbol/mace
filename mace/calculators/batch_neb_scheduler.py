
import logging
import sys
from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize.optimize import Optimizer

from mace import data
from mace.calculators.batch_relaxer import _get_mace_config_and_data
from mace.tools import torch_geometric

logger = logging.getLogger("MACE_BatchNEBScheduler")

class BatchNEBScheduler:
    """
    Scheduler for running multiple NEB optimizations in parallel with batched force calculations.
    
    This class orchestrates multiple NEB instances (each with its own optimizer) 
    by collecting all active images from all NEBs, computing their forces in a single 
    MACE batch, and then advancing each optimizer.
    """
    
    def __init__(
        self,
        nebs_and_optimizers: List[Tuple[object, Optimizer]],
        calculator,
        device: str = 'cuda',
        max_edges_per_batch: int = 100000, # Optional safety cap, not fully implemented logic yet
        batch_logfile: Union[str, object, None] = None,
    ):
        """
        Args:
            nebs_and_optimizers: List of tuples (neb_instance, optimizer_instance).
                                 NEB instances can be ase.mep.NEB or mace.calculators.batch_neb.BatchNEB.
            calculator: The MACE calculator instance (must be shared/compatible).
            device: Computation device.
            batch_logfile: Log file for batch info (str path, file object, or "-" for stdout).
        """
        self.nebs_and_optimizers = nebs_and_optimizers
        self.calculator = calculator
        self.device = device
        self.model = calculator.models[0]
        
        # Handle batch_logfile
        self.batch_logfile = batch_logfile
        self._batch_log_handle = None
        
        if isinstance(batch_logfile, str):
            if batch_logfile == "-":
                self._batch_log_handle = sys.stdout
            else:
                try:
                    self._batch_log_handle = open(batch_logfile, "a")
                except Exception as e:
                    logger.warning(f"Could not open batch_logfile {batch_logfile}: {e}")
        elif hasattr(batch_logfile, "write"):
            self._batch_log_handle = batch_logfile
        
        # Determine heads
        self.heads = getattr(calculator, "available_heads", ["Default"])
        if self.heads is None: self.heads = ["Default"]
        
        # Ensure model is on device
        self.model.to(self.device)
        
        # Configure BatchNEB instances to passive mode
        for neb, opt in self.nebs_and_optimizers:
            if hasattr(neb, "external_batching"):
                neb.external_batching = True

    def __del__(self):
        # Attempt to close if we opened it and it's not stdout
        if getattr(self, "_batch_log_handle", None) and isinstance(self.batch_logfile, str) and self.batch_logfile != "-":
            try:
                self._batch_log_handle.close()
            except:
                pass

    @classmethod
    def create(
        cls,
        calculator,
        atoms_bin: List[List[Atoms]],
        # NEB configuration
        neb_cls=None,
        neb_kwargs: dict = None,
        # Optimizer configuration
        optimizer_cls=None,
        optimizer_kwargs: dict = None,
        # Scheduler configuration
        **scheduler_kwargs
    ):
        """
        Factory method to easily create a BatchNEBScheduler from a list of paths.
        
        Args:
            calculator: Shared MACE calculator.
            atoms_bin: List of NEB paths (each path is a list of Atoms).
            neb_cls: Class to use for NEB (default: mace.calculators.batch_neb.BatchNEB).
            neb_kwargs: Keyword arguments for NEB class (e.g. k=0.1, climb=True).
            optimizer_cls: Optimizer class (default: ase.optimize.FIRE).
            optimizer_kwargs: Keyword arguments for optimizer (e.g. dt=0.1).
                              Note: logfile will be set to None by default to avoid file handle limits.
            **scheduler_kwargs: Arguments for BatchNEBScheduler (e.g. device, batch_logfile).
            
        Returns:
            Configured BatchNEBScheduler instance.
        """
        # Default imports if None
        if neb_cls is None:
            from mace.calculators.batch_neb import BatchNEB
            neb_cls = BatchNEB
            
        if optimizer_cls is None:
            from ase.optimize import FIRE
            optimizer_cls = FIRE
            
        neb_kwargs = neb_kwargs or {}
        optimizer_kwargs = optimizer_kwargs or {}
        
        # Default optimizer logfile to None to prevent opening hundreds of files
        if "logfile" not in optimizer_kwargs:
            optimizer_kwargs["logfile"] = None
            
        tasks = []
        for i, images in enumerate(atoms_bin):
            # Instantiate NEB
            # We inject calculator because BatchNEB needs it (or standard NEB might ignore if not used in init)
            # BatchNEB signature: (images, calculator, ...)
            # Standard ASE NEB: (images, k=0.1, ...) - doesn't take calculator in init usually!
            # But BatchNEB does. We should check or assume BatchNEB-like.
            # If standard NEB, we might need to attach calculator to images manually?
            # Standard NEB assumes images have calculators or we attach them.
            # BatchNEB takes 'calculator' arg.
            
            try:
                neb = neb_cls(images, calculator=calculator, **neb_kwargs)
            except TypeError:
                # Fallback for standard ASE NEB which doesn't take calculator arg in init
                neb = neb_cls(images, **neb_kwargs)
                # Attach calculator to images if needed? 
                # Actually Scheduler handles the calc via batching, but images need to have *some* calc 
                # or at least we need to be able to attach SinglePointCalculator.
            
            # Instantiate Optimizer
            opt = optimizer_cls(neb, **optimizer_kwargs)
            
            tasks.append((neb, opt))
            
        return cls(tasks, calculator, **scheduler_kwargs)

    def run(self, fmax: float = 0.05, steps: int = 200):
        """
        Run the optimization loop.
        
        Args:
            fmax: Convergence criterion (max force) for all optimizers.
            steps: Maximum number of steps.
        """
        
        for step in range(steps):
            # 1. Check convergence & Collect active optimizers
            active_items = [] # (neb, opt, index_in_list)
            all_converged = True
            
            for i, (neb, opt) in enumerate(self.nebs_and_optimizers):
                # We rely on optimizer's internal convergence check if available,
                # but ASE optimizers usually check convergence at the *end* of step.
                # However, to avoid computing forces for converged paths, we check manually?
                # Standard ASE run loop: while not converged: step().
                # We can check opt.converged() if it exposes it, or use fmax check.
                # Most ASE optimizers have a .converged(force_arrays) method or we check manually.
                
                # Let's assume we run until the optimizer thinks it's done.
                # But ASE optimizers don't easily expose "is_converged" without running logic.
                # We will check the NEB's residual forces.
                
                # If it's the first step, we might not have forces.
                # We force at least one step if not sure.
                
                # Using NEB.get_residual() requires forces to be calculated.
                # If we haven't calculated yet, residual might be old or error.
                
                # Simplified logic: We calculate forces for ALL active NEBs.
                # Then we step. If an optimizer finishes, we remove it from active list?
                # But ASE optimizers don't support "partial run" easily via standard API except .step().
                # We will just call .step() and assume the user checks log.
                
                # Wait, if we call .step() on a converged optimizer, it might just run more steps?
                # ASE optimizers usually don't stop automatically in .step(), only in .run().
                # So we must implement the convergence check here.
                
                # Get residual if available
                residual = 1000.0
                try:
                    residual = neb.get_residual()
                except:
                    pass # Forces not yet computed
                
                if residual > fmax:
                    all_converged = False
                    active_items.append((neb, opt))
                elif step == 0:
                     # Always run first step to establish baseline
                     all_converged = False
                     active_items.append((neb, opt))
            
            if all_converged and step > 0:
                logger.info(f"All NEB paths converged at step {step}.")
                break

            if not active_items:
                break

            # 2. Collect images from active NEBs
            # We need to collect all interior images (1:-1) for standard NEB.
            # Endpoints usually don't move, but we might need their energies?
            # Standard NEB usually only calls get_forces() which iterates 1:-1.
            # We will batch compute 1:-1.
            
            images_to_compute = [] # List of (atoms, original_neb_index, image_index)
            
            for neb, opt in active_items:
                # Assuming standard NEB where 0 and -1 are fixed
                # If endpoints need compute (e.g. not computed yet), we should include them?
                # For safety, let's check endpoints too, similar to BatchNEB logic
                
                indices = []
                # Interior
                for j in range(1, neb.nimages - 1):
                    indices.append(j)
                
                # Endpoints if missing info
                for j in [0, neb.nimages - 1]:
                     if neb.images[j].calc is None or 'energy' not in neb.images[j].calc.results:
                         if j not in indices:
                             indices.append(j)
                
                indices.sort()
                
                for j in indices:
                    images_to_compute.append((neb.images[j], neb, j))
            
            if not images_to_compute:
                # Should not happen if active items exist, but just in case
                continue

            # 3. Batch Compute
            self._batch_compute([img for img, _, _ in images_to_compute])
            
            # 4. Step Optimizers
            # The images now have SinglePointCalculators attached.
            # Calling opt.step() will trigger neb.get_forces(), which will read SPC.
            for neb, opt in active_items:
                # We pass fmax to step? No, step() takes no args usually.
                # ASE Optimizers store fmax in self.fmax if set via run(), 
                # but since we are driving loop, we don't use opt.run().
                # We just call step().
                opt.step()
                
                # After step, log progress?
                # logger.info(f"Step {step}: NEB residual {neb.get_residual():.4f}")

    def _batch_compute(self, atoms_list: List[Atoms]):
        """
        Helper to run MACE on a list of atoms and attach SinglePointCalculators.
        """
        if not atoms_list:
            return

        # --- Decoupling Step: Delegate graph building ---
        # If calculator has a custom batch builder, use it.
        # Otherwise use the default MACE builder.
        
        # We assume self.calculator is the MACE calculator object, but it could be wrapped.
        
        # Default MACE Logic
        data_list = []
        valid_atoms = []
        
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        t0 = time.time()
        
        def _build_data(atoms):
            try:
                # _get_mace_config_and_data is pure CPU work (numpy/matscipy)
                data_obj = _get_mace_config_and_data(atoms, self.calculator, heads=self.heads)
                return atoms, data_obj, None
            except Exception as e:
                return atoms, None, e

        # Use ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_build_data, atoms_list))
        
        t1 = time.time()
        
        for atoms, data_obj, error in results:
            if error:
                logger.error(f"Failed to convert atoms to MACE input: {error}")
                raise error
            if data_obj:
                data_list.append(data_obj)
                valid_atoms.append(atoms)
        
        if not data_list:
            return

        # Batch Forward
        batch = torch_geometric.batch.Batch.from_data_list(data_list).to(self.device)
        
        t2 = time.time()
        
        # --- Batch Logging ---
        total_atoms = batch.num_nodes if hasattr(batch, 'num_nodes') else sum(len(a) for a in atoms_list)
        msg = (
            f"BatchNEBScheduler: Computing forces for {len(data_list)} images "
            f"(Total {total_atoms} atoms) in a single global batch. "
            f"[GraphBuild: {t1-t0:.3f}s, Batch: {t2-t1:.3f}s]"
        )
        logger.info(msg)
        
        # Write to batch_logfile if available
        if self._batch_log_handle:
            try:
                self._batch_log_handle.write(f"# {msg}\n")
                if hasattr(self._batch_log_handle, "flush"):
                    self._batch_log_handle.flush()
            except Exception:
                pass
        
        # Attempt to write to active optimizers' logfiles if available
        for _, opt in self.nebs_and_optimizers:
            if hasattr(opt, 'logfile') and hasattr(opt.logfile, 'write'):
                try:
                    opt.logfile.write(f"# {msg}\n")
                    opt.logfile.flush()
                except Exception:
                    pass
        # ---------------------
        
        batch["node_attrs"].requires_grad_(True)
        batch["positions"].requires_grad_(True)
        
        use_compile = getattr(self.calculator, "use_compile", False)
        
        out = self.model(
            batch.to_dict(),
            compute_stress=False,
            training=use_compile
        )
        
        t3 = time.time()
        logger.debug(f"BatchNEBScheduler: Forward pass took {t3-t2:.3f}s")
        
        energies = out["energy"].detach().cpu().numpy()
        node_forces = out["forces"].detach().cpu().numpy()
        
        e_conv = self.calculator.energy_units_to_eV
        f_conv = self.calculator.energy_units_to_eV / self.calculator.length_units_to_A
        
        pointer = 0
        for i, atoms in enumerate(valid_atoms):
            n_atoms = len(atoms)
            e = energies[i] * e_conv
            f = node_forces[pointer : pointer + n_atoms] * f_conv
            pointer += n_atoms
            
            atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)

