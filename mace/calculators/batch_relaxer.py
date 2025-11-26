import os
import logging
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase.stress import full_3x3_to_voigt_6_stress
from ase.io import Trajectory 

from mace import data
from mace.tools import torch_geometric

logger = logging.getLogger("MACE_BatchRelax")

def _get_mace_config_and_data(atoms: Atoms, calculator, heads: List[str]) -> data.AtomicData:
    """
    Helper to generate MACE AtomicData from ASE Atoms.
    Now accepts dynamic 'heads' list.
    """
    # Key Specification
    key_spec = data.KeySpecification(
        info_keys={},
        arrays_keys={calculator.charges_key: "Qs"} if hasattr(calculator, "charges_key") else {}
    )

    # Config
    config = data.config_from_atoms(
        atoms,
        key_specification=key_spec,
        head_name=heads # Pass the specific head(s)
    )

    # Atomic Data
    atomic_data = data.AtomicData.from_config(
        config,
        z_table=calculator.z_table,
        cutoff=calculator.r_max,
        heads=heads, # Pass the specific head(s)
    )
    return atomic_data

class RelaxBatch:
    """
    Internal worker class to manage a dynamic batch of optimizers.
    """

    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        fmax: float = 0.01,
        atoms_filter_cls=None,
        max_n_steps: int = 500,
        device: str = 'cuda',
        optimizer_kwargs: Dict = None,
        target_heads: List[str] = None # <-- 新增：接收确定的 heads
    ):
        self.calc = calculator
        self.model = calculator.models[0]
        self.optimizer_cls = optimizer_cls
        self.fmax = fmax
        self.atoms_filter_cls = atoms_filter_cls
        self.max_n_steps = max_n_steps
        self.device = device
        self.optimizer_kwargs = optimizer_kwargs or {}
        
        # 确保 target_heads 是列表，默认为 ["Default"]
        self.target_heads = target_heads if target_heads else ["Default"]

        # Batch State
        self.opt_list: List[Optimizer] = []
        self.all_atoms: List[Atoms] = [] 
        self.edge_counts: List[int] = [] 
        self.opt_flags: List[bool] = []  
        self.ids: List[Any] = []
        
        self.trajectories: List[Union[Trajectory, None]] = []
        self.total_edges: int = 0

    @property
    def num_active(self) -> int:
        return sum(self.opt_flags)

    def insert(self, atoms: Atoms, num_edges: int, idx: Any, logfile=None, traj_file=None) -> None:
        """Insert a new atoms object into the batch."""
        atoms.calc = SinglePointCalculator(atoms)

        if self.atoms_filter_cls:
            filtered_atoms = self.atoms_filter_cls(atoms)
        else:
            filtered_atoms = atoms

        opt = self.optimizer_cls(
            filtered_atoms,
            logfile=logfile, 
            trajectory=None, 
            **self.optimizer_kwargs
        )
        opt.fmax = self.fmax

        traj_handler = None
        if traj_file:
            traj_handler = Trajectory(traj_file, 'w', atoms)
            traj_handler.write(atoms)

        self.opt_list.append(opt)
        self.all_atoms.append(atoms)
        self.edge_counts.append(num_edges)
        self.opt_flags.append(True)
        self.ids.append(idx)
        self.trajectories.append(traj_handler)
        
        self.total_edges += num_edges

    def pop_converged(self) -> List[Tuple[Any, Atoms]]:
        """Removes converged atoms and closes their trajectory files."""
        new_opt_list = []
        new_all_atoms = []
        new_edge_counts = []
        new_ids = []
        new_trajectories = []
        
        converged_items = []

        for i in range(len(self.opt_list)):
            if self.opt_flags[i]:
                new_opt_list.append(self.opt_list[i])
                new_all_atoms.append(self.all_atoms[i])
                new_edge_counts.append(self.edge_counts[i])
                new_ids.append(self.ids[i])
                new_trajectories.append(self.trajectories[i])
            else:
                converged_items.append((self.ids[i], self.all_atoms[i]))
                if self.trajectories[i] is not None:
                    self.trajectories[i].close()

        self.opt_list = new_opt_list
        self.all_atoms = new_all_atoms
        self.edge_counts = new_edge_counts
        self.ids = new_ids
        self.trajectories = new_trajectories
        self.opt_flags = [True] * len(self.opt_list)
        self.total_edges = sum(self.edge_counts)

        return converged_items

    def step(self):
        """Performs one optimization step."""
        if not self.opt_list:
            return

        # 1. Prepare Batch Data
        # --- 核心修改：使用 self.target_heads ---
        real_atoms_list = []
        for opt in self.opt_list:
            if self.atoms_filter_cls:
                real_atoms_list.append(opt.atoms.atoms)
            else:
                real_atoms_list.append(opt.atoms)

        data_list = [
            _get_mace_config_and_data(atoms, self.calc, heads=self.target_heads)
            for atoms in real_atoms_list
        ]
        # -------------------------------------

        loader = torch_geometric.dataloader.DataLoader(
            dataset=data_list,
            batch_size=len(data_list),
            shuffle=False,
            drop_last=False
        )
        batch = next(iter(loader)).to(self.device)

        # 2. Compute
        batch_clone = batch.clone()
        use_compile = getattr(self.calc, "use_compile", False)
        
        batch_clone["node_attrs"].requires_grad_(True)
        batch_clone["positions"].requires_grad_(True)
        
        compute_stress = (self.calc.model_type in ["MACE", "EnergyDipoleMACE"]) and (not use_compile)
        
        out = self.model(
            batch_clone.to_dict(),
            compute_stress=compute_stress,
            training=use_compile
        )

        energies = out["energy"].detach().cpu().numpy()
        node_forces = out["forces"].detach().cpu().numpy()
        stresses = out["stress"].detach().cpu().numpy() if compute_stress else None

        pointer = 0
        
        for i, opt in enumerate(self.opt_list):
            target_atoms = real_atoms_list[i]
            n_atoms = len(target_atoms)
            
            e = energies[i] * self.calc.energy_units_to_eV
            f = node_forces[pointer : pointer + n_atoms] * self.calc.energy_units_to_eV / self.calc.length_units_to_A
            pointer += n_atoms
            
            s = None
            if stresses is not None:
                s = full_3x3_to_voigt_6_stress(
                    stresses[i] * self.calc.energy_units_to_eV / self.calc.length_units_to_A**3
                )

            target_atoms.calc = SinglePointCalculator(
                target_atoms, energy=e, forces=f, stress=s
            )

            current_f = opt.atoms.get_forces().flatten()
            step_count = getattr(opt, "nsteps", 0) 
            
            if opt.converged(current_f) or (step_count >= self.max_n_steps):
                self.opt_flags[i] = False
            else:
                opt.step()
                if self.trajectories[i] is not None:
                    self.trajectories[i].write(target_atoms)

class BatchRelaxer:
    """Main Interface for MACE Batch Relaxation."""

    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        max_edges_per_batch: int = 30000,
        device: str = 'cuda'
    ):
        self.calc = calculator
        self.optimizer_cls = optimizer_cls
        self.max_edges = max_edges_per_batch
        self.device = device
        
        if len(calculator.models) != 1:
            raise ValueError("BatchRelaxer only supports single-model calculators.")

    def relax(
        self, 
        atoms_list: List[Atoms], 
        fmax: float = 0.02, 
        relax_cell: bool = False,
        head: Optional[str] = None, # <-- 新增：允许用户指定 head
        max_n_steps: int = 200,
        inplace: bool = True,
        trajectory_dir: Optional[str] = None,
        save_log_file: Optional[str] = None,
        verbose: bool = False,
        optimizer_kwargs: Dict = None
    ) -> List[Atoms]:
        
        # --- Head Validation Logic (新增的核心逻辑) ---
        # 1. 获取可用 heads，兼容旧版本
        available_heads = getattr(self.calc, "available_heads", ["Default"])
        if available_heads is None: 
            available_heads = ["Default"]
        
        target_heads_list = None

        if head is not None:
            # A. 用户手动指定了 head
            if head not in available_heads:
                raise ValueError(
                    f"Selected head '{head}' is not in available_heads: {available_heads}"
                )
            target_heads_list = [head]
            logger.info(f"Using manually selected head: {head}")
        else:
            # B. 用户未指定，自动检测
            if len(available_heads) == 1:
                target_heads_list = available_heads
                logger.debug(f"Auto-detected single available head: {available_heads[0]}")
            elif len(available_heads) > 1:
                # 多头模型，但未指定，报错防止歧义
                raise ValueError(
                    f"Calculator has multiple heads {available_heads}. "
                    "You must explicitly provide the 'head' argument to relax()."
                )
            else:
                # 理论上不应到达这里，作为 fallback
                target_heads_list = ["Default"]
        # ------------------------------------------------
        
        # Logging Setup
        log_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(log_level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(handler)
        
        if save_log_file:
            file_handler = logging.FileHandler(save_log_file, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(file_handler)

        if not inplace:
            atoms_list = [at.copy() for at in atoms_list]
        
        queue = {i: at for i, at in enumerate(atoms_list)}
        relaxed_results = {}
        
        if trajectory_dir:
            os.makedirs(trajectory_dir, exist_ok=True)

        filter_cls = FrechetCellFilter if relax_cell else None

        worker = RelaxBatch(
            self.calc,
            optimizer_cls=self.optimizer_cls,
            fmax=fmax,
            atoms_filter_cls=filter_cls,
            max_n_steps=max_n_steps,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
            target_heads=target_heads_list # <-- 传入确定的 heads
        )

        pbar = tqdm(total=len(atoms_list), desc="Batch Relaxing", unit="struct")
        
        while len(queue) > 0 or worker.num_active > 0:
            
            keys_to_remove = []
            for idx in list(queue.keys()):
                if worker.total_edges >= self.max_edges and worker.num_active > 0:
                    break
                
                atoms = queue[idx]
                
                try:
                    # 获取 edges 时也使用正确的 head
                    data_obj = _get_mace_config_and_data(atoms, self.calc, heads=target_heads_list)
                    n_edges = data_obj.edge_index.shape[1]
                except Exception as e:
                    logger.error(f"Failed to graph structure {idx}: {e}")
                    del queue[idx]
                    pbar.update(1)
                    continue

                if n_edges > self.max_edges and worker.num_active > 0:
                    break
                
                traj_path = None
                if trajectory_dir:
                    name = atoms.info.get('name', atoms.info.get('ID', f"{idx}"))
                    traj_path = os.path.join(trajectory_dir, f"{name}.traj")
                
                worker.insert(atoms, n_edges, idx, logfile=None, traj_file=traj_path)
                del queue[idx]
            
            if worker.num_active > 0:
                worker.step()
            
            converged = worker.pop_converged()
            if converged:
                for idx, atoms in converged:
                    relaxed_results[idx] = atoms
                    pbar.update(1)

        pbar.close()
        logger.info(f"Relaxation finished.")
        
        return [relaxed_results.get(i, None) for i in range(len(atoms_list))]