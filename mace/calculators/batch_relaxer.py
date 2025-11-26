#2025.11.27 Jerry ECUST
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
# 引入 ase.io.write 用于流式输出
from ase.io import write as ase_write

from mace import data
from mace.tools import torch_geometric

logger = logging.getLogger("MACE_BatchRelax")

def _get_mace_config_and_data(atoms: Atoms, calculator, heads: List[str]) -> data.AtomicData:
    """Helper to generate MACE AtomicData from ASE Atoms."""
    key_spec = data.KeySpecification(
        info_keys={},
        arrays_keys={calculator.charges_key: "Qs"} if hasattr(calculator, "charges_key") else {}
    )

    config = data.config_from_atoms(
        atoms,
        key_specification=key_spec,
        head_name=heads 
    )

    atomic_data = data.AtomicData.from_config(
        config,
        z_table=calculator.z_table,
        cutoff=calculator.r_max,
        heads=heads, 
    )
    return atomic_data

class RelaxBatch:
    """Internal worker class to manage a dynamic batch of optimizers."""

    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        fmax: float = 0.01,
        atoms_filter_cls=None,
        max_n_steps: int = 500,
        device: str = 'cuda',
        optimizer_kwargs: Dict = None,
        target_heads: List[str] = None 
    ):
        self.calc = calculator
        self.model = calculator.models[0]
        self.optimizer_cls = optimizer_cls
        self.fmax = fmax
        self.atoms_filter_cls = atoms_filter_cls
        self.max_n_steps = max_n_steps
        self.device = device
        self.optimizer_kwargs = optimizer_kwargs or {}
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

        # --- 参数覆盖逻辑 (从 atoms.info 读取 opt_kwargs) ---
        final_kwargs = self.optimizer_kwargs.copy()
        if 'opt_kwargs' in atoms.info and isinstance(atoms.info['opt_kwargs'], dict):
            final_kwargs.update(atoms.info['opt_kwargs'])
        # -----------------------------------------------

        opt = self.optimizer_cls(
            filtered_atoms,
            logfile=logfile, 
            trajectory=None, # 禁用内部 Trajectory，手动管理
            **final_kwargs
        )
        opt.fmax = self.fmax

        # --- 手动 Trajectory 管理 ---
        # 只有当 traj_file 不为 None 时，才记录过程
        traj_handler = None
        if traj_file:
            traj_handler = Trajectory(traj_file, 'w', atoms)
            traj_handler.write(atoms)
        # --------------------------

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

        # 1. 解包真实原子 (Fix: Shape Mismatch for Cell Relax)
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
                # 仅当 trajectories[i] 存在时才写入
                if self.trajectories[i] is not None:
                    self.trajectories[i].write(target_atoms)

class BatchRelaxer:
    """Main Interface for MACE Batch Relaxation."""

    def __init__(
        self,
        calculator,
        optimizer_cls=FIRE,
        max_edges_per_batch: int = 30000,
        relax_cell: bool = False, 
        device: str = 'cuda'
    ):
        self.calc = calculator
        self.optimizer_cls = optimizer_cls
        self.max_edges = max_edges_per_batch
        self.default_relax_cell = relax_cell
        self.device = device
        
        if len(calculator.models) != 1:
            raise ValueError("BatchRelaxer only supports single-model calculators.")

    def relax(
        self, 
        atoms_list: List[Atoms], 
        fmax: float = 0.02, 
        relax_cell: Optional[bool] = None, 
        head: Optional[str] = None, 
        max_n_steps: int = 200,
        inplace: bool = True,
        
        # --- 路径控制参数 ---
        trajectory_dir: Optional[str] = None,       # 控制是否保存【过程轨迹】 (.traj)
        append_trajectory_file: Optional[str] = None, # 控制是否流式保存【最终结果】 (.xyz)
        # ------------------

        save_log_file: Optional[str] = None,
        verbose: bool = False,
        optimizer_kwargs: Dict = None
    ) -> List[Atoms]:
        """
        Run batch relaxation.

        Args:
            atoms_list: List of ASE atoms to relax.
            trajectory_dir: If set, saves optimization history for EACH structure (e.g. dir/0.traj). 
                            Set to None to DISABLE process trajectory generation (saves disk space).
            append_trajectory_file: If set, appends the FINAL relaxed structure of each atom 
                                    to this single file (e.g. 'relaxed.xyz') immediately upon convergence.
        """
        
        # 1. 确定 relax_cell
        use_relax_cell = relax_cell if relax_cell is not None else self.default_relax_cell

        # 2. Head 检测
        available_heads = getattr(self.calc, "available_heads", ["Default"])
        if available_heads is None: available_heads = ["Default"]
        
        target_heads_list = None
        if head is not None:
            if head not in available_heads:
                raise ValueError(f"Selected head '{head}' not in {available_heads}")
            target_heads_list = [head]
            logger.info(f"Using manually selected head: {head}")
        else:
            if len(available_heads) == 1:
                target_heads_list = available_heads
            elif len(available_heads) > 1:
                raise ValueError(f"Multiple heads {available_heads} found. Please specify 'head=...'.")
            else:
                target_heads_list = ["Default"]
        
        # Logging
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
            
        # 准备流式输出文件句柄
        stream_obj = None
        if append_trajectory_file:
            stream_obj = open(append_trajectory_file, 'w')

        filter_cls = FrechetCellFilter if use_relax_cell else None

        worker = RelaxBatch(
            self.calc,
            optimizer_cls=self.optimizer_cls,
            fmax=fmax,
            atoms_filter_cls=filter_cls,
            max_n_steps=max_n_steps,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
            target_heads=target_heads_list
        )

        pbar = tqdm(total=len(atoms_list), desc="Batch Relaxing", unit="struct")
        
        try:
            while len(queue) > 0 or worker.num_active > 0:
                keys_to_remove = []
                for idx in list(queue.keys()):
                    if worker.total_edges >= self.max_edges and worker.num_active > 0:
                        break
                    
                    atoms = queue[idx]
                    try:
                        data_obj = _get_mace_config_and_data(atoms, self.calc, heads=target_heads_list)
                        n_edges = data_obj.edge_index.shape[1]
                    except Exception as e:
                        logger.error(f"Failed to graph structure {idx}: {e}")
                        del queue[idx]
                        pbar.update(1)
                        continue

                    if n_edges > self.max_edges and worker.num_active > 0:
                        break
                    
                    # --- 核心逻辑: 控制过程轨迹 ---
                    traj_path = None
                    if trajectory_dir:
                        # 只有当 trajectory_dir 不为 None 时，才生成路径
                        name = atoms.info.get('name', atoms.info.get('ID', f"{idx}"))
                        traj_path = os.path.join(trajectory_dir, f"{name}.traj")
                    # ---------------------------
                    
                    worker.insert(atoms, n_edges, idx, logfile=None, traj_file=traj_path)
                    del queue[idx]
                
                if worker.num_active > 0:
                    worker.step()
                
                converged = worker.pop_converged()
                if converged:
                    for idx, atoms in converged:
                        relaxed_results[idx] = atoms
                        pbar.update(1)
                        
                        # --- 流式写入最终结果 ---
                        if stream_obj:
                            ase_write(stream_obj, atoms, format='extxyz')
                            stream_obj.flush() # 确保实时写入
                        # ---------------------
        finally:
            if stream_obj:
                stream_obj.close()
            pbar.close()

        logger.info(f"Relaxation finished.")
        return [relaxed_results.get(i, None) for i in range(len(atoms_list))]