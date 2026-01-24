import os
import logging
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

    head_name: str
    if isinstance(heads, list):
        head_name = heads[0] if len(heads) > 0 else "Default"
    else:
        head_name = str(heads)

    config = data.config_from_atoms(
        atoms,
        key_specification=key_spec,
        head_name=head_name
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
        target_heads: List[str] = None,
        compute_stress: Optional[bool] = None,
        data_builder: Optional[Callable[[Atoms], data.AtomicData]] = None,
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
        self.compute_stress = compute_stress
        self.data_builder = (
            data_builder
            if data_builder is not None
            else (lambda atoms: _get_mace_config_and_data(atoms, self.calc, heads=self.target_heads))
        )

        # Batch State
        self.opt_list: List[Optimizer] = []
        self.all_atoms: List[Atoms] = [] 
        self.edge_counts: List[int] = [] 
        self.opt_flags: List[bool] = []  
        self.ids: List[Any] = []
        self.cached_data: List[Optional[data.AtomicData]] = []
        
        self.trajectories: List[Union[Trajectory, None]] = []
        self.total_edges: int = 0

    @property
    def num_active(self) -> int:
        return sum(self.opt_flags)

    def insert(
        self,
        atoms: Atoms,
        num_edges: int,
        idx: Any,
        logfile=None,
        traj_file=None,
        data_obj: Optional[data.AtomicData] = None,
    ) -> None:
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
        self.cached_data.append(data_obj)
        self.trajectories.append(traj_handler)
        
        self.total_edges += num_edges

    def pop_converged(self) -> List[Tuple[Any, Atoms]]:
        """Removes converged atoms and closes their trajectory files."""
        new_opt_list = []
        new_all_atoms = []
        new_edge_counts = []
        new_ids = []
        new_cached_data = []
        new_trajectories = []
        
        converged_items = []

        for i in range(len(self.opt_list)):
            if self.opt_flags[i]:
                new_opt_list.append(self.opt_list[i])
                new_all_atoms.append(self.all_atoms[i])
                new_edge_counts.append(self.edge_counts[i])
                new_ids.append(self.ids[i])
                new_cached_data.append(self.cached_data[i])
                new_trajectories.append(self.trajectories[i])
            else:
                converged_items.append((self.ids[i], self.all_atoms[i]))
                if self.trajectories[i] is not None:
                    self.trajectories[i].close()

        self.opt_list = new_opt_list
        self.all_atoms = new_all_atoms
        self.edge_counts = new_edge_counts
        self.ids = new_ids
        self.cached_data = new_cached_data
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

        data_list: List[data.AtomicData] = []
        for i, atoms in enumerate(real_atoms_list):
            cached = self.cached_data[i]
            if cached is not None:
                data_list.append(cached)
                self.cached_data[i] = None
            else:
                data_list.append(self.data_builder(atoms))

        self.edge_counts = [int(d.edge_index.shape[1]) for d in data_list]
        self.total_edges = sum(self.edge_counts)

        batch = torch_geometric.Batch.from_data_list(data_list).to(self.device)

        # 2. Compute
        use_compile = getattr(self.calc, "use_compile", False)
        
        batch["node_attrs"].requires_grad_(True)
        batch["positions"].requires_grad_(True)
        
        if self.compute_stress is not None:
            compute_stress = self.compute_stress
        else:
            compute_stress = (self.calc.model_type in ["MACE", "EnergyDipoleMACE"]) and (
                not use_compile
            )
        
        out = self.model(
            batch.to_dict(),
            compute_stress=compute_stress,
            training=use_compile
        )

        energies = out["energy"].detach().cpu().numpy()
        node_forces = out["forces"].detach().cpu().numpy()
        stresses = out["stress"].detach().cpu().numpy() if compute_stress else None
        ptr = batch.ptr.detach().cpu().numpy()
        
        for i, opt in enumerate(self.opt_list):
            target_atoms = real_atoms_list[i]
            
            start = int(ptr[i])
            end = int(ptr[i + 1])

            e = float(energies[i]) * self.calc.energy_units_to_eV
            f = (
                node_forces[start:end]
                * self.calc.energy_units_to_eV
                / self.calc.length_units_to_A
            )
            
            s = None
            if stresses is not None:
                stress_i = stresses[i]
                if getattr(stress_i, "ndim", 0) == 3:
                    stress_i = stress_i[0]
                s = full_3x3_to_voigt_6_stress(
                    stress_i
                    * self.calc.energy_units_to_eV
                    / self.calc.length_units_to_A**3
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
        compute_stress: Optional[bool] = None,
        
        # --- 路径控制参数 ---
        trajectory_dir: Optional[str] = None,         # 过程轨迹 (.traj) 存放目录，None 为不保存
        append_trajectory_file: Optional[str] = None, # 最终结果 (.xyz) 流式输出路径，None 为不保存
        # ------------------

        save_log_file: Optional[str] = None,
        verbose: bool = False,
        optimizer_kwargs: Dict = None
    ) -> List[Atoms]:
        """
        Run batch relaxation with dynamic batching, multi-gpu support, and streaming I/O.
        """
        
        # --- 1. 确定是否使用 Cell Relax ---
        use_relax_cell = relax_cell if relax_cell is not None else self.default_relax_cell

        # --- 2. Head 检测逻辑 ---
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
        
        # --- 3. 环境与日志设置 ---
        # 尝试获取 Rank ID 用于进度条显示
        try:
            rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
        except:
            rank = 0

        log_level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(log_level)
        logger.propagate = False
        handlers: List[logging.Handler] = []
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(stream_handler)
        handlers.append(stream_handler)
        if save_log_file:
            file_handler = logging.FileHandler(save_log_file, mode="w")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            logger.addHandler(file_handler)
            handlers.append(file_handler)

        if not inplace:
            atoms_list = [at.copy() for at in atoms_list]
        
        queue = {i: at for i, at in enumerate(atoms_list)}
        relaxed_results = {}
        
        if trajectory_dir:
            os.makedirs(trajectory_dir, exist_ok=True)
            
        # 准备流式输出文件句柄
        stream_obj = None
        if append_trajectory_file:
            # 使用 extxyz 格式以保留 energy/forces 等信息
            stream_obj = open(append_trajectory_file, 'w')

        filter_cls = FrechetCellFilter if use_relax_cell else None

        get_data = lambda atoms: _get_mace_config_and_data(atoms, self.calc, heads=target_heads_list)

        # --- 4. 初始化 Worker ---
        worker = RelaxBatch(
            self.calc,
            optimizer_cls=self.optimizer_cls,
            fmax=fmax,
            atoms_filter_cls=filter_cls,
            max_n_steps=max_n_steps,
            device=self.device,
            optimizer_kwargs=optimizer_kwargs,
            target_heads=target_heads_list,
            compute_stress=compute_stress,
            data_builder=get_data,
        )

        # --- 5. 初始化进度条 (带 Rank 和 负载监控) ---
        pbar = tqdm(
            total=len(atoms_list), 
            desc=f"[Rank {rank}] Relaxing", 
            unit="struct"
        )
        
        try:
            while len(queue) > 0 or worker.num_active > 0:
                
                # --- A. 填充 (FILL) ---
                keys_to_remove = []
                for idx in list(queue.keys()):
                    # 检查显存负载
                    if worker.total_edges >= self.max_edges and worker.num_active > 0:
                        break
                    
                    atoms = queue[idx]
                    try:
                        # 预计算边数 (使用正确的 head)
                        data_obj = get_data(atoms)
                        n_edges = data_obj.edge_index.shape[1]
                    except Exception as e:
                        logger.error(f"Failed to graph structure {idx}: {e}")
                        del queue[idx]
                        pbar.update(1)
                        continue

                    # 再次检查单个结构是否会导致溢出
                    if n_edges > self.max_edges and worker.num_active > 0:
                        break
                    
                    # 决定是否生成调试用的过程轨迹
                    traj_path = None
                    if trajectory_dir:
                        name = atoms.info.get('name', atoms.info.get('ID', f"{idx}"))
                        traj_path = os.path.join(trajectory_dir, f"{name}.traj")
                    
                    worker.insert(atoms, n_edges, idx, logfile=None, traj_file=traj_path, data_obj=data_obj)
                    del queue[idx]
                
                # --- B. 计算 (COMPUTE) ---
                if worker.num_active > 0:
                    worker.step()
                
                # --- C. 清理 (PURGE) ---
                converged = worker.pop_converged()
                if converged:
                    for idx, atoms in converged:
                        relaxed_results[idx] = atoms
                        pbar.update(1)
                        
                        # 流式写入最终结果
                        if stream_obj:
                            ase_write(stream_obj, atoms, format='extxyz')
                            stream_obj.flush() 
                
                # --- D. 更新监控信息 ---
                pbar.set_postfix(
                    active=worker.num_active, 
                    edges=f"{worker.total_edges/1000:.1f}k"
                )

        finally:
            # 确保关闭文件句柄
            if stream_obj:
                stream_obj.close()
            pbar.close()
            for h in handlers:
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        logger.info(f"Relaxation finished.")
        # 按原始顺序返回结果
        return [relaxed_results.get(i, None) for i in range(len(atoms_list))]
