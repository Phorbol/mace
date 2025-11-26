# MACE Batch Relaxer (English)



A high-throughput structure relaxation tool based on [MACE](https://github.com/ACEsuit/mace) and [ASE](https://wiki.fysik.dtu.dk/ase/). Designed for GPUs, it utilizes **Dynamic Batching** to mix atomic structures of varying sizes and convergence speeds, maximizing GPU saturation and throughput.↳



### ✨ Key Features



1. **High Throughput**: Automatically maintains a "full" batch. As soon as a structure converges and leaves the queue, a new one is inserted, keeping the GPU busy.↳
2. **Memory Safe**: Strictly controls GPU memory usage via `max_edges_per_batch` to prevent OOM errors.↳
3. **Comprehensive Support**:↳
   - **Fixed Cell**: Optimize atomic positions only.↳
   - **Variable Cell**: Optimize both positions and cell vectors (automatically handles `FrechetCellFilter` dimension mismatch).↳
   - **Multi-head**: Support for multi-head MACE models (select specific output heads like `'DFT'`, `'CC'`).
4. **Smart I/O**:
   - Generates independent `.traj` files for each structure.
   - Supports custom filenames via `atoms.info['name']`.
   - Fixes the empty-file bug inherent in standard ASE optimizers when running custom loops.



### 🚀 Tutorial





#### 1. Basic Usage: Fixed Cell Relaxation



Ideal for standard relaxation tasks using pre-trained MACE models.

Python

```
from ase.io import read, write
from mace.calculators import MACECalculator
from batch_relaxer import BatchRelaxer

# 1. Load Model (GPU required)
# Multi-head Model Need head para in this step and relax step
calc = MACECalculator(model_paths='large_model.model', device='cuda', default_dtype="float32")

# 2. Load Data
atoms_list = read('init.xyz', index=':')

# 3. Initialize Relaxer
# max_edges_per_batch: Key parameter for memory control
# RTX 3090/4090 (24G): 20000 ~ 30000
# A100 (80G): 60000 ~ 80000
relaxer = BatchRelaxer(calc, max_edges_per_batch=30000)

# 4. Run
results = relaxer.relax(
    atoms_list,
    fmax=0.02,              # Convergence threshold (eV/A)
    relax_cell=False,       # Fixed cell
    trajectory_dir="trajs"  # Output directory for trajectories
)

# 5. Save Results
write('relaxed.xyz', results)
```



#### 2. Advanced: Variable Cell Relaxation



Use this to find the ground state of crystals or apply pressure. The code handles the dimension changes caused by the Filter automatically.↳

Python

```
# Option A: Enable at initialization (Applies to all subsequent runs)
relaxer = BatchRelaxer(calc, max_edges_per_batch=20000, relax_cell=True)
relaxer.relax(atoms_list)

# Option B: Enable dynamically at runtime (Overrides default)
relaxer = BatchRelaxer(calc, max_edges_per_batch=20000) # Default False
relaxer.relax(atoms_list, relax_cell=True, fmax=0.01)   # Enabled for this run
```



#### 3. Expert: Multi-head Selection



If your model predicts multiple potential energy surfaces (PES), you can specify which one to optimize against.

Python

```
# Assume model has heads: ['Default', 'DFT', 'CC']
# Optimize using the 'CC' (Coupled Cluster) head
relaxer.relax(
    atoms_list, 
    head='CC',       # Specify Head
    fmax=0.02
)
```



#### 4. Trajectory Naming



The code supports smart naming. To get files named `water.traj` instead of `0.traj`, set the info dictionary.

Python

```
# Set custom names
atoms_list[0].info['name'] = "water_molecule"
atoms_list[1].info['ID'] = "structure_1024"

relaxer.relax(atoms_list, trajectory_dir="results")

# Output:
# results/water_molecule.traj
# results/structure_1024.traj
```

------



### ⚙️ Parameters



| Parameter             | Description                                                  | Recommended                           |
| --------------------- | ------------------------------------------------------------ | ------------------------------------- |
| `max_edges_per_batch` | **Critical**. Max total edges in the current batch. Controls VRAM usage. | 20k-30k (24G VRAM) 60k-80k (80G VRAM) |
| `fmax`                | Max force convergence criteria (eV/A). Avoid values < 0.01 for MLIPs. | 0.01 - 0.05                           |
| `relax_cell`          | Whether to relax unit cell vectors. Applies `FrechetCellFilter`. | True / False                          |
| `head`                | Output head name for multi-head models.                      | None or 'HeadName'                    |
| `trajectory_dir`      | Directory to save individual `.traj` files.                  | Path string                           |
| `save_log_file`       | Path to save detailed execution logs.                        | "relax.log"                           |
| `inplace`             | Whether to modify the input atoms list in-place.             | True                                  |

