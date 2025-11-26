# MACE Batch Relaxer (High-Performance GPU Optimization)





------



## MACE 批量结构优化器 (高性能 GPU 优化)



这是一个基于 [MACE](https://github.com/ACEsuit/mace) 和 [ASE](https://wiki.fysik.dtu.dk/ase/) 的高通量结构优化工具。它专为 GPU 设计，通过动态批处理（Dynamic Batching）技术，将不同大小、不同收敛速度的原子结构混合并行计算，最大化 GPU 利用率。



### ✨ 核心特性



1. **极速吞吐 (Dynamic Batching)**: 自动维护一个“满载”的批次。当某个结构收敛移出后，立即填入新结构，确保持续跑满 GPU。
2. **显存安全**: 通过 `max_edges_per_batch` 参数精准控制显存占用，防止 OOM (Out of Memory)。
3. **全功能支持**:
   - **固定晶胞 (Fixed Cell)**: 仅优化原子坐标。
   - **变胞优化 (Variable Cell)**: 同时优化原子坐标和晶胞参数 (自动处理 `FrechetCellFilter` 维度问题)。
   - **多头支持 (Multi-head)**: 支持 MACE 多头模型，可指定特定输出头 (如 `'DFT'`, `'CC'`)。
4. **智能 I/O**:
   - 自动生成独立的 `.traj` 轨迹文件。
   - 支持通过 `atoms.info['name']` 自定义文件名。
   - 修复了 ASE 默认优化器产生的空文件 Bug。



### 🛠️ 安装依赖



确保你的环境已安装以下库：

Bash

```
pip install mace-torch ase torch tqdm numpy
```



### 🚀 使用教程





#### 1. 基础用法：固定晶胞优化



这是最常用的场景，适用于由预训练 MACE 模型进行的常规结构弛豫。

Python

```
from ase.io import read, write
from mace.calculators import MACECalculator
from batch_relaxer import BatchRelaxer

# 1. 加载模型 (必须使用 GPU)
# 多头模型在这一步就需要指定head
calc = MACECalculator(model_paths='large_model.model', device='cuda', default_dtype="float32")

# 2. 读取数据
atoms_list = read('init.xyz', index=':')

# 3. 初始化优化器
# max_edges_per_batch: 控制显存占用的核心参数
# RTX 3090/4090 (24G) 推荐: 20000 ~ 30000
# A100 (80G) 推荐: 60000 ~ 80000
relaxer = BatchRelaxer(calc, max_edges_per_batch=30000)

# 4. 开始运行
results = relaxer.relax(
    atoms_list,
    fmax=0.02,              # 收敛阈值 (eV/A)
    relax_cell=False,       # 固定晶胞
    trajectory_dir="trajs"  # 轨迹保存目录
)

# 5. 保存结果
write('relaxed.xyz', results)
```



#### 2. 进阶用法：变胞优化 (Variable Cell)



如果你需要寻找晶体的基态结构或施加压力，需要开启变胞优化。代码会自动处理 Filter 带来的维度变化。

Python

```
# 方法 A: 在初始化时开启 (所有后续任务默认变胞)
relaxer = BatchRelaxer(calc, max_edges_per_batch=20000, relax_cell=True)
relaxer.relax(atoms_list)

# 方法 B: 在运行时动态开启 (覆盖默认设置)
relaxer = BatchRelaxer(calc, max_edges_per_batch=20000) # 默认 False
relaxer.relax(atoms_list, relax_cell=True, fmax=0.01)   # 本次运行开启
```



#### 3. 专家用法：多头模型选择 (Multi-head)



如果你的模型输出了多个势能面（例如同时预测了 DFT 和 Coupled-Cluster 能量），你可以指定优化目标。

Python

```
# 假设模型包含 ['Default', 'DFT', 'CC']
# 指定使用 'CC' (Coupled Cluster) 的势能面进行优化
relaxer.relax(
    atoms_list, 
    head='CC',       # 指定 Head
    fmax=0.02
)
```



#### 4. 轨迹文件的命名与管理



代码支持智能命名。如果你希望输出的轨迹文件不仅仅是 `0.traj`, `1.traj`，可以在输入的 Atoms 对象中设置信息。

Python

```
# 设置自定义名字
atoms_list[0].info['name'] = "water_molecule"
atoms_list[1].info['ID'] = "structure_1024"

relaxer.relax(atoms_list, trajectory_dir="results")

# 输出结果:
# results/water_molecule.traj
# results/structure_1024.traj
```

------



### ⚙️ 参数详解



| **参数**              | **说明**                                                     | **推荐值**                          |
| --------------------- | ------------------------------------------------------------ | ----------------------------------- |
| `max_edges_per_batch` | **最重要参数**。当前 Batch 中所有结构的总边数上限。决定显存占用。 | 20k-30k (24G显存) 60k-80k (80G显存) |
| `fmax`                | 最大力收敛标准 (eV/A)。建议不要低于 0.01，否则 MLIP 可能难以收敛。 | 0.01 - 0.05                         |
| `relax_cell`          | 是否优化晶胞 (True/False)。会自动应用 `FrechetCellFilter`。  | 根据物理需求                        |
| `head`                | 多头模型的输出头名称。如果模型只有单头，无需设置。           | None 或 'HeadName'                  |
| `trajectory_dir`      | 轨迹保存文件夹。若为 `None` 则不保存过程。                   | 任意路径字符串                      |
| `save_log_file`       | 保存详细运行日志的文件路径。                                 | "relax.log"                         |
| `inplace`             | 是否直接修改输入的 atoms 对象列表。                          | True                                |

------

