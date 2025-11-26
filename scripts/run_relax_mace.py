import os
import torch
from ase.io import read, write
from mace.calculators import MACECalculator
from mace.calculators.batch_relaxer import BatchRelaxer

# ================= 1. 多卡环境感知 =================
def setup_distributed_env():
    # 尝试读取 Slurm 环境变量
    # SLURM_PROCID: 全局进程号 (0 到 Total_GPUs - 1)
    # SLURM_NTASKS: 总进程数 (Total_GPUs)
    # SLURM_LOCALID: 当前节点内的进程号 (0 到 GPUs_per_Node - 1)

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    # 也可以兼容 torch.distributed.launch (如果不用 Slurm)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    print(f"Process: Global Rank {rank}/{world_size}, Local Rank {local_rank}, Device: cuda:{local_rank}")
    return rank, world_size, local_rank

# ================= 2. 主逻辑 =================
def main():
    # 获取环境信息
    rank, world_size, local_rank = setup_distributed_env()

    # 绑定当前进程到指定 GPU
    device = f"cuda:{local_rank}"

    # 1. 加载所有数据 (每个进程都读取，开销很小；如果数据极大建议分文件读)
    # 假设 all_atoms 有 1000 个
    all_atoms = read("/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz", index=":")

    # 2. 数据切分 (Data Sharding)
    # 只有属于当前 rank 的数据才会被处理
    # 语法: list[start:end:step]
    # 例如 8 卡:
    # Rank 0 取: 0, 8, 16...
    # Rank 1 取: 1, 9, 17...
    my_atoms = all_atoms[rank::world_size]

    print(f"[Rank {rank}] Assigned {len(my_atoms)} structures.")

    if len(my_atoms) == 0:
        print(f"[Rank {rank}] No data to process, exiting.")
        return

    # 3. 初始化模型 (加载到对应的 device)
    calc = MACECalculator(
        model_paths='/home/sjtu-caoxiaoming/gengjianrui/.cache/mace/mace-omat-0-small.model',
        device=device,  # <--- 关键：使用 local_rank 对应的设备
        default_dtype="float32"
    )

    # 4. 初始化 BatchRelaxer
    relaxer = BatchRelaxer(calc, max_edges_per_batch=40000, device=device)

    # 5. 设置独立的输出目录，防止文件冲突
    # 建议每个 Rank 写到不同的子文件夹，或者不同的文件名
    output_dir = f"trajs_rank_{rank}"
    os.makedirs(output_dir, exist_ok=True)

    # 6. 开始运行
    relaxed_results = relaxer.relax(
        my_atoms,
        fmax=0.02,
        trajectory_dir=output_dir, # 每个 Rank 写自己的文件夹
        save_log_file=f"log_rank_{rank}.txt" # 每个 Rank 写自己的日志
    )

    # 7. 保存该分片的结果
    write(f"relaxed_rank_{rank}.xyz", relaxed_results)
    print(f"[Rank {rank}] Finished.")

if __name__ == "__main__":
    main()