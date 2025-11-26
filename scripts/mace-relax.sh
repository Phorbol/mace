#!/bin/bash
#SBATCH --job-name=MACE-relax
#SBATCH --partition=8V100V0
#SBATCH --nodes=2
#SBATCH --ntasks=8          # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=4   # Specify the GPUs-per-node
#SBATCH --qos=rush-8gpu # Depending on your needs [Priority: rush-4gpu = rush-8gpu > improper-gpu > huge-gpu]

#export OMP_NUM_THREADS=8

nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
source /opt/envs/anaconda3.env
conda activate mace_env
srun python run_relax_mace.py