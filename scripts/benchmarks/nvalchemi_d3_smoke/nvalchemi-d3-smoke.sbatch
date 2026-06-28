#!/bin/bash
#SBATCH --job-name=MACE-D3-SMOKE
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=improper-gpu

set -euo pipefail

nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &

source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/nvalchemi_d3_smoke/run_nvalchemi_d3_smoke.py --repeat 50
