#!/bin/bash
#SBATCH --job-name=MACE-FORCE-MODES
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=rush-1o2gpu

set -euo pipefail
source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}
python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/profile_force_energy_modes.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0:32}" \
  --warmup "${MACE_FORCE_PROFILE_WARMUP:-10}" \
  --repeats "${MACE_FORCE_PROFILE_REPEATS:-30}" \
  --output "${MACE_FORCE_PROFILE_OUTPUT:-force_energy_mode_profile_${SLURM_JOB_ID}.json}"
