#!/bin/bash
#SBATCH --job-name=MACE-STEP-PROFILE
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu

set -euo pipefail
source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}
python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/profile_training_step_phases.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0:32}" \
  --warmup "${MACE_STEP_PROFILE_WARMUP:-3}" \
  --repeats "${MACE_STEP_PROFILE_REPEATS:-10}" \
  --output "${MACE_STEP_PROFILE_OUTPUT:-training_step_phase_profile_${SLURM_JOB_ID}.json}"
