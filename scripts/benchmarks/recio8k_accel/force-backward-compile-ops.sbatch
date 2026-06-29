#!/bin/bash
#SBATCH --job-name=MACE-FORCE-COMPILE-OPS
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

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/probe_force_backward_compile_ops.py \
  --device "${MACE_FORCE_COMPILE_OPS_DEVICE:-cuda}" \
  --cases "${MACE_FORCE_COMPILE_OPS_CASES:-all}" \
  --dtype "${MACE_FORCE_COMPILE_OPS_DTYPE:-float32}" \
  --compile-mode "${MACE_FORCE_COMPILE_OPS_MODE:-default}" \
  ${MACE_FORCE_COMPILE_OPS_DISABLE_DONATED_BUFFER:+--disable-donated-buffer} \
  --warmup "${MACE_FORCE_COMPILE_OPS_WARMUP:-2}" \
  --repeats "${MACE_FORCE_COMPILE_OPS_REPEATS:-5}" \
  --output "${MACE_FORCE_COMPILE_OPS_OUTPUT:-force_backward_compile_ops_${SLURM_JOB_ID}.json}"
