#!/bin/bash
#SBATCH --job-name=MACE-FORCE-OPS
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=flood-1o2gpu

set -euo pipefail
source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}
EXTRA_ARGS=()
if [[ "${MACE_FORCE_OP_PROFILE_TRAIN_COMPILE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--train-compile)
fi
EXTRA_ARGS+=(--train-compile-mode "${MACE_FORCE_OP_PROFILE_TRAIN_COMPILE_MODE:-default}")
if [[ "${MACE_FORCE_OP_PROFILE_TRAIN_COMPILE_FULLGRAPH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--train-compile-fullgraph)
fi
if [[ "${MACE_FORCE_OP_PROFILE_TRAIN_COMPILE_ALLOW_FALLBACK:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-train-compile-allow-fallback)
fi
if [[ "${MACE_FORCE_OP_PROFILE_RECORD_SHAPES:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--record-shapes)
fi
if [[ "${MACE_FORCE_OP_PROFILE_GROUP_BY_INPUT_SHAPE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--group-by-input-shape)
fi
if [[ "${MACE_FORCE_OP_PROFILE_MEMORY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-memory)
fi
python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/profile_force_ops.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0:32}" \
  --modes "${MACE_FORCE_OP_PROFILE_MODES:-energy_forward,force_loss_backward}" \
  --warmup "${MACE_FORCE_OP_PROFILE_WARMUP:-2}" \
  --top-k "${MACE_FORCE_OP_PROFILE_TOP_K:-25}" \
  --output "${MACE_FORCE_OP_PROFILE_OUTPUT:-force_op_profile_${SLURM_JOB_ID}.json}" \
  "${EXTRA_ARGS[@]}"
