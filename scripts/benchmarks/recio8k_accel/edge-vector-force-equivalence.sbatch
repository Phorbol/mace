#!/bin/bash
#SBATCH --job-name=MACE-EDGE-FORCE-EQ
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

CUEQ_ARGS=()
if [[ "${MACE_EDGE_FORCE_ENABLE_CUEQ:-1}" == "1" ]]; then
  CUEQ_ARGS+=(--enable-cueq)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_CONV_FUSION:-1}" == "0" ]]; then
  CUEQ_ARGS+=(--no-cueq-conv-fusion)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_OPTIMIZE_ALL:-1}" == "0" ]]; then
  CUEQ_ARGS+=(--no-cueq-optimize-all)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_OPTIMIZE_LINEAR:-0}" == "1" ]]; then
  CUEQ_ARGS+=(--cueq-optimize-linear)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_OPTIMIZE_CHANNELWISE:-0}" == "1" ]]; then
  CUEQ_ARGS+=(--cueq-optimize-channelwise)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_OPTIMIZE_SYMMETRIC:-0}" == "1" ]]; then
  CUEQ_ARGS+=(--cueq-optimize-symmetric)
fi
if [[ "${MACE_EDGE_FORCE_CUEQ_OPTIMIZE_FCTP:-0}" == "1" ]]; then
  CUEQ_ARGS+=(--cueq-optimize-fctp)
fi

MAKE_FX_ARGS=()
if [[ "${MACE_EDGE_FORCE_MAKE_FX:-0}" == "1" ]]; then
  MAKE_FX_ARGS+=(--make-fx)
  MAKE_FX_ARGS+=(--make-fx-tracing-mode "${MACE_EDGE_FORCE_MAKE_FX_TRACING_MODE:-real}")
fi
if [[ "${MACE_EDGE_FORCE_STRIP_MAKE_FX_DETACH:-0}" == "1" ]]; then
  MAKE_FX_ARGS+=(--strip-make-fx-detach)
fi
if [[ "${MACE_EDGE_FORCE_COMPILE_MAKE_FX:-0}" == "1" ]]; then
  MAKE_FX_ARGS+=(--compile-make-fx)
  MAKE_FX_ARGS+=(--make-fx-compile-mode "${MACE_EDGE_FORCE_MAKE_FX_COMPILE_MODE:-default}")
fi
if [[ "${MACE_EDGE_FORCE_MAKE_FX_COMPILE_DYNAMIC:-1}" == "0" ]]; then
  MAKE_FX_ARGS+=(--no-make-fx-compile-dynamic)
fi

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/probe_edge_vector_force_equivalence.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0}" \
  --device "${MACE_EDGE_FORCE_DEVICE:-cuda}" \
  --hidden-channels "${MACE_EDGE_FORCE_HIDDEN_CHANNELS:-8}" \
  --max-ell "${MACE_EDGE_FORCE_MAX_ELL:-1}" \
  --num-interactions "${MACE_EDGE_FORCE_NUM_INTERACTIONS:-1}" \
  --correlation "${MACE_EDGE_FORCE_CORRELATION:-1}" \
  --atol "${MACE_EDGE_FORCE_ATOL:-1e-5}" \
  --rtol "${MACE_EDGE_FORCE_RTOL:-1e-4}" \
  --output "${MACE_EDGE_FORCE_OUTPUT:-edge_vector_force_equivalence_${SLURM_JOB_ID}.json}" \
  "${CUEQ_ARGS[@]}" \
  "${MAKE_FX_ARGS[@]}"
