#!/bin/bash
#SBATCH --job-name=MACE-EDGE-FORCE-EPOCH
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu

set -euo pipefail
source /opt/envs/anaconda3.env
conda activate "${MACE_EDGE_EPOCH_CONDA_ENV:-mace_develop}"
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}

cueq_profile="${MACE_EDGE_EPOCH_CUEQ_PROFILE:-minus_linear}"
cueq_args=()
extra_args=()
if [[ -n "${MACE_EDGE_EPOCH_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${MACE_EDGE_EPOCH_EXTRA_ARGS})
fi
if [[ "${MACE_EDGE_EPOCH_RESET_COMPILE_STATE:-False}" == "True" ]]; then
  extra_args+=(--edge-reset-compile-state)
fi
if [[ "${MACE_EDGE_EPOCH_CLEAR_CACHE_ON_MISS:-False}" == "True" ]]; then
  extra_args+=(--edge-clear-cache-on-miss)
fi
if [[ "${MACE_EDGE_EPOCH_SKIP_TRAINING_STEP:-False}" == "True" ]]; then
  extra_args+=(--skip-training-step)
fi
if [[ "${MACE_EDGE_EPOCH_DIAGNOSE_GATE_STATE:-False}" == "True" ]]; then
  extra_args+=(--edge-diagnose-gate-state)
fi
case "${cueq_profile}" in
  full)
    cueq_args+=(--enable-cueq --cueq-optimize-all)
    ;;
  minus_linear)
    cueq_args+=(
      --enable-cueq
      --no-cueq-optimize-all
      --cueq-optimize-channelwise
      --cueq-optimize-fctp
      --cueq-optimize-symmetric
      --no-cueq-optimize-linear
    )
    ;;
  none)
    cueq_args+=(--no-enable-cueq)
    ;;
  *)
    echo "Unknown MACE_EDGE_EPOCH_CUEQ_PROFILE=${cueq_profile}" >&2
    exit 2
    ;;
esac

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/profile_edge_force_training_epochs.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0:64}" \
  --batch-size "${MACE_EDGE_EPOCH_BATCH_SIZE:-32}" \
  --epochs "${MACE_EDGE_EPOCH_EPOCHS:-2}" \
  --device cuda \
  --optimizers "${MACE_EDGE_EPOCH_OPTIMIZERS:-adam,hybrid_muon}" \
  --modes "${MACE_EDGE_EPOCH_MODES:-position_eager,edge_compile}" \
  --hidden-channels "${MACE_EDGE_EPOCH_HIDDEN_CHANNELS:-64}" \
  --max-ell "${MACE_EDGE_EPOCH_MAX_ELL:-2}" \
  --num-interactions "${MACE_EDGE_EPOCH_NUM_INTERACTIONS:-2}" \
  --correlation "${MACE_EDGE_EPOCH_CORRELATION:-3}" \
  --edge-compile-mode "${MACE_EDGE_EPOCH_COMPILE_MODE:-default}" \
  --output "${MACE_EDGE_EPOCH_OUTPUT:-edge_force_training_epochs_${SLURM_JOB_ID}.json}" \
  "${cueq_args[@]}" \
  "${extra_args[@]}"
