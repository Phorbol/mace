#!/bin/bash
#SBATCH --job-name=MACE-EDGE-FORCE-STEP
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

cueq_profile="${MACE_EDGE_STEP_CUEQ_PROFILE:-minus_linear}"
cueq_args=()
extra_args=()
if [[ -n "${MACE_EDGE_STEP_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${MACE_EDGE_STEP_EXTRA_ARGS})
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
    echo "Unknown MACE_EDGE_STEP_CUEQ_PROFILE=${cueq_profile}" >&2
    exit 2
    ;;
esac

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/profile_edge_force_training_steps.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0:32}" \
  --device cuda \
  --optimizers "${MACE_EDGE_STEP_OPTIMIZERS:-adam,hybrid_muon}" \
  --modes "${MACE_EDGE_STEP_MODES:-position_eager,edge_eager,edge_compile}" \
  --hidden-channels "${MACE_EDGE_STEP_HIDDEN_CHANNELS:-64}" \
  --max-ell "${MACE_EDGE_STEP_MAX_ELL:-2}" \
  --num-interactions "${MACE_EDGE_STEP_NUM_INTERACTIONS:-2}" \
  --correlation "${MACE_EDGE_STEP_CORRELATION:-3}" \
  --warmup "${MACE_EDGE_STEP_WARMUP:-5}" \
  --repeats "${MACE_EDGE_STEP_REPEATS:-20}" \
  --edge-compile-mode "${MACE_EDGE_STEP_COMPILE_MODE:-default}" \
  --output "${MACE_EDGE_STEP_OUTPUT:-edge_force_training_steps_${SLURM_JOB_ID}.json}" \
  "${cueq_args[@]}" \
  "${extra_args[@]}"
