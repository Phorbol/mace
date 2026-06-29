#!/bin/bash
#SBATCH --job-name=MACE-COMPILE-PROBE
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=improper-gpu

set -euo pipefail

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &
source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}

CUEQ_ARGS=()
if [[ "${MACE_PROBE_ENABLE_CUEQ:-1}" == "1" ]]; then
  CUEQ_ARGS+=(--enable-cueq)
fi

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/recio8k_accel/probe_training_compile.py \
  --xyz "${RECIO_XYZ:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}" \
  --indices "${RECIO_INDICES:-0}" \
  --device cuda \
  --warmup "${MACE_PROBE_WARMUP:-1}" \
  --repeats "${MACE_PROBE_REPEATS:-5}" \
  --hidden-channels "${MACE_PROBE_HIDDEN_CHANNELS:-32}" \
  --max-ell "${MACE_PROBE_MAX_ELL:-1}" \
  --num-interactions "${MACE_PROBE_NUM_INTERACTIONS:-2}" \
  --correlation "${MACE_PROBE_CORRELATION:-2}" \
  --output "training_compile_probe_${SLURM_JOB_ID}.json" \
  "${CUEQ_ARGS[@]}"
