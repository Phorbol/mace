#!/bin/bash
#SBATCH --job-name=MACE-D3-BENCH
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

ARGS=(
    --repeat "${D3_REPEAT:-100}"
    --max-structures "${D3_MAX_STRUCTURES:-8}"
    --json-output "${D3_JSON_OUTPUT:-nvalchemi_d3_benchmark_job-${SLURM_JOB_ID}.json}"
)

if [[ -n "${D3_XYZ:-}" ]]; then
    ARGS+=(--xyz "${D3_XYZ}")
fi
if [[ -n "${D3_INDICES:-}" ]]; then
    ARGS+=(--indices "${D3_INDICES}")
fi
if [[ -n "${D3_SUPERCELL:-}" ]]; then
    ARGS+=(--supercell "${D3_SUPERCELL}")
fi
if [[ "${D3_COMPARE_TORCH_DFTD:-0}" == "1" ]]; then
    ARGS+=(--compare-torch-dftd)
fi

python /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace/scripts/benchmarks/nvalchemi_d3_smoke/run_nvalchemi_d3_benchmark.py "${ARGS[@]}"
