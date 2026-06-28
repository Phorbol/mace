#!/bin/bash
#SBATCH --job-name=MACE-RECIO8K-ACCEL
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=__QOS__

set -euo pipefail

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-${SLURM_JOB_ID}.log &
source /opt/envs/anaconda3.env
conda activate mace_env
export PYTHONPATH=/home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace:${PYTHONPATH:-}

python -m mace.cli.run_train --config=config.yaml
