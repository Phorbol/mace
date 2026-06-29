#!/usr/bin/env bash
#SBATCH --job-name=mace-edge-cache
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=runs/recio8k_edge_cache/%x-%j.out
#SBATCH --error=runs/recio8k_edge_cache/%x-%j.err

set -euo pipefail

source /opt/envs/anaconda3.env
conda activate mace_env

cd /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace

RUN_ROOT="${RUN_ROOT:-runs/recio8k_edge_cache}"
TRAIN_FILE="${TRAIN_FILE:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/models" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results"

python -m mace.cli.run_train \
  --name="${NAME:-recio8k_edge_cache_repeat_smoke}" \
  --train_file="${TRAIN_FILE}" \
  --valid_fraction=0.05 \
  --test_file="${TRAIN_FILE}" \
  --E0s=average \
  --model=ScaleShiftMACE \
  --num_interactions=2 \
  --num_channels="${NUM_CHANNELS:-64}" \
  --max_L="${MAX_L:-1}" \
  --correlation=3 \
  --r_max=5.0 \
  --batch_size="${BATCH_SIZE:-32}" \
  --valid_batch_size="${VALID_BATCH_SIZE:-32}" \
  --max_num_epochs="${MAX_NUM_EPOCHS:-2}" \
  --patience=999 \
  --eval_interval=1 \
  --error_table=PerAtomMAE \
  --default_dtype=float32 \
  --device=cuda \
  --seed=123 \
  --shuffle="${SHUFFLE:-False}" \
  --enable_cueq=True \
  --cueq_optimize_all \
  --no-cueq_optimize_linear \
  --cueq_optimize_channelwise \
  --cueq_optimize_symmetric \
  --cueq_optimize_fctp \
  --cueq_conv_fusion \
  --optimizer=hybrid_muon \
  --edge_force_compile \
  --edge_force_compile_tracing_mode=real \
  --edge_force_compile_cache_policy="${EDGE_FORCE_CACHE_POLICY:-repeat_only}" \
  --edge_force_compile_min_repeats="${EDGE_FORCE_MIN_REPEATS:-2}" \
  --edge_force_compile_mode=default \
  --edge_force_compile_dynamic=True \
  --edge_force_compile_graph=True \
  --no-edge_force_compile_allow_fallback \
  --work_dir="${RUN_ROOT}" \
  --log_dir="${RUN_ROOT}/logs" \
  --model_dir="${RUN_ROOT}/models" \
  --checkpoints_dir="${RUN_ROOT}/checkpoints" \
  --results_dir="${RUN_ROOT}/results"
