#!/usr/bin/env bash
#SBATCH --job-name=mace-edge-cache
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=runs/recio8k_edge_cache/%x-%j.out
#SBATCH --error=runs/recio8k_edge_cache/%x-%j.err

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
source /opt/envs/anaconda3.env
conda activate "${MACE_CONDA_ENV:-mace_develop}"

cd /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace

RUN_ROOT="${RUN_ROOT:-runs/recio8k_edge_cache}"
TRAIN_FILE="${TRAIN_FILE:-/home/sjtu-caoxiaoming/gengjianrui/test/mace/RECIO/8k/train.xyz}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/models" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results"

# V100 RECIO-8k benchmark defaults follow the current accepted safe compile
# path: dynamic symbolic cache with FX-only force/backward tracing. Full
# Inductor graph lowering remains experimental until it passes real-batch
# parity and RECIO accuracy gates.
BATCH_SIZE="${BATCH_SIZE:-16}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-32}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-10}"
EDGE_FORCE_BUCKET_ATOMS="${EDGE_FORCE_BUCKET_ATOMS:-512,768}"
EDGE_FORCE_BUCKET_EDGES="${EDGE_FORCE_BUCKET_EDGES:-8192,16384}"

CUEQ_CONV_FUSION_FLAG=(--cueq_conv_fusion)
if [ "${CUEQ_CONV_FUSION:-True}" != "True" ]; then
  CUEQ_CONV_FUSION_FLAG=(--no-cueq_conv_fusion)
fi

CUEQ_ARGS=(--enable_cueq="${ENABLE_CUEQ:-False}")
case "${CUEQ_PROFILE:-safe}" in
  safe)
    CUEQ_ARGS+=(
      --no-cueq_optimize_all
      --no-cueq_optimize_linear
      --cueq_optimize_channelwise
      --cueq_optimize_symmetric
      --no-cueq_optimize_fctp
    )
    ;;
  minus_linear)
    CUEQ_ARGS+=(
      --no-cueq_optimize_all
      --no-cueq_optimize_linear
      --cueq_optimize_channelwise
      --cueq_optimize_symmetric
      --cueq_optimize_fctp
    )
    ;;
  full)
    CUEQ_ARGS+=(
      --cueq_optimize_all
      --cueq_optimize_linear
      --cueq_optimize_channelwise
      --cueq_optimize_symmetric
      --cueq_optimize_fctp
    )
    ;;
  linear)
    CUEQ_ARGS+=(
      --cueq_optimize_all
      --cueq_optimize_linear
      --no-cueq_optimize_channelwise
      --no-cueq_optimize_symmetric
      --no-cueq_optimize_fctp
    )
    ;;
  off)
    CUEQ_ARGS=(--enable_cueq=False)
    CUEQ_CONV_FUSION_FLAG=(--no-cueq_conv_fusion)
    ;;
  *)
    echo "Unknown CUEQ_PROFILE=${CUEQ_PROFILE}. Expected safe, minus_linear, full, linear, or off." >&2
    exit 2
    ;;
esac

EDGE_FORCE_COMPILE_ARGS=()
if [ "${EDGE_FORCE_COMPILE:-True}" = "True" ]; then
  EDGE_FORCE_DYNAMIC_FLAG=(--edge_force_compile_dynamic)
  if [ "${EDGE_FORCE_DYNAMIC:-True}" != "True" ]; then
    EDGE_FORCE_DYNAMIC_FLAG=(--no-edge_force_compile_dynamic)
  fi
  EDGE_FORCE_GRAPH_FLAG=(--no-edge_force_compile_graph)
  if [ "${EDGE_FORCE_GRAPH:-False}" = "True" ]; then
    EDGE_FORCE_GRAPH_FLAG=(--edge_force_compile_graph)
  fi
  if [ "${EDGE_FORCE_GRAPH:-False}" = "True" ] && [ "${EDGE_FORCE_REQUIRE_INDUCTOR_ACK:-False}" != "True" ]; then
    echo "EDGE_FORCE_GRAPH=True enables experimental Inductor graph lowering." >&2
    echo "Set EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True after parity/accuracy gating." >&2
    exit 2
  fi
  EDGE_FORCE_SHAPE_PADDING_FLAG=(--edge_force_compile_shape_padding)
  if [ "${EDGE_FORCE_SHAPE_PADDING:-True}" != "True" ]; then
    EDGE_FORCE_SHAPE_PADDING_FLAG=(--no-edge_force_compile_shape_padding)
  fi
  EDGE_FORCE_CACHE_HIT_GATE_FLAG=(--no-edge_force_compile_cache_hit_gate)
  if [ "${EDGE_FORCE_CACHE_HIT_GATE:-False}" = "True" ]; then
    EDGE_FORCE_CACHE_HIT_GATE_FLAG=(--edge_force_compile_cache_hit_gate)
  fi
  EDGE_FORCE_PARITY_GRADIENTS_FLAG=(--edge_force_compile_parity_check_gradients)
  if [ "${EDGE_FORCE_PARITY_CHECK_GRADIENTS:-True}" != "True" ]; then
    EDGE_FORCE_PARITY_GRADIENTS_FLAG=(--no-edge_force_compile_parity_check_gradients)
  fi
  EDGE_FORCE_PARITY_STRICT_FLAG=(--edge_force_compile_parity_check_strict)
  if [ "${EDGE_FORCE_PARITY_CHECK_STRICT:-True}" != "True" ]; then
    EDGE_FORCE_PARITY_STRICT_FLAG=(--no-edge_force_compile_parity_check_strict)
  fi
  EDGE_FORCE_FIXED_PROBE_GRADIENTS_FLAG=(--no-edge_force_compile_fixed_probe_gradients)
  if [ "${EDGE_FORCE_FIXED_PROBE_GRADIENTS:-False}" = "True" ]; then
    EDGE_FORCE_FIXED_PROBE_GRADIENTS_FLAG=(--edge_force_compile_fixed_probe_gradients)
  fi
  EDGE_FORCE_FIXED_PROBE_STRICT_FLAG=(--no-edge_force_compile_fixed_probe_strict)
  if [ "${EDGE_FORCE_FIXED_PROBE_STRICT:-False}" = "True" ]; then
    EDGE_FORCE_FIXED_PROBE_STRICT_FLAG=(--edge_force_compile_fixed_probe_strict)
  fi
  EDGE_FORCE_COMPILE_ARGS=(
    --edge_force_compile
    "${EDGE_FORCE_CACHE_HIT_GATE_FLAG[@]}"
    --edge_force_compile_parity_check_interval="${EDGE_FORCE_PARITY_CHECK_INTERVAL:-0}"
    "${EDGE_FORCE_PARITY_GRADIENTS_FLAG[@]}"
    "${EDGE_FORCE_PARITY_STRICT_FLAG[@]}"
    --edge_force_compile_fixed_probe_interval="${EDGE_FORCE_FIXED_PROBE_INTERVAL:-0}"
    "${EDGE_FORCE_FIXED_PROBE_GRADIENTS_FLAG[@]}"
    "${EDGE_FORCE_FIXED_PROBE_STRICT_FLAG[@]}"
    --edge_force_compile_tracing_mode="${EDGE_FORCE_TRACING_MODE:-symbolic}"
    --edge_force_compile_atol="${EDGE_FORCE_ATOL:-2e-2}"
    --edge_force_compile_rtol="${EDGE_FORCE_RTOL:-2e-4}"
    --edge_force_compile_cache_policy="${EDGE_FORCE_CACHE_POLICY:-dynamic}"
    --edge_force_compile_min_repeats="${EDGE_FORCE_MIN_REPEATS:-2}"
    --edge_force_compile_bucket_atoms="${EDGE_FORCE_BUCKET_ATOMS}"
    --edge_force_compile_bucket_edges="${EDGE_FORCE_BUCKET_EDGES}"
    --edge_force_compile_bucket_margin="${EDGE_FORCE_BUCKET_MARGIN:-0.0}"
    --edge_force_compile_mode=default
    --edge_force_compile_max_fusion_size="${EDGE_FORCE_MAX_FUSION_SIZE:-8}"
    --edge_force_compile_spherical_harmonics="${EDGE_FORCE_SH:-polynomial}"
    --edge_force_compile_force_gradient_mode="${EDGE_FORCE_GRADIENT_MODE:-edge}"
    --edge_force_compile_setup_gate="${EDGE_FORCE_SETUP_GATE:-strict}"
    "${EDGE_FORCE_DYNAMIC_FLAG[@]}"
    "${EDGE_FORCE_GRAPH_FLAG[@]}"
    "${EDGE_FORCE_SHAPE_PADDING_FLAG[@]}"
    --no-edge_force_compile_allow_fallback
  )
fi

python -m mace.cli.run_train \
  --name="${NAME:-recio8k_edge_cache_repeat_smoke}" \
  --train_file="${TRAIN_FILE}" \
  --valid_fraction=0.05 \
  --test_file="${TRAIN_FILE}" \
  --E0s=average \
  --energy_key=energy \
  --forces_key=forces \
  --model=ScaleShiftMACE \
  --num_interactions=2 \
  --num_channels="${NUM_CHANNELS:-64}" \
  --max_L="${MAX_L:-1}" \
  --correlation=3 \
  --r_max=5.0 \
  --batch_size="${BATCH_SIZE}" \
  --valid_batch_size="${VALID_BATCH_SIZE}" \
  --max_num_epochs="${MAX_NUM_EPOCHS}" \
  --patience=999 \
  --eval_interval="${EVAL_INTERVAL:-1}" \
  --error_table=PerAtomMAE \
  --default_dtype=float32 \
  $( [ "${TRAIN_TF32:-False}" = "True" ] && printf %s "--train_tf32" || printf %s "--no-train_tf32" ) \
  --train_amp_dtype="${TRAIN_AMP_DTYPE:-none}" \
  --device=cuda \
  --seed="${SEED:-123}" \
  --shuffle="${SHUFFLE:-False}" \
  "${CUEQ_ARGS[@]}" \
  "${CUEQ_CONV_FUSION_FLAG[@]}" \
  --optimizer="${OPTIMIZER:-hybrid_muon}" \
  --scheduler="${SCHEDULER:-ReduceLROnPlateau}" \
  --lr_wsd_warmup_steps="${LR_WSD_WARMUP_STEPS:-0}" \
  --lr_wsd_warmup_ratio="${LR_WSD_WARMUP_RATIO:-0.03}" \
  --lr_wsd_warmup_start_factor="${LR_WSD_WARMUP_START_FACTOR:-0.1}" \
  --lr_wsd_stop_lr_ratio="${LR_WSD_STOP_LR_RATIO:-1e-3}" \
  --lr_wsd_decay_phase_ratio="${LR_WSD_DECAY_PHASE_RATIO:-0.1}" \
  --lr_wsd_decay_type="${LR_WSD_DECAY_TYPE:-inverse_linear}" \
  "${EDGE_FORCE_COMPILE_ARGS[@]}" \
  --work_dir="${RUN_ROOT}" \
  --log_dir="${RUN_ROOT}/logs" \
  --model_dir="${RUN_ROOT}/models" \
  --checkpoints_dir="${RUN_ROOT}/checkpoints" \
  --results_dir="${RUN_ROOT}/results"
