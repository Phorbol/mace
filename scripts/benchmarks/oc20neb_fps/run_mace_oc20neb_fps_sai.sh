#!/usr/bin/env bash
#SBATCH --job-name=mace-oc20neb-fps
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
source /opt/envs/anaconda3.env
conda activate "${MACE_CONDA_ENV:-mace_develop}"

cd /home/sjtu-caoxiaoming/gengjianrui/trae-research-code/mace

DATA_ROOT="${DATA_ROOT:-runs/oc20neb_fps_extxyz}"
RUN_ROOT="${RUN_ROOT:-runs/oc20neb_fps}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/train.extxyz}"
VALID_FILE="${VALID_FILE:-${DATA_ROOT}/valid.extxyz}"
TEST_FILE="${TEST_FILE:-${VALID_FILE}}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/models" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/results"

BATCH_SIZE="${BATCH_SIZE:-8}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-8}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-32}"
EVAL_INTERVAL="${EVAL_INTERVAL:-8}"

CUEQ_CONV_FUSION_FLAG=(--cueq_conv_fusion)
if [ "${CUEQ_CONV_FUSION:-True}" != "True" ]; then
  CUEQ_CONV_FUSION_FLAG=(--no-cueq_conv_fusion)
fi

CUEQ_ARGS=(--enable_cueq="${ENABLE_CUEQ:-True}")
case "${CUEQ_PROFILE:-full}" in
  full)
    CUEQ_ARGS+=(
      --cueq_optimize_all
      --cueq_optimize_linear
      --cueq_optimize_channelwise
      --cueq_optimize_symmetric
      --cueq_optimize_fctp
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
  off)
    CUEQ_ARGS=(--enable_cueq=False)
    CUEQ_CONV_FUSION_FLAG=(--no-cueq_conv_fusion)
    ;;
  *)
    echo "Unknown CUEQ_PROFILE=${CUEQ_PROFILE}. Expected full, minus_linear, or off." >&2
    exit 2
    ;;
esac

EDGE_FORCE_COMPILE_ARGS=()
if [ "${EDGE_FORCE_COMPILE:-False}" = "True" ]; then
  EDGE_FORCE_GRAPH_FLAG=(--no-edge_force_compile_graph)
  if [ "${EDGE_FORCE_GRAPH:-True}" = "True" ]; then
    EDGE_FORCE_GRAPH_FLAG=(--edge_force_compile_graph)
  fi
  if [ "${EDGE_FORCE_GRAPH:-True}" = "True" ] && [ "${EDGE_FORCE_REQUIRE_INDUCTOR_ACK:-False}" != "True" ]; then
    echo "EDGE_FORCE_GRAPH=True enables experimental Inductor graph lowering." >&2
    echo "Set EDGE_FORCE_REQUIRE_INDUCTOR_ACK=True after parity/accuracy gating." >&2
    exit 2
  fi
  EDGE_FORCE_PARITY_STRICT_FLAG=(--edge_force_compile_parity_check_strict)
  if [ "${EDGE_FORCE_PARITY_CHECK_STRICT:-False}" != "True" ]; then
    EDGE_FORCE_PARITY_STRICT_FLAG=(--no-edge_force_compile_parity_check_strict)
  fi
  EDGE_FORCE_COMPILE_ARGS=(
    --edge_force_compile
    --edge_force_compile_tracing_mode="${EDGE_FORCE_TRACING_MODE:-symbolic}"
    "${EDGE_FORCE_GRAPH_FLAG[@]}"
    --edge_force_compile_dynamic
    --edge_force_compile_shape_padding
    --edge_force_compile_cache_policy="${EDGE_FORCE_CACHE_POLICY:-dynamic}"
    --no-edge_force_compile_cache_hit_gate
    --edge_force_compile_min_repeats="${EDGE_FORCE_MIN_REPEATS:-2}"
    --edge_force_compile_spherical_harmonics="${EDGE_FORCE_SH:-polynomial}"
    --edge_force_compile_force_gradient_mode="${EDGE_FORCE_GRADIENT_MODE:-edge}"
    --edge_force_compile_setup_gate="${EDGE_FORCE_SETUP_GATE:-strict}"
    --edge_force_compile_max_fusion_size="${EDGE_FORCE_MAX_FUSION_SIZE:-8}"
    --edge_force_compile_atol="${EDGE_FORCE_ATOL:-2e-2}"
    --edge_force_compile_rtol="${EDGE_FORCE_RTOL:-2e-4}"
    --edge_force_compile_parity_check_interval="${EDGE_FORCE_PARITY_CHECK_INTERVAL:-1000}"
    --edge_force_compile_parity_check_gradients
    "${EDGE_FORCE_PARITY_STRICT_FLAG[@]}"
    --edge_force_compile_fixed_probe_interval="${EDGE_FORCE_FIXED_PROBE_INTERVAL:-1000}"
    --no-edge_force_compile_fixed_probe_gradients
    --no-edge_force_compile_allow_fallback
  )
fi

HYBRID_MUON_ARGS=()
if [ "${OPTIMIZER:-adam}" = "hybrid_muon" ]; then
  HYBRID_MUON_ARGS=(
    --hybrid_muon_mode="${HYBRID_MUON_MODE:-2d}"
    --hybrid_muon_routing="${HYBRID_MUON_ROUTING:-mace}"
    --hybrid_muon_lr_factor="${HYBRID_MUON_LR_FACTOR:-0.1}"
    --hybrid_muon_weight_decay="${HYBRID_MUON_WEIGHT_DECAY:-0.0}"
  )
fi

python -m mace.cli.run_train \
  --name="${NAME:-oc20neb_fps_l1c64_adam_cueq}" \
  --train_file="${TRAIN_FILE}" \
  --valid_file="${VALID_FILE}" \
  --test_file="${TEST_FILE}" \
  --E0s=average \
  --energy_key=energy \
  --forces_key=forces \
  --model=ScaleShiftMACE \
  --num_interactions="${NUM_INTERACTIONS:-2}" \
  --num_channels="${NUM_CHANNELS:-64}" \
  --max_L="${MAX_L:-1}" \
  --correlation="${CORRELATION:-3}" \
  --r_max="${R_MAX:-6.0}" \
  --batch_size="${BATCH_SIZE}" \
  --valid_batch_size="${VALID_BATCH_SIZE}" \
  --max_num_epochs="${MAX_NUM_EPOCHS}" \
  --patience=999 \
  --eval_interval="${EVAL_INTERVAL}" \
  --error_table=PerAtomMAE \
  --default_dtype=float32 \
  $( [ "${TRAIN_TF32:-True}" = "True" ] && printf %s "--train_tf32" || printf %s "--no-train_tf32" ) \
  --train_amp_dtype="${TRAIN_AMP_DTYPE:-none}" \
  --device=cuda \
  --seed="${SEED:-456}" \
  --shuffle="${SHUFFLE:-True}" \
  "${CUEQ_ARGS[@]}" \
  "${CUEQ_CONV_FUSION_FLAG[@]}" \
  --optimizer="${OPTIMIZER:-adam}" \
  "${HYBRID_MUON_ARGS[@]}" \
  --scheduler="${SCHEDULER:-WSD}" \
  --lr_scheduler_interval="${LR_SCHEDULER_INTERVAL:-auto}" \
  --lr="${LR:-0.001}" \
  --weight_decay="${WEIGHT_DECAY:-0.001}" \
  --loss=weighted \
  --energy_weight="${ENERGY_WEIGHT:-20.0}" \
  --forces_weight="${FORCES_WEIGHT:-20.0}" \
  --lr_wsd_warmup_steps="${LR_WSD_WARMUP_STEPS:-0}" \
  --lr_wsd_warmup_ratio="${LR_WSD_WARMUP_RATIO:-0.003}" \
  --lr_wsd_warmup_start_factor="${LR_WSD_WARMUP_START_FACTOR:-0.2}" \
  --lr_wsd_stop_lr_ratio="${LR_WSD_STOP_LR_RATIO:-1e-3}" \
  --lr_wsd_decay_phase_ratio="${LR_WSD_DECAY_PHASE_RATIO:-0.1}" \
  --lr_wsd_decay_type="${LR_WSD_DECAY_TYPE:-inverse_linear}" \
  "${EDGE_FORCE_COMPILE_ARGS[@]}" \
  --work_dir="${RUN_ROOT}" \
  --log_dir="${RUN_ROOT}/logs" \
  --model_dir="${RUN_ROOT}/models" \
  --checkpoints_dir="${RUN_ROOT}/checkpoints" \
  --results_dir="${RUN_ROOT}/results"
