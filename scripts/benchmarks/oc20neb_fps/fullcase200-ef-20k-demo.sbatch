#!/usr/bin/env bash
#SBATCH --job-name=MACE-OC20NEB-EF20K
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=runs/oc20neb_fullcase200_ef_20k/%x-%j.out
#SBATCH --error=runs/oc20neb_fullcase200_ef_20k/%x-%j.err

set -euo pipefail

REPO_ROOT=${MACE_OC20NEB_REPO_ROOT:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel}
PYTHON_BIN=${MACE_OC20NEB_PYTHON:-/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
DATA_ROOT=${MACE_OC20NEB_DATA_ROOT:-${REPO_ROOT}/runs/oc20neb_fullcase200_fps_extxyz}
TRAIN_FILE=${MACE_OC20NEB_TRAIN_FILE:-${DATA_ROOT}/train.extxyz}
VALID_FILE=${MACE_OC20NEB_VALID_FILE:-${DATA_ROOT}/valid.extxyz}
TEST_FILE=${MACE_OC20NEB_TEST_FILE:-${VALID_FILE}}
RUN_ROOT=${MACE_OC20NEB_RUN_ROOT:-${REPO_ROOT}/runs/oc20neb_fullcase200_ef_20k/${SLURM_JOB_ID:-local}}
TARGET_STEPS=${MACE_OC20NEB_TARGET_STEPS:-20000}
BATCH_SIZE=${MACE_OC20NEB_BATCH_SIZE:-8}
VALID_BATCH_SIZE=${MACE_OC20NEB_VALID_BATCH_SIZE:-16}
SEED=${MACE_OC20NEB_SEED:-456}
EVAL_REPORTS=${MACE_OC20NEB_EVAL_REPORTS:-5}
CHECKPOINT_REPORTS=${MACE_OC20NEB_CHECKPOINT_REPORTS:-${EVAL_REPORTS}}
STAGE_TWO=${MACE_OC20NEB_STAGE_TWO:-True}
STAGE_TWO_FRACTION=${MACE_OC20NEB_STAGE_TWO_FRACTION:-0.75}
STAGE1_ENERGY_WEIGHT=${MACE_OC20NEB_STAGE1_ENERGY_WEIGHT:-1.0}
STAGE1_FORCES_WEIGHT=${MACE_OC20NEB_STAGE1_FORCES_WEIGHT:-100.0}
STAGE2_ENERGY_WEIGHT=${MACE_OC20NEB_STAGE2_ENERGY_WEIGHT:-100.0}
STAGE2_FORCES_WEIGHT=${MACE_OC20NEB_STAGE2_FORCES_WEIGHT:-1.0}
LOSS_PREFACTOR_SCHEDULE=${MACE_OC20NEB_LOSS_PREFACTOR_SCHEDULE:-step_linear}
CASES=${MACE_OC20NEB_CASES:-adamw,hybrid_muon,cueq_adamw,cueq_hybrid_muon}
BASE_OPTIMIZER=${MACE_OC20NEB_BASE_OPTIMIZER:-adamw}
ENFORCE_STATIC_SOURCE=${MACE_OC20NEB_ENFORCE_STATIC_SOURCE:-True}
COMPILE_BUCKET_ATOMS=${MACE_OC20NEB_COMPILE_BUCKET_ATOMS:-384,512,640,800}
COMPILE_BUCKET_EDGES=${MACE_OC20NEB_COMPILE_BUCKET_EDGES:-8192,12288,16384,24576,32768,40960}
COMPILE_BUCKET_MARGIN=${MACE_OC20NEB_COMPILE_BUCKET_MARGIN:-0}
COMPILE_MAX_CACHE_ENTRIES=${MACE_OC20NEB_COMPILE_MAX_CACHE_ENTRIES:-2}
COMPILE_SETUP_GATE=${MACE_OC20NEB_COMPILE_SETUP_GATE:-strict}
COMPILE_PARITY_GRADIENTS=${MACE_OC20NEB_COMPILE_PARITY_GRADIENTS:-False}
COMPILE_PARITY_CHECK_STRICT=${MACE_OC20NEB_COMPILE_PARITY_CHECK_STRICT:-False}
SCHEDULER=${MACE_OC20NEB_SCHEDULER:-WSD}
LR_SCHEDULER_INTERVAL=${MACE_OC20NEB_LR_SCHEDULER_INTERVAL:-step}
LR_WSD_WARMUP_RATIO=${MACE_OC20NEB_LR_WSD_WARMUP_RATIO:-0.03}
LR_WSD_STOP_LR_RATIO=${MACE_OC20NEB_LR_WSD_STOP_LR_RATIO:-0.001}
LR_WSD_DECAY_PHASE_RATIO=${MACE_OC20NEB_LR_WSD_DECAY_PHASE_RATIO:-0.1}
LR_WSD_DECAY_TYPE=${MACE_OC20NEB_LR_WSD_DECAY_TYPE:-inverse_linear}
LR=${MACE_OC20NEB_LR:-0.001}
WEIGHT_DECAY=${MACE_OC20NEB_WEIGHT_DECAY:-0.001}
HYBRID_MUON_MODE=${MACE_OC20NEB_HYBRID_MUON_MODE:-2d}
HYBRID_MUON_ROUTING=${MACE_OC20NEB_HYBRID_MUON_ROUTING:-mace}
HYBRID_MUON_TACE_MODULE_INCLUDE=${MACE_OC20NEB_HYBRID_MUON_TACE_MODULE_INCLUDE:-*}
HYBRID_MUON_TACE_MODULE_LR_SCALE=${MACE_OC20NEB_HYBRID_MUON_TACE_MODULE_LR_SCALE:-1.0}
HYBRID_MUON_LR_FACTOR=${MACE_OC20NEB_HYBRID_MUON_LR_FACTOR:-0.1}
HYBRID_MUON_STAGE_TWO_LR_FACTOR=${MACE_OC20NEB_HYBRID_MUON_STAGE_TWO_LR_FACTOR:-1.0}
HYBRID_MUON_STAGE_TWO_ROUTE=${MACE_OC20NEB_HYBRID_MUON_STAGE_TWO_ROUTE:-keep}
HYBRID_MUON_WEIGHT_DECAY=${MACE_OC20NEB_HYBRID_MUON_WEIGHT_DECAY:-0.0}
HYBRID_MUON_ADAM_VARIANT=${MACE_OC20NEB_HYBRID_MUON_ADAM_VARIANT:-adamw}
HYBRID_MUON_LR_SCALE_MODE=${MACE_OC20NEB_HYBRID_MUON_LR_SCALE_MODE:-original}
HYBRID_MUON_MATCH_RMS_COEFF=${MACE_OC20NEB_HYBRID_MUON_MATCH_RMS_COEFF:-0.18}
HYBRID_MUON_MAGMA_LITE=${MACE_OC20NEB_HYBRID_MUON_MAGMA_LITE:-False}
HYBRID_MUON_MAGMA_INITIAL_SCORE=${MACE_OC20NEB_HYBRID_MUON_MAGMA_INITIAL_SCORE:-0.5}
HYBRID_MUON_MAGMA_WARMUP_STEPS=${MACE_OC20NEB_HYBRID_MUON_MAGMA_WARMUP_STEPS:-0}
HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP=${MACE_OC20NEB_HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP:-False}

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Train extxyz not found: ${TRAIN_FILE}" >&2
  echo "Prepare it with scripts/benchmarks/oc20neb_fps/prepare_fullcase200_fps_extxyz.py" >&2
  exit 2
fi
if [[ ! -f "${VALID_FILE}" ]]; then
  echo "Valid extxyz not found: ${VALID_FILE}" >&2
  echo "Prepare it with scripts/benchmarks/oc20neb_fps/prepare_fullcase200_fps_extxyz.py" >&2
  exit 2
fi

SOURCE_GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || printf unknown)
SOURCE_GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || printf unknown)
source_tree_changed() {
  ! git diff --quiet --exit-code 2>/dev/null || ! git diff --cached --quiet --exit-code 2>/dev/null
}
source_guard_enabled() {
  case "${ENFORCE_STATIC_SOURCE}" in
    True|true|1|yes|YES|on|ON) return 0 ;;
    False|false|0|no|NO|off|OFF) return 1 ;;
    *)
      echo "Invalid MACE_OC20NEB_ENFORCE_STATIC_SOURCE=${ENFORCE_STATIC_SOURCE}; expected True or False" >&2
      exit 2
      ;;
  esac
}
assert_static_source() {
  local current_commit
  current_commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)
  if [[ "${current_commit}" != "${SOURCE_GIT_COMMIT}" ]]; then
    echo "Repository HEAD changed during job: ${SOURCE_GIT_COMMIT} -> ${current_commit}" >&2
    return 3
  fi
  if source_tree_changed; then
    echo "Repository tracked files changed during job; refusing to mix source versions" >&2
    return 3
  fi
}
if source_guard_enabled; then
  assert_static_source
fi

export TRAIN_FILE BATCH_SIZE TARGET_STEPS EVAL_REPORTS CHECKPOINT_REPORTS STAGE_TWO_FRACTION
read -r TRAIN_SIZE MAX_EPOCHS EVAL_INTERVAL EVAL_INTERVAL_UPDATES CHECKPOINT_INTERVAL_UPDATES START_STAGE_TWO_UPDATE < <("${PYTHON_BIN}" - <<'INNERPY'
import math
from ase.io import read
import os
train_file = os.environ["TRAIN_FILE"]
batch_size = int(os.environ["BATCH_SIZE"])
target_steps = int(os.environ["TARGET_STEPS"])
eval_reports = max(1, int(os.environ["EVAL_REPORTS"]))
checkpoint_reports = max(1, int(os.environ["CHECKPOINT_REPORTS"]))
stage_two_fraction = float(os.environ["STAGE_TWO_FRACTION"])
train_size = len(read(train_file, index=":"))
if train_size <= 0:
    raise SystemExit(f"empty training file: {train_file}")
max_epochs = max(1, math.ceil(target_steps * batch_size / train_size))
eval_interval = max(1, max_epochs // eval_reports)
eval_interval_updates = max(1, target_steps // eval_reports)
checkpoint_interval_updates = max(1, target_steps // checkpoint_reports)
start_stage_two_update = min(target_steps, max(1, math.floor(target_steps * stage_two_fraction)))
print(train_size, max_epochs, eval_interval, eval_interval_updates, checkpoint_interval_updates, start_stage_two_update)
INNERPY
)
LOSS_PREFACTOR_START_UPDATE=${MACE_OC20NEB_LOSS_PREFACTOR_START_UPDATE:-${START_STAGE_TWO_UPDATE}}
LOSS_PREFACTOR_END_UPDATE=${MACE_OC20NEB_LOSS_PREFACTOR_END_UPDATE:-${TARGET_STEPS}}

cat > "${RUN_ROOT}/manifest.json" <<JSON
{
  "batch_size": ${BATCH_SIZE},
  "valid_batch_size": ${VALID_BATCH_SIZE},
  "data_root": "${DATA_ROOT}",
  "eval_interval": ${EVAL_INTERVAL},
  "eval_interval_updates": ${EVAL_INTERVAL_UPDATES},
  "checkpoint_interval_updates": ${CHECKPOINT_INTERVAL_UPDATES},
  "max_num_epochs": ${MAX_EPOCHS},
  "max_num_updates": ${TARGET_STEPS},
  "python": "${PYTHON_BIN}",
  "repo_root": "${REPO_ROOT}",
  "target_steps": ${TARGET_STEPS},
  "stage_two": "${STAGE_TWO}",
  "stage_two_fraction": ${STAGE_TWO_FRACTION},
  "stage_two_start_update": ${START_STAGE_TWO_UPDATE},
  "loss_prefactor_schedule": "${LOSS_PREFACTOR_SCHEDULE}",
  "loss_prefactor_start_update": ${LOSS_PREFACTOR_START_UPDATE},
  "loss_prefactor_end_update": ${LOSS_PREFACTOR_END_UPDATE},
  "scheduler": "${SCHEDULER}",
  "lr_scheduler_interval": "${LR_SCHEDULER_INTERVAL}",
  "lr_wsd_warmup_ratio": ${LR_WSD_WARMUP_RATIO},
  "lr_wsd_stop_lr_ratio": ${LR_WSD_STOP_LR_RATIO},
  "lr_wsd_decay_phase_ratio": ${LR_WSD_DECAY_PHASE_RATIO},
  "lr_wsd_decay_type": "${LR_WSD_DECAY_TYPE}",
  "lr": ${LR},
  "weight_decay": ${WEIGHT_DECAY},
  "hybrid_muon_mode": "${HYBRID_MUON_MODE}",
  "hybrid_muon_routing": "${HYBRID_MUON_ROUTING}",
  "hybrid_muon_tace_module_include": "${HYBRID_MUON_TACE_MODULE_INCLUDE}",
  "hybrid_muon_tace_module_lr_scale": ${HYBRID_MUON_TACE_MODULE_LR_SCALE},
  "hybrid_muon_lr_factor": ${HYBRID_MUON_LR_FACTOR},
  "hybrid_muon_stage_two_lr_factor": ${HYBRID_MUON_STAGE_TWO_LR_FACTOR},
  "hybrid_muon_stage_two_route": "${HYBRID_MUON_STAGE_TWO_ROUTE}",
  "hybrid_muon_weight_decay": ${HYBRID_MUON_WEIGHT_DECAY},
  "hybrid_muon_adam_variant": "${HYBRID_MUON_ADAM_VARIANT}",
  "hybrid_muon_lr_scale_mode": "${HYBRID_MUON_LR_SCALE_MODE}",
  "hybrid_muon_match_rms_coeff": ${HYBRID_MUON_MATCH_RMS_COEFF},
  "hybrid_muon_magma_lite": "${HYBRID_MUON_MAGMA_LITE}",
  "hybrid_muon_magma_initial_score": ${HYBRID_MUON_MAGMA_INITIAL_SCORE},
  "hybrid_muon_magma_warmup_steps": ${HYBRID_MUON_MAGMA_WARMUP_STEPS},
  "hybrid_muon_magma_bypass_first_step": "${HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP}",
  "slurm_gpus_per_node": "${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-unknown}}",
  "stage1_energy_weight": ${STAGE1_ENERGY_WEIGHT},
  "stage1_forces_weight": ${STAGE1_FORCES_WEIGHT},
  "stage2_energy_weight": ${STAGE2_ENERGY_WEIGHT},
  "stage2_forces_weight": ${STAGE2_FORCES_WEIGHT},
  "base_optimizer": "${BASE_OPTIMIZER}",
  "cases": "${CASES}",
  "git_commit": "${SOURCE_GIT_COMMIT}",
  "git_branch": "${SOURCE_GIT_BRANCH}",
  "enforce_static_source": "${ENFORCE_STATIC_SOURCE}",
  "compile_setup_gate": "${COMPILE_SETUP_GATE}",
  "compile_max_cache_entries": ${COMPILE_MAX_CACHE_ENTRIES},
  "compile_parity_gradients": "${COMPILE_PARITY_GRADIENTS}",
  "compile_parity_check_strict": "${COMPILE_PARITY_CHECK_STRICT}",
  "train_file": "${TRAIN_FILE}",
  "train_size": ${TRAIN_SIZE},
  "valid_file": "${VALID_FILE}"
}
JSON

stage_two_args=()
case "${STAGE_TWO}" in
  True|true|1|yes|YES|on|ON)
    stage_two_args=(
      --stage_two
      --start_stage_two_update="${START_STAGE_TWO_UPDATE}"
      --stage_two_energy_weight="${STAGE2_ENERGY_WEIGHT}"
      --stage_two_forces_weight="${STAGE2_FORCES_WEIGHT}"
    )
    ;;
  False|false|0|no|NO|off|OFF)
    ;;
  *)
    echo "Invalid MACE_OC20NEB_STAGE_TWO=${STAGE_TWO}; expected True or False" >&2
    exit 2
    ;;
esac

common_args=(
  --train_file="${TRAIN_FILE}"
  --valid_file="${VALID_FILE}"
  --test_file="${TEST_FILE}"
  --E0s=average
  --energy_key=energy
  --forces_key=forces
  --model=ScaleShiftMACE
  --num_channels="${MACE_OC20NEB_NUM_CHANNELS:-64}"
  --max_L="${MACE_OC20NEB_MAX_L:-1}"
  --num_interactions="${MACE_OC20NEB_NUM_INTERACTIONS:-2}"
  --correlation="${MACE_OC20NEB_CORRELATION:-3}"
  --r_max="${MACE_OC20NEB_R_MAX:-6.0}"
  --batch_size="${BATCH_SIZE}"
  --valid_batch_size="${VALID_BATCH_SIZE}"
  --max_num_epochs="${MAX_EPOCHS}"
  --max_num_updates="${TARGET_STEPS}"
  --patience=999999
  --eval_interval="${EVAL_INTERVAL}"
  --eval_interval_updates="${EVAL_INTERVAL_UPDATES}"
  --checkpoint_interval_updates="${CHECKPOINT_INTERVAL_UPDATES}"
  --error_table="${MACE_OC20NEB_ERROR_TABLE:-PerAtomMAE}"
  --default_dtype=float32
  --train_amp_dtype=none
  --train_tf32
  --device=cuda
  --seed="${SEED}"
  --shuffle=True
  --enable_cueq="${MACE_OC20NEB_ENABLE_CUEQ:-False}"
  --optimizer="${BASE_OPTIMIZER}"
  --scheduler="${SCHEDULER}"
  --lr_scheduler_interval="${LR_SCHEDULER_INTERVAL}"
  --lr_wsd_warmup_ratio="${LR_WSD_WARMUP_RATIO}"
  --lr_wsd_stop_lr_ratio="${LR_WSD_STOP_LR_RATIO}"
  --lr_wsd_decay_phase_ratio="${LR_WSD_DECAY_PHASE_RATIO}"
  --lr_wsd_decay_type="${LR_WSD_DECAY_TYPE}"
  --lr="${LR}"
  --weight_decay="${WEIGHT_DECAY}"
  --loss=weighted
  --energy_weight="${STAGE1_ENERGY_WEIGHT}"
  --forces_weight="${STAGE1_FORCES_WEIGHT}"
  --loss_prefactor_schedule="${LOSS_PREFACTOR_SCHEDULE}"
  --loss_prefactor_start_update="${LOSS_PREFACTOR_START_UPDATE}"
  --loss_prefactor_end_update="${LOSS_PREFACTOR_END_UPDATE}"
  "${stage_two_args[@]}"
  --clip_grad="${MACE_OC20NEB_CLIP_GRAD:-10.0}"
)

case "${MACE_OC20NEB_RESTART_LATEST:-False}" in
  True|true|1|yes|YES|on|ON)
    common_args+=(--restart_latest)
    ;;
  False|false|0|no|NO|off|OFF|"")
    ;;
  *)
    echo "Invalid MACE_OC20NEB_RESTART_LATEST=${MACE_OC20NEB_RESTART_LATEST}; expected True or False" >&2
    exit 2
    ;;
esac

run_case() {
  local case_name=$1
  shift
  local case_dir="${RUN_ROOT}/${case_name}"
  mkdir -p "${case_dir}/logs" "${case_dir}/models" "${case_dir}/checkpoints" "${case_dir}/results"
  local case_env_name
  local restart_source_dir
  case_env_name="MACE_OC20NEB_RESTART_${case_name^^}_CHECKPOINTS_DIR"
  restart_source_dir="${!case_env_name:-${MACE_OC20NEB_RESTART_CHECKPOINTS_DIR:-}}"
  if [[ -n "${restart_source_dir}" ]]; then
    if [[ ! -d "${restart_source_dir}" ]]; then
      echo "Restart checkpoint directory not found for ${case_name}: ${restart_source_dir}" >&2
      return 2
    fi
    find "${restart_source_dir}" -maxdepth 1 -type f -name '*.pt' -exec cp -n {} "${case_dir}/checkpoints/" \;
  fi

  local monitor_pid=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucvmte -o T > "${case_dir}/nvdmon_job-${SLURM_JOB_ID:-local}_${case_name}.log" &
    monitor_pid=$!
  fi

  set +e
  "${PYTHON_BIN}" -m mace.cli.run_train \
    --name="oc20neb_fullcase200_ef20k_${case_name}" \
    --work_dir="${case_dir}" \
    --log_dir="${case_dir}/logs" \
    --model_dir="${case_dir}/models" \
    --checkpoints_dir="${case_dir}/checkpoints" \
    --results_dir="${case_dir}/results" \
    "${common_args[@]}" \
    "$@"
  local status=$?
  set -e

  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
    wait "${monitor_pid}" >/dev/null 2>&1 || true
  fi
  return "${status}"
}

case_enabled() {
  local wanted=$1
  local selected
  IFS=',' read -r -a selected <<< "${CASES}"
  for selected_case in "${selected[@]}"; do
    selected_case=${selected_case//[[:space:]]/}
    if [[ "${selected_case}" == "${wanted}" ]]; then
      return 0
    fi
  done
  return 1
}

run_selected_case() {
  local case_name=$1
  shift
  if case_enabled "${case_name}"; then
    if source_guard_enabled; then
      assert_static_source
    fi
    echo "Running OC20NEB 20k case: ${case_name}"
    run_case "${case_name}" "$@"
  else
    echo "Skipping OC20NEB 20k case: ${case_name}"
  fi
}

compile_args=(
  --edge_force_compile
  --edge_force_compile_force_gradient_mode=positions
  --edge_force_compile_tracing_mode=real
  --edge_force_compile_setup_gate="${COMPILE_SETUP_GATE}"
  --edge_force_compile_cache_policy=bucket
  --edge_force_compile_bucket_atoms="${COMPILE_BUCKET_ATOMS}"
  --edge_force_compile_bucket_edges="${COMPILE_BUCKET_EDGES}"
  --edge_force_compile_bucket_margin="${COMPILE_BUCKET_MARGIN}"
  --edge_force_compile_max_cache_entries="${COMPILE_MAX_CACHE_ENTRIES}"
  --edge_force_compile_min_repeats=2
  --edge_force_compile_parity_check_interval=1000
  --edge_force_compile_fixed_probe_interval=1000
  --no-edge_force_compile_allow_fallback
  --no-edge_force_compile_disable_negative_speedup
)

case "${COMPILE_PARITY_GRADIENTS}" in
  True|true|1|yes|YES|on|ON)
    compile_args+=(--edge_force_compile_parity_check_gradients)
    ;;
  False|false|0|no|NO|off|OFF)
    compile_args+=(--no-edge_force_compile_parity_check_gradients)
    ;;
  *)
    echo "Invalid MACE_OC20NEB_COMPILE_PARITY_GRADIENTS=${COMPILE_PARITY_GRADIENTS}; expected True or False" >&2
    exit 2
    ;;
esac

case "${COMPILE_PARITY_CHECK_STRICT}" in
  True|true|1|yes|YES|on|ON)
    compile_args+=(--edge_force_compile_parity_check_strict)
    ;;
  False|false|0|no|NO|off|OFF)
    compile_args+=(--no-edge_force_compile_parity_check_strict)
    ;;
  *)
    echo "Invalid MACE_OC20NEB_COMPILE_PARITY_CHECK_STRICT=${COMPILE_PARITY_CHECK_STRICT}; expected True or False" >&2
    exit 2
    ;;
esac

hybrid_muon_magma_flags=()
case "${HYBRID_MUON_MAGMA_LITE}" in
  True|true|1|yes|YES|on|ON)
    hybrid_muon_magma_flags+=(--hybrid_muon_magma_lite)
    ;;
  False|false|0|no|NO|off|OFF)
    ;;
  *)
    echo "Invalid MACE_OC20NEB_HYBRID_MUON_MAGMA_LITE=${HYBRID_MUON_MAGMA_LITE}; expected True or False" >&2
    exit 2
    ;;
esac
case "${HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP}" in
  True|true|1|yes|YES|on|ON)
    hybrid_muon_magma_flags+=(--hybrid_muon_magma_bypass_first_step)
    ;;
  False|false|0|no|NO|off|OFF)
    ;;
  *)
    echo "Invalid MACE_OC20NEB_HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP=${HYBRID_MUON_MAGMA_BYPASS_FIRST_STEP}; expected True or False" >&2
    exit 2
    ;;
esac

hybrid_muon_args=(
  --optimizer=hybrid_muon
  --hybrid_muon_mode="${HYBRID_MUON_MODE}"
  --hybrid_muon_routing="${HYBRID_MUON_ROUTING}"
  --hybrid_muon_tace_module_include="${HYBRID_MUON_TACE_MODULE_INCLUDE}"
  --hybrid_muon_tace_module_lr_scale="${HYBRID_MUON_TACE_MODULE_LR_SCALE}"
  --hybrid_muon_lr_factor="${HYBRID_MUON_LR_FACTOR}"
  --hybrid_muon_stage_two_lr_factor="${HYBRID_MUON_STAGE_TWO_LR_FACTOR}"
  --hybrid_muon_stage_two_route="${HYBRID_MUON_STAGE_TWO_ROUTE}"
  --hybrid_muon_weight_decay="${HYBRID_MUON_WEIGHT_DECAY}"
  --hybrid_muon_adam_variant="${HYBRID_MUON_ADAM_VARIANT}"
  --hybrid_muon_lr_scale_mode="${HYBRID_MUON_LR_SCALE_MODE}"
  --hybrid_muon_match_rms_coeff="${HYBRID_MUON_MATCH_RMS_COEFF}"
  --hybrid_muon_magma_initial_score="${HYBRID_MUON_MAGMA_INITIAL_SCORE}"
  --hybrid_muon_magma_warmup_steps="${HYBRID_MUON_MAGMA_WARMUP_STEPS}"
  "${hybrid_muon_magma_flags[@]}"
)

hybrid_muon_tace_args=(
  "${hybrid_muon_args[@]}"
  --hybrid_muon_routing=tace
)

hybrid_muon_module_args=(
  "${hybrid_muon_args[@]}"
  --hybrid_muon_routing=module
)

cueq_args=(
  --enable_cueq=True
)

extra_args=()
if [[ -n "${MACE_OC20NEB_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${MACE_OC20NEB_EXTRA_ARGS}"
fi
post_args=()
if [[ -n "${MACE_OC20NEB_POST_ARGS:-}" ]]; then
  read -r -a post_args <<< "${MACE_OC20NEB_POST_ARGS}"
fi

run_selected_case adamw "${extra_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon "${extra_args[@]}" "${hybrid_muon_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon_tace "${extra_args[@]}" "${hybrid_muon_tace_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon_module "${extra_args[@]}" "${hybrid_muon_module_args[@]}" "${post_args[@]}"
run_selected_case cueq_adamw "${extra_args[@]}" "${cueq_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon_tace "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_tace_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon_module "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_module_args[@]}" "${post_args[@]}"

run_selected_case eager "${extra_args[@]}" "${post_args[@]}"
run_selected_case cueq "${extra_args[@]}" "${cueq_args[@]}" "${post_args[@]}"
run_selected_case compile "${extra_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case hybrid_muon_compile "${extra_args[@]}" "${hybrid_muon_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case cueq_compile "${extra_args[@]}" "${cueq_args[@]}" "${compile_args[@]}" "${post_args[@]}"
run_selected_case cueq_hybrid_muon_compile "${extra_args[@]}" "${cueq_args[@]}" "${hybrid_muon_args[@]}" "${compile_args[@]}" "${post_args[@]}"

"${PYTHON_BIN}" scripts/benchmarks/recio8k_accel/parse_metrics.py "${RUN_ROOT}" > "${RUN_ROOT}/summary.json"
"${PYTHON_BIN}" scripts/benchmarks/oc20neb_fps/summarize_fullcase200_ef20k_matrix.py \
  "${RUN_ROOT}" \
  --csv "${RUN_ROOT}/matrix_summary.csv" \
  --comparisons-csv "${RUN_ROOT}/matrix_comparisons.csv" \
  --comparisons-json "${RUN_ROOT}/matrix_comparisons.json" \
  > "${RUN_ROOT}/matrix_summary.json"
cat "${RUN_ROOT}/summary.json"
cat "${RUN_ROOT}/matrix_summary.json"
cat "${RUN_ROOT}/matrix_comparisons.json"
