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

export TRAIN_FILE BATCH_SIZE TARGET_STEPS EVAL_REPORTS
read -r TRAIN_SIZE MAX_EPOCHS EVAL_INTERVAL < <("${PYTHON_BIN}" - <<'INNERPY'
import math
from ase.io import read
import os
train_file = os.environ["TRAIN_FILE"]
batch_size = int(os.environ["BATCH_SIZE"])
target_steps = int(os.environ["TARGET_STEPS"])
eval_reports = max(1, int(os.environ["EVAL_REPORTS"]))
train_size = len(read(train_file, index=":"))
if train_size <= 0:
    raise SystemExit(f"empty training file: {train_file}")
max_epochs = max(1, math.ceil(target_steps * batch_size / train_size))
eval_interval = max(1, max_epochs // eval_reports)
print(train_size, max_epochs, eval_interval)
INNERPY
)

cat > "${RUN_ROOT}/manifest.json" <<JSON
{
  "batch_size": ${BATCH_SIZE},
  "data_root": "${DATA_ROOT}",
  "eval_interval": ${EVAL_INTERVAL},
  "max_num_epochs": ${MAX_EPOCHS},
  "python": "${PYTHON_BIN}",
  "repo_root": "${REPO_ROOT}",
  "target_steps": ${TARGET_STEPS},
  "train_file": "${TRAIN_FILE}",
  "train_size": ${TRAIN_SIZE},
  "valid_file": "${VALID_FILE}"
}
JSON

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
  --patience=999999
  --eval_interval="${EVAL_INTERVAL}"
  --error_table=PerAtomMAE
  --default_dtype=float32
  --train_amp_dtype=none
  --train_tf32
  --device=cuda
  --seed="${SEED}"
  --shuffle=True
  --enable_cueq="${MACE_OC20NEB_ENABLE_CUEQ:-False}"
  --optimizer=adam
  --scheduler=ReduceLROnPlateau
  --lr="${MACE_OC20NEB_LR:-0.001}"
  --weight_decay="${MACE_OC20NEB_WEIGHT_DECAY:-0.001}"
  --loss=weighted
  --energy_weight="${MACE_OC20NEB_ENERGY_WEIGHT:-20.0}"
  --forces_weight="${MACE_OC20NEB_FORCES_WEIGHT:-20.0}"
  --clip_grad="${MACE_OC20NEB_CLIP_GRAD:-10.0}"
)

run_case() {
  local case_name=$1
  shift
  local case_dir="${RUN_ROOT}/${case_name}"
  mkdir -p "${case_dir}/logs" "${case_dir}/models" "${case_dir}/checkpoints" "${case_dir}/results"

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

run_case eager
run_case compile \
  --edge_force_compile \
  --edge_force_compile_force_gradient_mode=positions \
  --edge_force_compile_tracing_mode=real \
  --edge_force_compile_setup_gate=strict \
  --edge_force_compile_cache_policy=dynamic \
  --edge_force_compile_min_repeats=2 \
  --edge_force_compile_parity_check_interval=1000 \
  --edge_force_compile_fixed_probe_interval=1000 \
  --no-edge_force_compile_allow_fallback \
  --no-edge_force_compile_disable_negative_speedup

"${PYTHON_BIN}" scripts/benchmarks/recio8k_accel/parse_metrics.py "${RUN_ROOT}" > "${RUN_ROOT}/summary.json"
cat "${RUN_ROOT}/summary.json"
