#!/usr/bin/env bash
#SBATCH --job-name=MACE-DPA4-STRESS20K
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=runs/dpa4_stress_20k_demo/%x-%j.out
#SBATCH --error=runs/dpa4_stress_20k_demo/%x-%j.err

set -euo pipefail

REPO_ROOT=${MACE_DPA4_DEMO_REPO_ROOT:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel}
PYTHON_BIN=${MACE_DPA4_DEMO_PYTHON:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
DATA_XYZ=${MACE_DPA4_DEMO_XYZ:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/row_000060_34ee2613164a1ac6_Pd-Zn_111_-bulk1127-shift0.xyz}
RUN_ROOT=${MACE_DPA4_DEMO_RUN_ROOT:-${REPO_ROOT}/runs/dpa4_stress_20k_demo/${SLURM_JOB_ID:-local}}
TARGET_STEPS=${MACE_DPA4_DEMO_TARGET_STEPS:-20000}
BATCH_SIZE=${MACE_DPA4_DEMO_BATCH_SIZE:-8}
VALID_FRACTION=${MACE_DPA4_DEMO_VALID_FRACTION:-0.05}
SEED=${MACE_DPA4_DEMO_SEED:-456}
EVAL_REPORTS=${MACE_DPA4_DEMO_EVAL_REPORTS:-5}

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${DATA_XYZ}" ]]; then
  echo "Training XYZ not found: ${DATA_XYZ}" >&2
  exit 2
fi

read -r NUM_CONFIGS TRAIN_SIZE MAX_EPOCHS EVAL_INTERVAL < <("${PYTHON_BIN}" - <<PY
import math
from ase.io import read
xyz = "${DATA_XYZ}"
valid_fraction = float("${VALID_FRACTION}")
batch_size = int("${BATCH_SIZE}")
target_steps = int("${TARGET_STEPS}")
eval_reports = max(1, int("${EVAL_REPORTS}"))
num_configs = len(read(xyz, index=":"))
train_size = min(num_configs - int(valid_fraction * num_configs), num_configs - 1)
if train_size <= 0:
    raise SystemExit(f"not enough configs in {xyz}: {num_configs}")
max_epochs = max(1, math.ceil(target_steps * batch_size / train_size))
eval_interval = max(1, max_epochs // eval_reports)
print(num_configs, train_size, max_epochs, eval_interval)
PY
)

cat > "${RUN_ROOT}/manifest.json" <<JSON
{
  "batch_size": ${BATCH_SIZE},
  "data_xyz": "${DATA_XYZ}",
  "eval_interval": ${EVAL_INTERVAL},
  "max_num_epochs": ${MAX_EPOCHS},
  "num_configs": ${NUM_CONFIGS},
  "python": "${PYTHON_BIN}",
  "repo_root": "${REPO_ROOT}",
  "target_steps": ${TARGET_STEPS},
  "train_size": ${TRAIN_SIZE},
  "valid_fraction": ${VALID_FRACTION}
}
JSON

common_args=(
  --train_file="${DATA_XYZ}"
  --valid_fraction="${VALID_FRACTION}"
  --test_file="${DATA_XYZ}"
  --E0s=average
  --energy_key=energy
  --forces_key=forces
  --stress_key=stress
  --model=ScaleShiftMACE
  --num_channels=32
  --max_L=1
  --num_interactions=2
  --correlation=3
  --r_max=5.0
  --batch_size="${BATCH_SIZE}"
  --valid_batch_size=16
  --max_num_epochs="${MAX_EPOCHS}"
  --patience=999999
  --eval_interval="${EVAL_INTERVAL}"
  --error_table=PerAtomMAEstressvirials
  --default_dtype=float32
  --train_amp_dtype=none
  --train_tf32
  --device=cuda
  --seed="${SEED}"
  --shuffle=False
  --enable_cueq=False
  --optimizer=adam
  --scheduler=ReduceLROnPlateau
  --lr=0.005
  --weight_decay=5e-7
  --loss=stress
  --energy_weight=1.0
  --forces_weight=100.0
  --stress_weight=1.0
  --clip_grad=10.0
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
    --name="dpa4_stress20k_${case_name}" \
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

