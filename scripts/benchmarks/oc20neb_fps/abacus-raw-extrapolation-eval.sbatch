#!/usr/bin/env bash
#SBATCH --job-name=MACE-ABACUS-OOD
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=runs/abacus_raw_extrapolation_eval/%x-%j.out
#SBATCH --error=runs/abacus_raw_extrapolation_eval/%x-%j.err

set -euo pipefail

REPO_ROOT=${MACE_ABACUS_REPO_ROOT:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel}
PYTHON_BIN=${MACE_ABACUS_PYTHON:-/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
DATA_REPO=${MACE_ABACUS_DATA_REPO:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/bin/RSQN-sai-experimental-data}
RUN_ROOT=${MACE_ABACUS_RUN_ROOT:-${REPO_ROOT}/runs/abacus_raw_extrapolation_eval/${SLURM_JOB_ID:-local}}
LIMIT_STRUCTURES=${MACE_ABACUS_LIMIT_STRUCTURES:-200}
INCLUDE_SOCKET=${MACE_ABACUS_INCLUDE_SOCKET:-0}

MACE_MH1_MODEL=${MACE_ABACUS_MACE_MH1_MODEL:-/home/gengjianrui/.cache/mace/mace-mh-1.model}
MACE_OMAT50K_MODEL=${MACE_ABACUS_MACE_OMAT50K_MODEL:-${DATA_REPO}/models/mace_oc20neb_omat_medium_finetune_50k/oc20neb_omat_medium_replay_multihead_ef20_compile_only_50k_compiled.model}
MACE_FPS5K_MODEL=${MACE_ABACUS_MACE_FPS5K_MODEL:-${DATA_REPO}/models/mace_oc20neb_fps5k_l1c64_b32_stage2/oc20neb_fps5k_l1c64_b32_adam_cueq_compile_200k_stage2_stagetwo_compiled.model}
DPA4_MODEL=${MACE_ABACUS_DPA4_MODEL:-${DATA_REPO}/models/dpa4mini_oc20neb_fps5k_train_valid10k_finetune_50k_lr1e-4/best.ckpt-50000.t-1.pt}

# Default to selected20 because it contains both sella and vib abacus_evals.
# Override with a colon-separated MACE_ABACUS_EXPERIMENT_ROOTS for broader sweeps.
DEFAULT_ROOT=${DATA_REPO}/experiments/abacus_rpbe_last10_seed_pure_sella_selected20
EXPERIMENT_ROOTS_RAW=${MACE_ABACUS_EXPERIMENT_ROOTS:-${DEFAULT_ROOT}}

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

IFS=':' read -r -a EXPERIMENT_ROOTS <<< "${EXPERIMENT_ROOTS_RAW}"
root_args=()
for root in "${EXPERIMENT_ROOTS[@]}"; do
  if [[ ! -d "${root}" ]]; then
    echo "ABACUS experiment root not found: ${root}" >&2
    exit 2
  fi
  root_args+=(--experiment-root "${root}")
done

common_args=(
  "${root_args[@]}"
  --limit-structures "${LIMIT_STRUCTURES}"
  --write-label-extxyz
  --device cuda
  --default-dtype float32
)
if [[ "${INCLUDE_SOCKET}" == "1" ]]; then
  common_args+=(--include-socket)
fi

cat > "${RUN_ROOT}/manifest.json" <<JSON
{
  "data_repo": "${DATA_REPO}",
  "dpa4_model": "${DPA4_MODEL}",
  "experiment_roots": "${EXPERIMENT_ROOTS_RAW}",
  "include_socket": "${INCLUDE_SOCKET}",
  "limit_structures": ${LIMIT_STRUCTURES},
  "mace_fps5k_model": "${MACE_FPS5K_MODEL}",
  "mace_mh1_model": "${MACE_MH1_MODEL}",
  "mace_omat50k_model": "${MACE_OMAT50K_MODEL}",
  "python": "${PYTHON_BIN}",
  "repo_root": "${REPO_ROOT}"
}
JSON

run_mace_eval() {
  local name=$1
  local model=$2
  local head=$3
  local out_dir="${RUN_ROOT}/${name}"
  mkdir -p "${out_dir}"
  local head_args=()
  if [[ -n "${head}" ]]; then
    head_args+=(--mace-head="${head}")
  fi
  "${PYTHON_BIN}" scripts/benchmarks/oc20neb_fps/evaluate_abacus_raw_extrapolation.py \
    "${common_args[@]}" \
    --model "${name}=mace:${model}" \
    "${head_args[@]}" \
    --output-dir "${out_dir}"
  cat "${out_dir}/summary.json"
}

run_mace_eval mace_mh1 "${MACE_MH1_MODEL}" oc20_usemppbe
run_mace_eval mace_omat50k "${MACE_OMAT50K_MODEL}" oc20neb
run_mace_eval mace_fps5k "${MACE_FPS5K_MODEL}" ""

# DPA4 preflight is explicit because this env may lack the DPA4 DeepMD runtime plugin.
# Known current failure mode on login: RuntimeError: Unknown model type: dpa4.
set +e
"${PYTHON_BIN}" - <<PY > "${RUN_ROOT}/dpa4_preflight.log" 2>&1
from deepmd.calculator import DP
DP(model="${DPA4_MODEL}", nlist_backend="ase")
print("DPA4_PREFLIGHT_OK")
PY
dpa4_status=$?
set -e
if [[ "${dpa4_status}" -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/benchmarks/oc20neb_fps/evaluate_abacus_raw_extrapolation.py \
    "${common_args[@]}" \
    --model "dpa4=deepmd:${DPA4_MODEL}" \
    --deepmd-nlist-backend ase \
    --output-dir "${RUN_ROOT}/dpa4"
  cat "${RUN_ROOT}/dpa4/summary.json"
else
  cat > "${RUN_ROOT}/dpa4_skip.json" <<JSON
{
  "model": "${DPA4_MODEL}",
  "reason": "DPA4 DeepMD calculator preflight failed; see dpa4_preflight.log. Common current error: Unknown model type: dpa4",
  "status": ${dpa4_status}
}
JSON
  cat "${RUN_ROOT}/dpa4_skip.json"
fi
