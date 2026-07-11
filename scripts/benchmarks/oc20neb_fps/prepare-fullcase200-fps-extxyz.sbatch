#!/usr/bin/env bash
#SBATCH --job-name=MACE-OC20NEB-FPSPREP
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=runs/oc20neb_fullcase200_fps_extxyz/%x-%j.out
#SBATCH --error=runs/oc20neb_fullcase200_fps_extxyz/%x-%j.err

set -euo pipefail

REPO_ROOT=${MACE_OC20NEB_PREP_REPO_ROOT:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel}
PYTHON_BIN=${MACE_OC20NEB_PREP_PYTHON:-/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
DATA_ROOT=${MACE_OC20NEB_PREP_DATA_ROOT:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/bin/RSQN-sai-experimental-data/data/dft_trajs_for_release}
MANIFEST=${MACE_OC20NEB_PREP_MANIFEST:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/bin/RSQN-sai-experimental-data/experiments/sella_interp_first10_selected200/selected200_manifest.jsonl}
MODEL=${MACE_OC20NEB_PREP_MODEL:-/home/gengjianrui/.cache/mace/mace-mh-1.model}
HEAD=${MACE_OC20NEB_PREP_HEAD:-oc20_usemppbe}
OUTPUT_DIR=${MACE_OC20NEB_PREP_OUTPUT_DIR:-${REPO_ROOT}/runs/oc20neb_fullcase200_fps_extxyz}
FEATURES_NPZ=${MACE_OC20NEB_PREP_FEATURES_NPZ:-${OUTPUT_DIR}/mace_mh1_node_feats_features.npz}
TRAIN_SIZE=${MACE_OC20NEB_PREP_TRAIN_SIZE:-5000}
VALID_SIZE=${MACE_OC20NEB_PREP_VALID_SIZE:-10000}
BATCH_SIZE=${MACE_OC20NEB_PREP_BATCH_SIZE:-16}
CHUNK_SIZE=${MACE_OC20NEB_PREP_CHUNK_SIZE:-256}
FPS_FEATURE_DIM=${MACE_OC20NEB_PREP_FPS_FEATURE_DIM:-128}
SEED=${MACE_OC20NEB_PREP_SEED:-20260711}
DEVICE=${MACE_OC20NEB_PREP_DEVICE:-cuda}
OVERWRITE=${MACE_OC20NEB_PREP_OVERWRITE:-0}

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "MACE feature model not found: ${MODEL}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Selected200 manifest not found: ${MANIFEST}" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "OC20NEB trajectory data root not found: ${DATA_ROOT}" >&2
  exit 2
fi

feature_args=(
  --manifest "${MANIFEST}"
  --data-root "${DATA_ROOT}"
  --model "${MODEL}"
  --head "${HEAD}"
  --output "${FEATURES_NPZ}"
  --batch-size "${BATCH_SIZE}"
  --chunk-size "${CHUNK_SIZE}"
  --device "${DEVICE}"
  --default-dtype float32
)
prepare_args=(
  --manifest "${MANIFEST}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --features-npz "${FEATURES_NPZ}"
  --train-size "${TRAIN_SIZE}"
  --valid-size "${VALID_SIZE}"
  --seed "${SEED}"
  --fps-feature-dim "${FPS_FEATURE_DIM}"
)
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "True" ]]; then
  feature_args+=(--overwrite)
  prepare_args+=(--overwrite)
fi

if [[ -f "${FEATURES_NPZ}" && "${OVERWRITE}" != "1" && "${OVERWRITE}" != "true" && "${OVERWRITE}" != "True" ]]; then
  echo "Skipping feature extraction because ${FEATURES_NPZ} already exists"
else
  "${PYTHON_BIN}" scripts/benchmarks/oc20neb_fps/extract_fullcase200_mace_features.py "${feature_args[@]}"
fi
"${PYTHON_BIN}" scripts/benchmarks/oc20neb_fps/prepare_fullcase200_fps_extxyz.py "${prepare_args[@]}"
