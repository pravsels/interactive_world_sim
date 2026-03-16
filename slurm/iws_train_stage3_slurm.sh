#!/usr/bin/env bash
#SBATCH --job-name=iws-train-stage3
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --requeue

set -euo pipefail

module purge
module load brics/apptainer-multi-node

# ---- Isambard runtime paths ----
# Keep repo on home (small), keep heavy artifacts on scratch (large).
PROJECT_NAME="${PROJECT_NAME:-interactive_world_sim}"
PROJECT_CODE="${PROJECT_CODE:-u6cr}"
REPO_DIR="${REPO_DIR:-$HOME/${PROJECT_NAME}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_CODE}/${USER}/${PROJECT_NAME}}"

# Scratch-backed paths for heavy files.
SIF_PATH="${SIF_PATH:-${SCRATCH_ROOT}/containers/interactive-world-sim_isambard-arm64.sif}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${SCRATCH_ROOT}/outputs}"
HF_CACHE="${HF_CACHE:-${SCRATCH_ROOT}/huggingface_cache}"
WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${SCRATCH_ROOT}/wandb_cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${SCRATCH_ROOT}/wandb_config}"
PYTHON_EXT_DIR="${PYTHON_EXT_DIR:-${SCRATCH_ROOT}/python_packages}"

# ---- Train params are loaded from YAML ----
# Override path with:
#   sbatch --export=ALL,TRAIN_CONFIG_YAML=/path/to/configurations/isambard_train_stage3.yaml slurm/iws_train_stage3_slurm.sh
# NOTE: default uses REPO_DIR because sbatch runs scripts from Slurm spool paths.
TRAIN_CONFIG_YAML="${TRAIN_CONFIG_YAML:-${REPO_DIR}/configurations/isambard_train_stage3.yaml}"
if [[ ! -f "${TRAIN_CONFIG_YAML}" ]]; then
  echo "Training config yaml not found: ${TRAIN_CONFIG_YAML}" >&2
  exit 1
fi

CONFIG_DIR="$(dirname "${TRAIN_CONFIG_YAML}")"
CONFIG_NAME="$(basename "${TRAIN_CONFIG_YAML}" .yaml)"

# Use WAN-format H5 on scratch by default (images + action deltas).
# Camera convention reference: camera_0 = wrist, camera_1 = front.
WAN_H5_PATH="${WAN_H5_PATH:-/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_wan224.h5}"
WAN_H5_CONTAINER_PATH="${WAN_H5_CONTAINER_PATH:-/mnt/wan_dataset.h5}"
WAN_STATS_JSON_PATH="${WAN_STATS_JSON_PATH:-/scratch/u6cr/pravsels.u6cr/latent_safety/arx5_datasets_6Feb_26_stats.json}"
WAN_STATS_JSON_CONTAINER_PATH="${WAN_STATS_JSON_CONTAINER_PATH:-/mnt/wan_dataset_stats.json}"
DATASET_OVERRIDE_ARGS="${DATASET_OVERRIDE_ARGS:-dataset.h5_path=${WAN_H5_CONTAINER_PATH} dataset.stats_json_path=${WAN_STATS_JSON_CONTAINER_PATH}}"

# Resume helper for 1-day walltime jobs.
# - LOAD_CKPT_PATH: absolute/local path to a stage-3 checkpoint to continue from.
LOAD_ARGS=""
if [[ -n "${LOAD_CKPT_PATH:-}" ]]; then
  LOAD_ARGS="${LOAD_ARGS} load=${LOAD_CKPT_PATH}"
fi

TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
TRAIN_CMD="python main.py --config-path ${CONFIG_DIR} --config-name ${CONFIG_NAME} ${DATASET_OVERRIDE_ARGS} ${LOAD_ARGS} ${TRAIN_EXTRA_ARGS}"

mkdir -p "${DATA_DIR}" "${OUTPUTS_DIR}" "${HF_CACHE}" \
  "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${PYTHON_EXT_DIR}"

if [[ ! -f "${WAN_H5_PATH}" ]]; then
  echo "WAN H5 not found: ${WAN_H5_PATH}" >&2
  exit 1
fi
if [[ ! -f "${WAN_STATS_JSON_PATH}" ]]; then
  echo "WAN stats json not found: ${WAN_STATS_JSON_PATH}" >&2
  exit 1
fi

echo "[$(date -Is)] starting training on $(hostname)"
echo "TRAIN_CONFIG_YAML=${TRAIN_CONFIG_YAML}"
echo "SIF_PATH=${SIF_PATH}"
echo "REPO_DIR=${REPO_DIR}"
echo "SCRATCH_ROOT=${SCRATCH_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUTS_DIR=${OUTPUTS_DIR}"
echo "WAN_H5_PATH=${WAN_H5_PATH}"
echo "WAN_H5_CONTAINER_PATH=${WAN_H5_CONTAINER_PATH}"
echo "WAN_STATS_JSON_PATH=${WAN_STATS_JSON_PATH}"
echo "WAN_STATS_JSON_CONTAINER_PATH=${WAN_STATS_JSON_CONTAINER_PATH}"
echo "PYTHON_EXT_DIR=${PYTHON_EXT_DIR}"
echo "CONFIG_NAME=${CONFIG_NAME}"
if [[ -n "${LOAD_CKPT_PATH:-}" ]]; then
  echo "LOAD_CKPT_PATH=${LOAD_CKPT_PATH}"
fi

start_time="$(date -Is --utc)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started (UTC): ${start_time}"
echo "===================================="

set +e
srun --ntasks=1 --gpus=1 --cpu-bind=cores \
apptainer exec --nv \
  --bind "${REPO_DIR}:/workspace" \
  --bind "${DATA_DIR}:/workspace/data" \
  --bind "${WAN_H5_PATH}:${WAN_H5_CONTAINER_PATH}" \
  --bind "${WAN_STATS_JSON_PATH}:${WAN_STATS_JSON_CONTAINER_PATH}" \
  --bind "${OUTPUTS_DIR}:/workspace/outputs" \
  --bind "${HF_CACHE}:/root/.cache/huggingface" \
  --bind "${WANDB_DIR}:${WANDB_DIR}" \
  --bind "${WANDB_CACHE_DIR}:${WANDB_CACHE_DIR}" \
  --bind "${WANDB_CONFIG_DIR}:${WANDB_CONFIG_DIR}" \
  --bind "${PYTHON_EXT_DIR}:${PYTHON_EXT_DIR}" \
  "${SIF_PATH}" \
  bash -lc "cd /workspace && \
    TORCH_SITE=\$(python -c 'import torch,os; print(os.path.dirname(torch.__path__[0]))') && \
    export LD_LIBRARY_PATH=\${TORCH_SITE}/nvidia/cudnn/lib:\${TORCH_SITE}/nvidia/cublas/lib:\${TORCH_SITE}/torch/lib:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:\$LD_LIBRARY_PATH && \
    python -c 'import torch; v=torch.backends.cudnn.version(); print(f\"cudnn_version={v}\"); assert v >= 91000, f\"cuDNN {v} too old\"' && \
    export HF_HOME=/root/.cache/huggingface && \
    export WANDB_DIR=${WANDB_DIR} && \
    export WANDB_CACHE_DIR=${WANDB_CACHE_DIR} && \
    export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR} && \
    mkdir -p ${PYTHON_EXT_DIR} && \
    export PYTHONPATH=${PYTHON_EXT_DIR}:/workspace:\$PYTHONPATH && \
    python -c 'from numcodecs.blosc import cbuffer_sizes' >/dev/null 2>&1 || \
      python -m pip install --upgrade --no-deps --target ${PYTHON_EXT_DIR} numcodecs==0.11.0 && \
    ${TRAIN_CMD}"
EXIT_CODE=$?
set -e

end_time="$(date -Is --utc)"
echo ""
echo "===================================="
echo "Started (UTC):  ${start_time}"
echo "Finished (UTC): ${end_time}"
echo "Exit Code: ${EXIT_CODE}"
echo "===================================="

if [ "${EXIT_CODE}" -ne 0 ]; then
  echo ""
  echo "ERROR: Training failed with exit code ${EXIT_CODE}"
  echo "Check slurm-${SLURM_JOB_ID}.err for details"
  exit "${EXIT_CODE}"
fi
