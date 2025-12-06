#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

# --- Source Dataset Configuration ---
source "${PROJECT_ROOT}/scripts/config_dataset.sh"
: "${DATASET:=imagenet1k_avg}"
case "$DATASET" in
  imagenet1k_avg) dataset_imagenet1k_avg ;;
  imagenet1k_dinov2) dataset_imagenet1k_dinov2 ;;
  imagenet1k_eva02) dataset_imagenet1k_eva02 ;;
  glink_ir101) dataset_glink_ir101 ;;
  glink_vit) dataset_glink_vit ;;
  commerce) dataset_commerce ;;
  bookcorpus) dataset_bookcorpus ;;
  *) dataset_imagenet1k_avg ;;
esac

# --- Environment Settings ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="ivfpq"
mode="${MODE:-search}"

nlist="${NLIST:-12000}"
pqm="${PQM:-48}"
rerank_k="${RERANK_K:-300}"

RECALL_SCRIPT_PATH="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
PYTHON_SCRIPT_PATH="${PROJECT_ROOT}/test/benchmark_ivfpq.py"

INDEX_DIR="${pre_path}/${algorithm}"
INDEX_FILENAME="${PREFIX}_ivf${nlist}_pq${pqm}.index"
FULL_INDEX_PATH="${INDEX_DIR}/${INDEX_FILENAME}"

LOG_DIR="${pre_path}/${algorithm}"
LOG_FILE="${LOG_DIR}/${PREFIX}_${mode}.log"
mkdir -p "${LOG_DIR}"

# Ensure output directories exist
mkdir -p "${INDEX_DIR}"

# --- Execution ---
python3 "${PYTHON_SCRIPT_PATH}" \
    "${DATA_PRE_PATH}" \
    "${PREFIX}" \
    "${TRAIN_NAME}" \
    "${TEST_NAME}" \
    "${RECALL_SCRIPT_PATH}" \
    --dim "${DATA_DIM}" \
    --data_num "${data_num:-0}" \
    --query_num "${query_num:-0}" \
    --mode "${mode}" \
    --algorithm "${algorithm}" \
    --top_k "${K}" \
    --nlist "${nlist}" \
    --pqm "${pqm}" \
    --rerank_k "${rerank_k}" \
    --index_path "${FULL_INDEX_PATH}" \
    --output_path "${pre_path}" \
    --data_type "${DATASET_TYPE}" | tee -a "${LOG_FILE}"
