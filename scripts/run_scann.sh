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

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="scann"
mode="${MODE:-build}"

num_leaves="${NUM_LEAVES:-2000}"
num_leaves_to_search="${NUM_LEAVES_TO_SEARCH:-1000}"

INDEX_DIR="${pre_path}/${algorithm}"
LOG_DIR="${pre_path}/${algorithm}"
INDEX_PREFIX_PATH="${INDEX_DIR}/${PREFIX}.index"
LOG_PATH="${LOG_DIR}/${PREFIX}.log"

mkdir -p "${INDEX_DIR}" "${LOG_DIR}"

# --- Execution ---
python3 "${PROJECT_ROOT}/test/benchmark_scann.py" \
  "${DATA_PRE_PATH}" \
  "${PREFIX}" \
  "${TRAIN_NAME}" \
  "${TEST_NAME}" \
  "${algorithm}" \
  "${K}" \
  "${data_num:-0}" \
  "${query_num:-0}" \
  "${DATA_DIM}" \
  "${num_leaves}" \
  "${num_leaves_to_search}" \
  "${DATASET_TYPE}" \
  "${mode}" \
  "${INDEX_PREFIX_PATH}" \
  --project_root "${PROJECT_ROOT}" | tee -a "${LOG_PATH}"
