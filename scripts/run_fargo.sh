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

# --- Build ---
if [ ! -d "build" ]; then
  echo "Directory build does not exist, creating it"
  mkdir build
fi
cd build
cmake ..
make -j8 benchmark_fargo

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="fargo"
mode="${MODE:-search}"
L=6
K_fargo=12
ub=(0 30 60 80 100 130 160 180 200 300 400 500 600 700)
type="ip"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_fargo}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_fargo}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_fargo}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_L${L}_K${K_fargo}"

mkdir -p "${pre_path}/${algorithm}"
mkdir -p "$(dirname "${INDEX_PREFIX_PATH}")" "$(dirname "${RESULT_PREFIX_PATH}")" "$(dirname "${log_file}")"

# --- Execution ---
case "$mode" in
  build)
    echo "Building index..."
    ./test/benchmark_fargo "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${L}" "${K_fargo}" 0 0.8 1 "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
    ;;
  search)
    echo "Searching index..."
    for ub_val in "${ub[@]}"; do
      echo "========================================" | tee -a "${log_file}"
      echo "Running with ub: $ub_val, break_ratio: 1" | tee -a "${log_file}"
      ./test/benchmark_fargo "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${L}" "${K_fargo}" "$ub_val" 0.8 1 "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
      python3 "${recall_path}" "${DATA_PRE_PATH}" "${PREFIX}" "${TRAIN_NAME}" "${TEST_NAME}" "${algorithm}" "${K}" "${type}" "${result_name}" "${pre_path}" | tee -a "${log_file}"
      echo "========================================" | tee -a "${log_file}"
    done
    ;;
  *)
    echo "Invalid mode. Use 'build' or 'search'."
    exit 1
    ;;
esac
