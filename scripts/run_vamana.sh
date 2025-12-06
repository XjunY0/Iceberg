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
make -j8 benchmark_vamana

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="vamana"
mode="${MODE:-search}"
L=60
R=48
C=256
alpha=1.2
efs=(100 150 200 250 300 350 400 450 500 550 600 800 1000 1500)
type="nn"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${R}_L${C}_alpha${alpha}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${R}_L${C}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_M${R}_L${C}_alpha${alpha}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_M${R}_L${C}"

# Ensure output directories exist
mkdir -p "${pre_path}/${algorithm}"
mkdir -p "$(dirname "${INDEX_PREFIX_PATH}")" "$(dirname "${RESULT_PREFIX_PATH}")" "$(dirname "${log_file}")"

# --- Execution ---
case "$mode" in
  build)
    echo "Building index..."
    ./test/benchmark_vamana "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${L}" "${R}" "${C}" "${alpha}" "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
    ;;
  search)
    echo "Searching index..."
    for ef_search in "${efs[@]}"; do
      echo "========================================" | tee -a "${log_file}"
      echo "Running with efs: $ef_search" | tee -a "${log_file}"
      ./test/benchmark_vamana "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}"  "${L}" "${R}" "${C}" "${alpha}" "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" "${ef_search}" | tee -a "${log_file}"
      python3 "${recall_path}" "${DATA_PRE_PATH}" "${PREFIX}" "${TRAIN_NAME}" "${TEST_NAME}" "${algorithm}" "${K}" "${type}" "${result_name}" "${pre_path}" | tee -a "${log_file}"
      echo "========================================" | tee -a "${log_file}"
    done
    ;;
  *)
    echo "Invalid mode. Use 'build' or 'search'."
    exit 1
    ;;
esac
