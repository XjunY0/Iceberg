#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
# Resolve project root (default to repo root inferred from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Ensure commands run at the project root
cd "${PROJECT_ROOT}"

# --- Source Dataset Configuration ---
source "${PROJECT_ROOT}/scripts/config_dataset.sh"
# Allow selecting dataset via env DATASET, default to imagenet1k_avg
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
make -j8 benchmark_mag

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="mag"
mode="${MODE:-search}"
threshold=5
R_IP=15
L=60
R=48
C=300
M=48
# efs=(100 120 150 200 250 300 350 400 450 500 550 600 800 1000 1500)
efs=(100 150 200 250 300 500 800 1000 1500)
type="ip"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${C}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${C}.result"
KNNG_PATH="${DATA_PRE_PATH}/${TRAIN_NAME}.knng"
log_file="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${C}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_M${M}_L${C}"

# Ensure output directories exist
mkdir -p "${pre_path}/${algorithm}"
mkdir -p "$(dirname "${INDEX_PREFIX_PATH}")" "$(dirname "${RESULT_PREFIX_PATH}")" "$(dirname "${log_file}")"

# --- Execution ---
case "$mode" in
  build)
    echo "Building index..."
    ./test/benchmark_mag "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${L}" "${R}" "${C}" "${R_IP}" "${M}" "${threshold}" "${INDEX_PREFIX_PATH}" "${KNNG_PATH}" "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
    ;;
  search)
    echo "Searching index..."
    for ef_search in "${efs[@]}"; do
      echo "========================================" | tee -a "${log_file}"
      echo "Running with efs: $ef_search" | tee -a "${log_file}"
      ./test/benchmark_mag "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${L}" "${R}" "${C}" "${R_IP}" "${M}" "${threshold}" "${INDEX_PREFIX_PATH}" "${KNNG_PATH}" "${RESULT_PREFIX_PATH}" "${ef_search}" | tee -a "${log_file}"
      python3 "${recall_path}" "${DATA_PRE_PATH}" "${PREFIX}" "${TRAIN_NAME}" "${TEST_NAME}" "${algorithm}" "${K}" "${type}" "${result_name}" "${pre_path}" | tee -a "${log_file}"
      echo "========================================" | tee -a "${log_file}"
    done
    ;;
  *)
    echo "Invalid mode. Use 'build' or 'search'."
    exit 1
    ;;

esac
