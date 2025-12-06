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
make -j8 benchmark_dblsh

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="dblsh"
L=5
K_dblsh=10
type="nn"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_L${L}_K${K_dblsh}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_L${L}_K${K_dblsh}"

# Ensure output directories exist
mkdir -p "${pre_path}/${algorithm}"
mkdir -p "$(dirname "${INDEX_PREFIX_PATH}")" "$(dirname "${RESULT_PREFIX_PATH}")" "$(dirname "${log_file}")"

beta_list=(0.8 0.9 0.92 0.94 0.96 0.98)
for beta in "${beta_list[@]}"; do
  echo "========================================" | tee -a "${log_file}"
  echo "Running with beta: $beta" | tee -a "${log_file}"
  ./test/benchmark_dblsh "${BASE_PATH}" "${QUERY_FILE}" "${DATA_DIM}" "${K}" "${L}" "${K_dblsh}" 1.5 "$beta" 0.1 "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
  python3 "${recall_path}" "${DATA_PRE_PATH}" "${PREFIX}" "${TRAIN_NAME}" "${TEST_NAME}" "${algorithm}" "${K}" "${type}" "${result_name}" "${pre_path}" | tee -a "${log_file}"
  echo "========================================" | tee -a "${log_file}"
done
