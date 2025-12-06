

# --- Configuration ---
# Resolve project root (default to repo root inferred from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# Ensure commands run at the project root
cd "${PROJECT_ROOT}"

# --- Source Dataset Configuration ---
source "${PROJECT_ROOT}/scripts/config_dataset.sh"
# Allow selecting dataset via env DATASET, default to imagenet1k_avg
: "${DATASET:=imagenet1k_dinov2}"
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
make -j8 benchmark_hnsw

# --- Parameters & Paths ---
pre_path="${PROJECT_ROOT}/benchmark_index"
algorithm="hnsw"
mode=search
efc=256
M=32
efs=(100)
type="nn"

INDEX_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.index"
RESULT_PREFIX_PATH="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.result"
log_file="${pre_path}/${algorithm}/${PREFIX}_M${M}_L${efc}.log"
recall_path="${PROJECT_ROOT}/tools/recall_${DATASET_TYPE}.py"
result_name="${PREFIX}_M${M}_L${efc}"

# Ensure output directories exist (avoid tee failure under set -e -o pipefail)
mkdir -p "${pre_path}/${algorithm}"
# Double guard in case variables change in future
mkdir -p "$(dirname "${INDEX_PREFIX_PATH}")" "$(dirname "${RESULT_PREFIX_PATH}")" "$(dirname "${log_file}")"

# --- Execution ---
case "$mode" in
  build)
    echo "Building index..."
    ./test/benchmark_hnsw "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${efc}" "${M}" "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" | tee -a "${log_file}"
    ;;
  search)
    echo "Searching index..."
    for ef_search in "${efs[@]}"; do
      echo "========================================" | tee -a "${log_file}"
      echo "Running with efs: $ef_search" | tee -a "${log_file}"
      ./test/benchmark_hnsw "${BASE_PATH}" "${QUERY_FILE}" "${mode}" "${DATA_DIM}" "${K}" "${efc}" "${M}" "${INDEX_PREFIX_PATH}" "${RESULT_PREFIX_PATH}" "${ef_search}" | tee -a "${log_file}"
      python3 "${recall_path}" "${DATA_PRE_PATH}" "${PREFIX}" "${TRAIN_NAME}" "${TEST_NAME}" "${algorithm}" "${K}" "${type}" "${result_name}" "${pre_path}" | tee -a "${log_file}"
      echo "========================================" | tee -a "${log_file}"
    done
    ;;
  *)
    echo "Invalid mode. Use 'build' or 'search'."
    exit 1
    ;;
esac
