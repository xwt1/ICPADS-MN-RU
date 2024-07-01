#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize configuration
bash "$ROOT_DIR/config/init_config.sh"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/generateIndexGroundTruth"
mkdir -p $LOG_DIR

# Define programs and their corresponding log files using two parallel arrays
programs=(
  "hnsw_prime_generate_sift_index"
  "hnsw_prime_generate_gist_index"
  "hnsw_prime_generate_imageNet_index"
  "hnsw_prime_generate_sift2M_index"
  "compute_imageNet2M_groundtruth"
  "compute_imagenet_groundtruth"
  "compute_gist_1M_groundTruth"
)

log_files=(
  "hnsw_prime_generate_sift_index.log"
  "hnsw_prime_generate_gist_index.log"
  "hnsw_prime_generate_imageNet_index.log"
  "hnsw_prime_generate_sift2M_index.log"
  "compute_imageNet2M_groundtruth.log"
  "compute_imagenet_groundtruth.log"
  "compute_gist_1M_groundTruth.log"
)


# Start programs sequentially and redirect logs
for i in "${!programs[@]}"; do
  program="${programs[$i]}"
  LOG_FILE="${LOG_DIR}/${log_files[$i]}"
  touch $LOG_FILE
  echo ${ROOT_DIR}/build/src/$program
  ${ROOT_DIR}/build/src/$program $ROOT_DIR > $LOG_FILE 2>&1
  echo "done ${program}"
done

echo "All programs have been started sequentially. Log files are stored in the ${LOG_DIR} directory."