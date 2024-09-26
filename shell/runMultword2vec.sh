#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize configuration
bash "$ROOT_DIR/config/init_config.sh"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/full_coverage"
mkdir -p $LOG_DIR

echo $ROOT_DIR

# Define programs and their corresponding log files using two parallel arrays
programs=(
  "edge_connected_replaced_update_7_full_coverage_word2vec"
  "edge_connected_replaced_update_8_full_coverage_word2vec"
  "edge_connected_replaced_update_9_full_coverage_word2vec"
  "edge_connected_replaced_update_10_full_coverage_word2vec"
  "replaced_update_full_coverage_word2vec"
  "faiss_full_coverage_word2vec"
)

log_files=(
  "edge_connected_replaced_update_7_full_coverage_word2vec.log"
  "edge_connected_replaced_update_8_full_coverage_word2vec.log"
  "edge_connected_replaced_update_9_full_coverage_word2vec.log"
  "edge_connected_replaced_update_10_full_coverage_word2vec.log"
  "replaced_update_full_coverage_word2vec.log"
  "faiss_full_coverage_word2vec.log"
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