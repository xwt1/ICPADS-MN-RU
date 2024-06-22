
#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize configuration
bash "$ROOT_DIR/config/init_config.sh"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/new_insert"
mkdir -p $LOG_DIR

echo $ROOT_DIR

# Define programs and their corresponding log files using two parallel arrays
programs=(
  "edge_connected_replaced_update_7_new_insert_sift_2M"
  "edge_connected_replaced_update_8_new_insert_sift_2M"
  "edge_connected_replaced_update_9_new_insert_sift_2M"
  "edge_connected_replaced_update_10_new_insert_sift_2M"
  "replaced_update_new_insert_sift_2M"
)

log_files=(
      "edge_connected_replaced_update_7_new_insert_sift_2M.log"
      "edge_connected_replaced_update_8_new_insert_sift_2M.log"
      "edge_connected_replaced_update_9_new_insert_sift_2M.log"
      "edge_connected_replaced_update_10_new_insert_sift_2M.log"
      "replaced_update_new_insert_sift_2M.log"
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