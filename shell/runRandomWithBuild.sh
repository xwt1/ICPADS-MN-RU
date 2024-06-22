
#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Initialize configuration
bash "$ROOT_DIR/config/init_config.sh"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/random"
mkdir -p $LOG_DIR

echo $ROOT_DIR

# Define programs and their corresponding log files using two parallel arrays
programs=(
  "generate_gist_random_data"
  "generate_imageNet_random_data"
  "generate_sift_random_data"
  "edge_connected_replaced_update_7_random_gist"
  "edge_connected_replaced_update_8_random_gist"
  "edge_connected_replaced_update_9_random_gist"
  "edge_connected_replaced_update_10_random_gist"
  "replaced_update_random_gist"
  "edge_connected_replaced_update_7_random_sift"
  "edge_connected_replaced_update_8_random_sift"
  "edge_connected_replaced_update_9_random_sift"
  "edge_connected_replaced_update_10_random_sift"
  "replaced_update_random_sift"
  "edge_connected_replaced_update_7_random_imageNet"
  "edge_connected_replaced_update_8_random_imageNet"
  "edge_connected_replaced_update_9_random_imageNet"
  "edge_connected_replaced_update_10_random_imageNet"
  "replaced_update_random_imageNet"
)

log_files=(
    "generate_gist_random_data.log"
    "generate_imageNet_random_data.log"
    "generate_sift_random_data.log"
    "edge_connected_replaced_update_7_random_gist.log"
    "edge_connected_replaced_update_8_random_gist.log"
    "edge_connected_replaced_update_9_random_gist.log"
    "edge_connected_replaced_update_10_random_gist.log"
    "replaced_update_random_gist.log"
    "edge_connected_replaced_update_7_random_sift.log"
    "edge_connected_replaced_update_8_random_sift.log"
    "edge_connected_replaced_update_9_random_sift.log"
    "edge_connected_replaced_update_10_random_sift.log"
    "replaced_update_random_sift.log"
    "edge_connected_replaced_update_7_random_imageNet.log"
    "edge_connected_replaced_update_8_random_imageNet.log"
    "edge_connected_replaced_update_9_random_imageNet.log"
    "edge_connected_replaced_update_10_random_imageNet.log"
    "replaced_update_random_imageNet.log"
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