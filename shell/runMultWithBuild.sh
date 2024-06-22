##!/bin/bash
#
## Initialize configuration
#bash ./config/init_config.sh
#
## Set root directory
#ROOT_DIR="../"
#
## Create log directory if it doesn't exist
#LOG_DIR="${ROOT_DIR}/log"
#mkdir -p $LOG_DIR
#
## Define programs and their corresponding log files
#declare -A programs
#programs=(
#  ["edge_connected_replaced_update_7_full_coverage_gist"]="edge_connected_replaced_update_7_full_coverage_gist.log"
#  ["edge_connected_replaced_update_8_full_coverage_gist"]="edge_connected_replaced_update_8_full_coverage_gist.log"
#  ["edge_connected_replaced_update_9_full_coverage_gist"]="edge_connected_replaced_update_9_full_coverage_gist.log"
#  ["edge_connected_replaced_update_10_full_coverage_gist"]="edge_connected_replaced_update_10_full_coverage_gist.log"
#  ["replaced_update_full_coverage_gist"]="replaced_update_full_coverage_gist.log"
#  ["edge_connected_replaced_update_7_full_coverage_sift"]="edge_connected_replaced_update_7_full_coverage_sift.log"
#  ["edge_connected_replaced_update_8_full_coverage_sift"]="edge_connected_replaced_update_8_full_coverage_sift.log"
#  ["edge_connected_replaced_update_9_full_coverage_sift"]="edge_connected_replaced_update_9_full_coverage_sift.log"
#  ["edge_connected_replaced_update_10_full_coverage_sift"]="edge_connected_replaced_update_10_full_coverage_sift.log"
#  ["replaced_update_full_coverage_sift"]="replaced_update_full_coverage_sift.log"
#  ["edge_connected_replaced_update_7_full_coverage_imageNet"]="edge_connected_replaced_update_7_full_coverage_imageNet.log"
#  ["edge_connected_replaced_update_8_full_coverage_imageNet"]="edge_connected_replaced_update_8_full_coverage_imageNet.log"
#  ["edge_connected_replaced_update_9_full_coverage_imageNet"]="edge_connected_replaced_update_9_full_coverage_imageNet.log"
#  ["edge_connected_replaced_update_10_full_coverage_imageNet"]="edge_connected_replaced_update_10_full_coverage_imageNet.log"
#  ["replaced_update_full_coverage_imageNet"]="replaced_update_full_coverage_imageNet.log"
#)
#
## Start programs and redirect logs
#for program in "${!programs[@]}"; do
#  LOG_FILE="${LOG_DIR}/${programs[$program]}"
#  touch $LOG_FILE
#  nohup ./${ROOT_DIR}/cmake-build-replaced_update_valid/src/$program $ROOT_DIR > $LOG_FILE 2>&1 &
#done
#
#echo "All programs have been started and are running in the background. Log files are stored in the ${LOG_DIR} directory."



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
  "edge_connected_replaced_update_7_full_coverage_gist"
  "edge_connected_replaced_update_8_full_coverage_gist"
  "edge_connected_replaced_update_9_full_coverage_gist"
  "edge_connected_replaced_update_10_full_coverage_gist"
  "replaced_update_full_coverage_gist"
  "edge_connected_replaced_update_7_full_coverage_sift"
  "edge_connected_replaced_update_8_full_coverage_sift"
  "edge_connected_replaced_update_9_full_coverage_sift"
  "edge_connected_replaced_update_10_full_coverage_sift"
  "replaced_update_full_coverage_sift"
  "edge_connected_replaced_update_7_full_coverage_imageNet"
  "edge_connected_replaced_update_8_full_coverage_imageNet"
  "edge_connected_replaced_update_9_full_coverage_imageNet"
  "edge_connected_replaced_update_10_full_coverage_imageNet"
  "replaced_update_full_coverage_imageNet"
)

log_files=(
  "edge_connected_replaced_update_7_full_coverage_gist.log"
  "edge_connected_replaced_update_8_full_coverage_gist.log"
  "edge_connected_replaced_update_9_full_coverage_gist.log"
  "edge_connected_replaced_update_10_full_coverage_gist.log"
  "replaced_update_full_coverage_gist.log"
  "edge_connected_replaced_update_7_full_coverage_sift.log"
  "edge_connected_replaced_update_8_full_coverage_sift.log"
  "edge_connected_replaced_update_9_full_coverage_sift.log"
  "edge_connected_replaced_update_10_full_coverage_sift.log"
  "replaced_update_full_coverage_sift.log"
  "edge_connected_replaced_update_7_full_coverage_imageNet.log"
  "edge_connected_replaced_update_8_full_coverage_imageNet.log"
  "edge_connected_replaced_update_9_full_coverage_imageNet.log"
  "edge_connected_replaced_update_10_full_coverage_imageNet.log"
  "replaced_update_full_coverage_imageNet.log"
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