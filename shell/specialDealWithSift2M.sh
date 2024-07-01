#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/specialDealWithSift2M"
mkdir -p $LOG_DIR

#cd $ROOT_DIR

python3 ${ROOT_DIR}/src/python/split_sift_2M/split_sift_2M.py $ROOT_DIR
python3 ${ROOT_DIR}/src/python/split_sift_2M/change_query_bvecs_to_fvecs.py $ROOT_DIR


programs=(
  "compute_sift_groundtruth"
)

log_files=(
  "compute_sift_groundtruth.log"
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