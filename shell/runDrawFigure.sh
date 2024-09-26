#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set root directory to the parent directory of the script directory
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Create log directory if it doesn't exist
LOG_DIR="${ROOT_DIR}/log/drawFigure"
mkdir -p $LOG_DIR

PYTHON_SCRIPT_DIR="${ROOT_DIR}/src/python"

echo $ROOT_DIR

# Define programs and their corresponding log files using two parallel arrays
programs=(
  "/draw_full_coverage/draw_gist/draw_replaced_update_insert_time.py"
  "/draw_full_coverage/draw_gist/draw_unreachable_point.py"
  "/draw_full_coverage/draw_imageNet/draw_replaced_update_insert_time.py"
  "/draw_full_coverage/draw_imageNet/draw_unreachable_point.py"
  "/draw_full_coverage/draw_sift/draw_replaced_update_insert_time.py"
  "/draw_full_coverage/draw_sift/draw_unreachable_point.py"
  "/draw_full_coverage/draw_word2vec/draw_replaced_update_insert_time.py"
  "/draw_full_coverage/draw_word2vec/draw_unreachable_point.py"

  "/draw_random/draw_gist/draw_replaced_update_insert_time.py"
  "/draw_random/draw_gist/draw_unreachable_point.py"
  "/draw_random/draw_imageNet/draw_replaced_update_insert_time.py"
  "/draw_random/draw_imageNet/draw_unreachable_point.py"
  "/draw_random/draw_sift/draw_replaced_update_insert_time.py"
  "/draw_random/draw_sift/draw_unreachable_point.py"
  "/draw_random/draw_word2vec/draw_replaced_update_insert_time.py"
  "/draw_random/draw_word2vec/draw_unreachable_point.py"

  "/draw_new_insert/draw_sift2M/draw_replaced_update_insert_time.py"
  "/draw_new_insert/draw_sift2M/draw_unreachable_point.py"

  "/draw_full_coverage/draw_gist_end_recall.py"
  "/draw_full_coverage/draw_imageNet_2M_end_recall.py"
  "/draw_full_coverage/draw_gist_end_recall_100.py"
  "/draw_full_coverage/draw_imageNet_2M_end_recall_10.py"
  "/draw_full_coverage/draw_sift_end_recall_100.py"
  "/draw_full_coverage/draw_word2vec_end_recall_100.py"

  "/draw_random/draw_gist_end_recall.py"
  "/draw_random/draw_imageNet_2M_end_recall.py"
  "/draw_random/draw_gist_end_recall_100.py"
  "/draw_random/draw_imageNet_2M_end_recall_10.py"
  "/draw_random/draw_sift_end_recall_100.py"


  "/draw_figure1/draw_figure1.py"

  "/draw_figure2/draw_figure2_extream.py"
  "/draw_figure2/draw_figure2_extream_recall.py"

  "/draw_backup/draw_backup.py"
)

log_files=(
      "full_coverage_gist_draw_replaced_update_insert_time.log"
      "full_coverage_gist_draw_unreachable_point.log"
      "full_coverage_imageNet_draw_replaced_update_insert_time.log"
      "full_coverage_imageNet_draw_unreachable_point.log"
      "full_coverage_sift_draw_replaced_update_insert_time.log"
      "full_coverage_sift_draw_unreachable_point.log"

      "random_gist_draw_replaced_update_insert_time.log"
      "random_coverage_gist_draw_unreachable_point.log"
      "random_imageNet_draw_replaced_update_insert_time.log"
      "random_coverage_imageNet_draw_unreachable_point.log"
      "random_coverage_sift_draw_replaced_update_insert_time.log"
      "random_coverage_sift_draw_unreachable_point.log"

      "new_insert_draw_replaced_update_insert_time.log"
      "new_insert_draw_unreachable_point.log"

      "full_coverage_draw_gist_end_recall.log"
      "full_coverage_draw_imageNet_2M_end_recall.log"
      "draw_gist_end_recall_100.log"
      "draw_imageNet_2M_end_recall_10.log"
      "draw_sift_end_recall_100.log"

      "random_draw_gist_end_recall.log"
      "random_draw_imageNet_2M_end_recall.log"
      "draw_gist_end_recall_100"
      "draw_imageNet_2M_end_recall_10"
      "draw_sift_end_recall_100"

      "draw_figure1.log"

      "draw_figure2_extream.log"
      "draw_figure2_extream_recall.log"

      "draw_backup.log"
)

# Start programs sequentially and redirect logs
for i in "${!programs[@]}"; do
  program="${programs[$i]}"
  LOG_FILE="${LOG_DIR}/${log_files[$i]}"
  touch $LOG_FILE
  echo  ${PYTHON_SCRIPT_DIR}$program
  python ${PYTHON_SCRIPT_DIR}$program $ROOT_DIR > $LOG_FILE 2>&1
  echo "done ${program}"
done

echo "All programs have been started sequentially. Log files are stored in the ${LOG_DIR} directory."