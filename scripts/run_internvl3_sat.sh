#!/bin/bash
set -e

# -----------------------------
# 配置参数
# -----------------------------
MODEL_PATH="/root/autodl-tmp/models/InternVL3-2B"
MODEL_NAME="InternVL3-2B"
DATA_NAME="SAT"
DATA_PATH="/root/autodl-tmp/data/SAT/"
DATA_FILE_NAME="SAT_test.parquet"
OUTPUT_DIR="/root/verl_evaluate/outputs/SAT"
REWARD_FUNCTION="sat_compute_score"

# 1. Generation (Inference)
echo "Running generation with $MODEL_NAME ..."
python evaluation/main_generation.py \
    model.path="$MODEL_PATH" \
    model.name="$MODEL_NAME" \
    data.name="$DATA_NAME" \
    data.path="$DATA_PATH" \
    data.file_name="$DATA_FILE_NAME" \
    data.output_path="$OUTPUT_DIR/" \

# 2. Evaluation (Scoring)
echo "=== Running evaluation on all generated files ==="

python evaluation/main_eval.py \
    data.path="$OUTPUT_DIR" \
    custom_reward_function.name="$REWARD_FUNCTION"

echo "Evaluation finished. Results saved in $OUTPUT_DIR"

echo "All question types finished."