#!/bin/bash
set -e

# -----------------------------
# 配置参数
# -----------------------------
MODEL_PATH="/root/autodl-tmp/models/InternVL2-2B"
MODEL_NAME="InternVL2-2B"
DATA_NAME="VSI_Bench"
DATA_PATH="/root/autodl-tmp/data/VSI_Bench/"
DATA_FILE_NAME="eval_vsibench.json"
OUTPUT_DIR="/root/verl_evaluate/outputs/VSI_Bench"
REWARD_FUNCTION="vsi_compute_score"

# question_type 列表
QUESTION_TYPES=("object_rel_direction_easy" "object_rel_direction_medium" "object_rel_direction_hard"
                "object_rel_distance" "route_planning" "obj_appearance_order"
                 "object_abs_distance"  "object_counting"  "object_size_estimation"  "room_size_estimation")

# -----------------------------
# 循环 question_type
# -----------------------------
for QTYPE in "${QUESTION_TYPES[@]}"; do
    echo "=== Running question_type: $QTYPE ==="
    # -------------------------
    # 1. Generation
    # -------------------------
    echo "Running generation for $QTYPE..."
    python evaluation/main_generation.py \
        model.path="$MODEL_PATH" \
        model.name="$MODEL_NAME" \
        data.name="$DATA_NAME" \
        data.path="$DATA_PATH" \
        data.file_name="$DATA_FILE_NAME" \
        data.output_path="$OUTPUT_DIR/" \
        data.question_type="$QTYPE"

done

# -----------------------------
# 2. Evaluation: 批量评测 OUTPUT_DIR 下所有 parquet 文件
# -----------------------------
echo "=== Running evaluation on all generated files ==="

python evaluation/main_eval.py \
    data.path="$OUTPUT_DIR" \
    custom_reward_function.name="$REWARD_FUNCTION"

echo "Evaluation finished. Results saved in $OUTPUT_DIR"

echo "All question types finished."
