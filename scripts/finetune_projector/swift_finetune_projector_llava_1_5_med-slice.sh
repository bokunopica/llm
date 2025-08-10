#!/bin/bash

# 查找可用端口（10000~65535随机）
get_free_port() {
    while true; do
        PORT=$(( ( RANDOM % 55535 ) + 10000 ))
        # 检查端口是否被占用
        if ! ss -tuln | grep -q ":$PORT "; then
            echo "$PORT"
            return
        fi
    done
}

# 计算GPU数量
count_gpus() {
    local gpu_string=$1
    # 移除空格并按逗号分割计算数量
    echo "$gpu_string" | tr ',' '\n' | wc -l
}

# 参数检查
if [ $# -ne 2 ]; then
    echo "错误: 必须提供两个参数！"
    echo "用法: $0 <CUDA_VISIBLE_DEVICES> <DATASET_SUFFIX>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$1
DATASET_SUFFIX=$2
MASTER_PORT=$(get_free_port)   # 动态获取可用端口
GPU_COUNT=$(count_gpus "$CUDA_VISIBLE_DEVICES")

MODEL_NAME="llava-med-v1.5-mistral-7b"
MODEL="/home/qianq/model/${MODEL_NAME}"
DATASET_PREFIX="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/"
DATASET="${DATASET_PREFIX}${DATASET_SUFFIX}"

NUM_TRAIN_EPOCHS=1
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=1e-6
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
USE_LORA=True
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.1
LOGGING_STEPS=10
SAVE_STEPS=250
EVAL_STEPS=250
SAVE_TOTAL_LIMIT=2
SEED=42
RUN_NAME="swift-projector-${MODEL_NAME}-EPOCH=${NUM_TRAIN_EPOCHS}-LR=${LEARNING_RATE}-DATASET=${DATASET_SUFFIX}"
OUTPUT_DIR="./results/${RUN_NAME}"

# 检查数据集路径
if [ ! -d "$DATASET" ]; then
    echo "错误: 数据集路径不存在: $DATASET"
    exit 1
fi

echo "=== Fine-tuning Configuration ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Master Port: $MASTER_PORT"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "================================="

mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

python finetune_projector.py \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --seed "$SEED" \
    --run_name "$RUN_NAME"

echo "Fine-tuning completed!"
