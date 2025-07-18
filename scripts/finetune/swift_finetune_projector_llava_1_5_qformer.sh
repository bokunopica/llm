#!/bin/bash

# 设置参数
CUDA_VISIBLE_DEVICES="1,3"
MODEL_NAME="llava-1.5-7b-hf"
MODEL="/home/qianq/model/${MODEL_NAME}"
DATASET="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"

NUM_TRAIN_EPOCHS=5
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=16
LEARNING_RATE=1e-6
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.1
USE_LORA=false
LORA_RANK=64
LORA_ALPHA=128
LORA_DROPOUT=0.1
LOGGING_STEPS=10
SAVE_STEPS=250
EVAL_STEPS=250
SAVE_TOTAL_LIMIT=2
SEED=42

# Q-former 配置
USE_QFORMER=true
NUM_QUERY_TOKENS=32
QFORMER_HIDDEN_SIZE=768
QFORMER_NUM_LAYERS=6
QFORMER_NUM_HEADS=12

RUN_NAME="swift-projector-${MODEL_NAME}-qformer-EPOCH=${NUM_TRAIN_EPOCHS}-LR=${LEARNING_RATE}"
OUTPUT_DIR="./results/${RUN_NAME}"

# 打印配置信息
echo "=== Fine-tuning Configuration with Q-former ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Training Epochs: $NUM_TRAIN_EPOCHS"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Use LoRA: $USE_LORA"
echo "Run Name: $RUN_NAME"
echo "--- Q-former Configuration ---"
echo "Use Q-former: $USE_QFORMER"
echo "Query Tokens: $NUM_QUERY_TOKENS"
echo "Hidden Size: $QFORMER_HIDDEN_SIZE"
echo "Num Layers: $QFORMER_NUM_LAYERS"
echo "Num Heads: $QFORMER_NUM_HEADS"
echo "=============================================="

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 运行训练 - 添加Q-former参数
torchrun --nproc_per_node=2 finetune_projector.py \
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
    --use_lora "$USE_LORA" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --seed "$SEED" \
    --run_name "$RUN_NAME" \
    --use_qformer "$USE_QFORMER" \
    --num_query_tokens "$NUM_QUERY_TOKENS" \
    --qformer_hidden_size "$QFORMER_HIDDEN_SIZE" \
    --qformer_num_layers "$QFORMER_NUM_LAYERS" \
    --qformer_num_heads "$QFORMER_NUM_HEADS"

echo "Q-former Fine-tuning completed!"