#!/bin/bash

# 定义公共变量
MODEL_PATH="/home/pico/model/llava-1.5-7b-hf"
DATASET_DIR="data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
OUTPUT_DIR="./results/llava-1.5-7b-hf-lora"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
SYSTEM_PROMPT="You are a professional medical imaging analysis assistant."

# 训练参数
LEARNING_RATE="5e-5"
NUM_EPOCHS="5"
BATCH_SIZE="1"
GRAD_ACCUM_STEPS="8"
EVAL_STEPS="2000"
SAVE_STEPS="2000"
LOGGING_STEPS="20"

# 使用LoRA进行高效微调
python finetune.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --lora \
  --fp16 \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --use_tensorboard \
  --tensorboard_dir "${TENSORBOARD_DIR}" \
  --system_prompt "${SYSTEM_PROMPT}"