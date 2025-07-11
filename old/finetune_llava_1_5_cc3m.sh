#!/bin/bash
# 定义公共变量
MODEL_PATH="/home/qianq/model/llava-1.5-7b-hf"
dataset_type="CC3M"
DATASET_DIR="data/image-text-to-text/LLaVA-CC3M-Pretrain-595K"
OUTPUT_DIR="./results/llava-1.5-7b-hf-projector-only-cc3m"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"
SYSTEM_PROMPT="You are a imaging analysis assistant."
DEEPSPEED_CONFIG="ds_config_zero2.json"

# 训练参数
LEARNING_RATE="5e-5"
NUM_EPOCHS="1"
BATCH_SIZE="1"        # DeepSpeed可以处理更小的批次
GRAD_ACCUM_STEPS="8"  # 增加梯度累积以补偿小批次
EVAL_STEPS="500"
SAVE_STEPS="500"
LOGGING_STEPS="20"

# 使用DeepSpeed进行训练
deepspeed --include localhost:2,3 finetune.py \
  --fp16 \
  --dataset_type "${dataset_type}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --train_type "projector_only" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --use_tensorboard \
  --tensorboard_dir "${TENSORBOARD_DIR}" \
  --system_prompt "${SYSTEM_PROMPT}" \
  --save_total_limit 1 \
  --load_best_model_at_end \
  --metric_for_best_model "eval_loss"