#!/bin/bash
# filepath: /home/qianq/mycodes/llm/test.sh

# 设置所有必要的参数
CUDA_VISIBLE_DEVICES="2,3"
INFER_BACKEND="pt"
MODEL="/home/qianq/model/llava-med-v1.5-mistral-7b"
MODEL_TYPE="llava1_5_hf"
ADAPTERS="/home/qianq/mycodes/llm/results/llava-med-v1.5-mistral-7b-EPOCH=1-LR=-DATASET=lidc/v1-20250811-062622/checkpoint-323"
DATASET="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/lidc"

# 推理配置
MAX_BATCH_SIZE=8
MAX_NEW_TOKENS=512
TEMPERATURE=0
TOP_P=0.9

# 数据集配置
VAL_DATASET_SAMPLE=-1
WRITE_BATCH_SIZE=32

# 分布式配置
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1

# 输出配置
RESULT_PATH="./inference_results.jsonl"
METRIC="acc"

# 运行推理
python inference.py \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --infer_backend "$INFER_BACKEND" \
    --model "$MODEL" \
    --model_type "$MODEL_TYPE" \
    --adapters "$ADAPTERS" \
    --dataset "$DATASET" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --val_dataset_sample "$VAL_DATASET_SAMPLE" \
    --write_batch_size "$WRITE_BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" \
    --result_path "$RESULT_PATH" \
    --metric "$METRIC"