#!/bin/bash
# VLLM 高性能推理

CUDA_VISIBLE_DEVICES="3"
INFER_BACKEND="vllm"
MODEL="/home/qianq/mycodes/llm/results/swift-projector-llava-med-v1.5-mistral-7b-EPOCH=5-LR=1e-6-DATASET=LIDC-IDRI-MLLM-CLF-EN-ATTRS/v2-20250809-200642/checkpoint-820"

DATASET_NAME="LIDC-IDRI-MLLM-CLF-EN-ATTRS"
DATASET="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img/${DATASET_NAME}"
MAX_BATCH_SIZE=16
MAX_NEW_TOKENS=512
TEMPERATURE=0.7 # sft模型的Temperature应该低一些 0.6
TOP_P=0.9
WRITE_BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=1
RESULT_PATH="${MODEL}/inference_${DATASET_NAME}.jsonl"
METRIC="acc"

echo "=== VLLM 高性能推理 ==="
echo "使用双GPU: $CUDA_VISIBLE_DEVICES"
echo "张量并行: $TENSOR_PARALLEL_SIZE"
echo "大批量处理: $MAX_BATCH_SIZE"
echo "======================"

python inference.py \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --infer_backend "$INFER_BACKEND" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --write_batch_size "$WRITE_BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --result_path "$RESULT_PATH" \
    --metric "$METRIC"

echo "VLLM推理完成!"