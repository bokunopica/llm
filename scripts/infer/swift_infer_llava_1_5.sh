#!/bin/bash
# VLLM 高性能推理

CUDA_VISIBLE_DEVICES="1,3"
INFER_BACKEND="vllm"
MODEL="/home/qianq/mycodes/llm/results/swift-projector-llava-1.5-7b-hf-epoch=5-lr=1e-5/v0-20250710-154830/checkpoint-250"
DATASET="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
MAX_BATCH_SIZE=16
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
VAL_DATASET_SAMPLE=500
WRITE_BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=2
RESULT_PATH="${MODEL}/inference_results.jsonl"
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
    --val_dataset_sample "$VAL_DATASET_SAMPLE" \
    --write_batch_size "$WRITE_BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --result_path "$RESULT_PATH" \
    --metric "$METRIC"

echo "VLLM推理完成!"