#!/bin/bash
# VLLM 高性能推理

CUDA_VISIBLE_DEVICES="2"
INFER_BACKEND="vllm"
MODEL="/home/qianq/mycodes/llm/results/swift-projector-llava-1.5-7b-hf-EPOCH=5-LR=1e-6/v1-20250711-140224/checkpoint-250"
DATASET="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
MAX_BATCH_SIZE=16
MAX_NEW_TOKENS=512
TEMPERATURE=0.7 # sft模型的Temperature应该低一些 0.6
TOP_P=0.9
WRITE_BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=1
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
    --write_batch_size "$WRITE_BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --result_path "$RESULT_PATH" \
    --metric "$METRIC"

echo "VLLM推理完成!"