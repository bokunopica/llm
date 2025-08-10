#!/bin/bash
# VLLM 高性能推理

# # 参数检查
# if [ $# -ne 2 ]; then
#     echo "错误: 必须提供两个参数！"
#     echo "用法: $0 <CUDA_VISIBLE_DEVICES> <DATASET_NAME>"
#     echo "示例:"
#     echo "  $0 0 lidc"
#     echo "  $0 1 lidc-v2"
#     exit 1
# fi

# 读取参数
# CUDA_VISIBLE_DEVICES=$1
# DATASET_NAME=$2
CUDA_VISIBLE_DEVICES="3"
DATASET_NAME="attr-lidc"

# 固定推理配置
INFER_BACKEND="vllm"
MODEL="/home/qianq/mycodes/llm/results/swift-projector-llava-med-v1.5-mistral-7b-EPOCH=5-LR=1e-6-DATASET=attr-lidc/v0-20250810-203841/checkpoint-820"
DATASET_PREFIX="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/"
DATASET="${DATASET_PREFIX}${DATASET_NAME}"

MAX_BATCH_SIZE=16
MAX_NEW_TOKENS=512
TEMPERATURE=0.7   # sft模型的Temperature应该低一些 0.6
TOP_P=0.9
WRITE_BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=1
RESULT_PATH="${MODEL}/inference_${DATASET_NAME}.jsonl"
METRIC="acc"

# 检查数据集路径
if [ ! -d "$DATASET" ]; then
    echo "错误: 数据集路径不存在: $DATASET"
    exit 1
fi

# 打印配置信息
echo "=== VLLM 高性能推理 ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "数据集: $DATASET"
echo "结果输出: $RESULT_PATH"
echo "张量并行: $TENSOR_PARALLEL_SIZE"
echo "大批量处理: $MAX_BATCH_SIZE"
echo "======================"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# 运行推理
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
