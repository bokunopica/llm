# 定义公共变量
MODEL_PATH="/home/qianq/model/llava-1.5-7b-hf"
DATASET_DIR="data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
OUTPUT_DIR="./results/llava-1.5-7b-hf-projector-only"
INFERENCE_OUTPUT_DIR="./inference_results"
SYSTEM_PROMPT="You are a professional medical imaging analysis assistant."
GPU_ID=1

# 推理参数
BATCH_SIZE=4
MAX_LENGTH=2048
TEMPERATURE=0.1
TOP_K=50
TOP_P=0.9

# 执行推理
CUDA_VISIBLE_DEVICES="${GPU_ID}" python inference.py \
  --model_dir "${OUTPUT_DIR}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${INFERENCE_OUTPUT_DIR}" \
  --is_test \
  --batch_size ${BATCH_SIZE} \
  --max_length ${MAX_LENGTH} \
  --use_fp16 \
  --temperature ${TEMPERATURE} \
  --top_k ${TOP_K} \
  --top_p ${TOP_P} \
  --system_prompt "${SYSTEM_PROMPT}"