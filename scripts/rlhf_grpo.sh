MAX_PIXELS=1003520 \
MODEL_PREFIX=/home/qianq/model/ \
MODEL=/home/qianq/model/llava-med-v1.5-mistral-7b \
EPOCH=1 \
DATASET_PREFIX=/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/ \
DATASET_NAME=attr-lidc \

swift rlhf \
    --rlhf_type grpo \
    --model /home/qianq/model/llava-med-v1.5-mistral-7b \
    --dataset /home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/${DATASET_NAME} \
    --split_dataset_ratio 0 \
    --reward_funcs reward_function.reward_function \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --temperature 1.0 \
    --top_p 0.9 \
    --top_k 50 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --model_type llava1_5_hf \
    --output_dir output \
    --num_iterations 1 \
    --beta 0.04