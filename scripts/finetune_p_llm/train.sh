# 2 * 76GiB
CUDA_VISIBLE_DEVICES=1,3 \
MAX_PIXELS=1003520 \
MODEL=/home/qianq/model/llava-med-v1.5-mistral-7b \
EPOCH=1 \
DATASET_PREFIX=/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/ \
DATASET_NAME=attr-lidc \

swift sft \
    --model /home/qianq/model/llava-med-v1.5-mistral-7b \
    --dataset '/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice/lidc' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 0 \
    --model_type llava1_5_hf \
    --output_dir results/${MODEL##*/}-EPOCH=${EPOCH}-LR=${LR}-DATASET=${DATASET_NAME}