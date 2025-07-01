# 使用LoRA进行高效微调
python finetune.py \
  --dataset_dir data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN \
  --output_dir ./results/llava-1.5-7b-hf-lora \
  --model_name_or_path /home/pico/model/llava-1.5-7b-hf \
  --lora \
  --fp16 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 20 \
  --use_tensorboard \
  --tensorboard_dir ./results/llava-medical-lora/tensorboard \
  --system_prompt "You are a professional medical imaging analysis assistant."