# 使用LoRA进行高效微调
python train_image_to_text.py \
  --dataset_dir data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN \
  --output_dir ./results/llava-medical-lora \
  --lora \
  --fp16 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --system_prompt "You are a professional medical imaging analysis assistant."
