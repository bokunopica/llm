python inference.py \
  --model_dir ./results/llava-medical \
  --image_path sample_image.jpg \
  --prompt "<image>\n详细描述这张医学影像。" \
  --system_prompt "你是一个专业的放射科医生，擅长肺部CT影像分析。" \
  --use_fp16