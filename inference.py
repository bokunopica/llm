import argparse
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def parse_args():
    parser = argparse.ArgumentParser(description="使用微调的模型进行图像-文本到文本推理")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="微调模型的目录路径")
    parser.add_argument("--image_path", type=str, required=True,
                        help="输入图像的路径")
    parser.add_argument("--prompt", type=str, default="<image>\nDescribe this medical image.",
                        help="图像提示语")
    parser.add_argument("--system_prompt", type=str, 
                        default="You are a helpful medical assistant.",
                        help="系统提示语")
    parser.add_argument("--max_length", type=int, default=512,
                        help="生成文本的最大长度")
    parser.add_argument("--use_fp16", action="store_true",
                        help="使用fp16进行推理以提高速度")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载模型和处理器
    print(f"正在从 {args.model_dir} 加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.use_fp16 and torch.cuda.is_available() else torch.float32
    
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
    ).to(device)
    
    # 加载图像
    print(f"正在加载图像: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    
    # 准备输入
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt}
    ]
    
    try:
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{args.system_prompt}\nUser: {args.prompt}\nAssistant:"
    
    # 处理输入
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # 生成响应
    print("生成响应中...")
    output = model.generate(
        **inputs,
        max_length=args.max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # 解码输出
    generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
    
    # 提取模型回答
    answer = generated_text.split("Assistant: ")[-1].strip()
    
    print("\n===== 生成结果 =====")
    print(answer)

if __name__ == "__main__":
    main()