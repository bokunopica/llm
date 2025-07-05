import argparse
import os
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from data import LIDCClassificationDataset, TrainLlavaModelCollator
from torch.utils.data import DataLoader
from tqdm import tqdm

# 加载图像
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="使用微调的模型进行图像-文本到文本推理")
    parser.add_argument("--model_dir", type=str, required=True, help="微调模型的目录路径")
    parser.add_argument("--dataset_dir", type=str, required=True, help="数据集目录，应包含train.jsonl和test.jsonl")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="推理结果保存目录")
    parser.add_argument("--is_test", action="store_true", help="是否在测试集上进行推理")
    parser.add_argument("--batch_size", type=int, default=4, help="推理批次大小")
    parser.add_argument("--max_length", type=int, default=512, help="生成文本的最大长度")
    parser.add_argument("--use_fp16", action="store_true", help="使用fp16进行推理以提高速度")
    parser.add_argument("--temperature", type=float, default=0.1, help="生成文本的温度，较低值可提高一致性")
    parser.add_argument("--top_k", type=int, default=50, help="生成文本时考虑的最高概率词数")
    parser.add_argument("--top_p", type=float, default=0.9, help="生成文本时累积概率的阈值")
    parser.add_argument("--do_sample", action="store_true", help="是否启用采样生成")
    parser.add_argument("--system_prompt", type=str, default="You are a medical expert.", help="系统提示，用于指导模型生成")
    return parser.parse_args()


def build_prompt_with_image(messages):
    """
    将 messages 列表手动拼接为 LLaVA 兼容的 prompt 格式，包含 <image> 标记。
    例如:
    messages = [
        {"role": "system", "content": "You are a medical expert."},
        {"role": "user", "content": "What do you see in this image?"}
    ]
    """
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            prompt += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            prompt += "<|assistant|>\n" + message["content"].strip() + "\n"
    prompt += "<|assistant|>\n"  # 最后加个 assistant 提示符，引导模型生成
    return prompt


def main():
    args = parse_args()

    # 设置设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.use_fp16 and torch.cuda.is_available() else torch.float32

    # 加载模型和处理器
    print(f"正在从 {args.model_dir} 加载模型和处理器...")
    processor = LlavaProcessor.from_pretrained(args.model_dir)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)

    # 加载数据集
    dataset_type = "test" if args.is_test else "train"
    print(f"正在加载 {dataset_type} 数据集: {args.dataset_dir}")
    dataset = LIDCClassificationDataset(args.dataset_dir, is_train=not args.is_test)
    print(f"{dataset_type.capitalize()} 数据集大小: {len(dataset)}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{dataset_type}_results.txt")

    # 推理
    print("开始推理...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, (query, answer, image_path) in enumerate(tqdm(dataset, desc="推理中", unit="样本")):
            # 构建输入prompt，确保包含<image>标记
            messages = [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": query}
            ]
            print('messages:', messages)
            
            # prompt = processor.tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            prompt = build_prompt_with_image(messages)


            image = Image.open(image_path).convert("RGB")
            
            # 处理输入
            print('promt:', prompt)
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(device)
                        
            # 生成回答
            generate_kwargs = {
                "max_length": args.max_length,
                "pad_token_id": processor.tokenizer.eos_token_id,
            }
            
            # 只在启用采样时添加采样参数
            if args.do_sample:
                generate_kwargs.update({
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                })
            else:
                generate_kwargs["do_sample"] = False
            
            print('before generate')
            outputs = model.generate(
                **inputs,
                **generate_kwargs
            )
            
            # 解码输出
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # 提取回答部分
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text.strip()
            
            results.append(f"Sample {i}: {response}")
            print(f"Sample {i}: {response}")
            
            # 每100个样本打印一次进度
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(dataset)} 个样本")

    # 保存结果
    print(f"保存推理结果到 {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for text in results:
            f.write(text + "\n")

    print("推理完成！")


if __name__ == "__main__":
    main()
