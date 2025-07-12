import os
import argparse
import torch
from swift.llm import sft_main, TrainArguments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA model projector with Swift"
    )

    # 环境配置
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="1,3",
        help="CUDA visible devices (default: 1,3)",
    )

    # 模型和数据配置
    parser.add_argument(
        "--model",
        type=str,
        default="/home/qianq/model/llava-1.5-7b-hf",
        help="Model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="llava1_6_mistral_hf",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN",
        help="Dataset path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/llava-1.5-7b-hf-swift-projector",
        help="Output directory",
    )

    # 训练配置
    parser.add_argument(
        "--num_train_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per device training batch size",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Per device evaluation batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # LoRA配置
    parser.add_argument(
        "--use_lora", type=bool, default=False, help="Use LoRA for training"
    )
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    # 保存和日志配置
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument(
        "--save_total_limit", type=int, default=2, help="Save total limit"
    )

    # 其他配置
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_name",
        type=str,
        default="llava-1.5-7b-lidc-projector",
        help="Run name for logging",
    )

    # 视觉处理器配置
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Patch size for vision processor",
    )

    return parser.parse_args()


def setup_processor_config(model_path, patch_size, image_size):
    """设置处理器配置"""
    import json

    # 读取处理器配置
    processor_config_path = os.path.join(model_path, "preprocessor_config.json")
    if os.path.exists(processor_config_path):
        with open(processor_config_path, "r") as f:
            config = json.load(f)

        # 更新配置
        if "vision_config" in config:
            config["vision_config"]["patch_size"] = patch_size
            config["vision_config"]["image_size"] = image_size
        else:
            config["patch_size"] = patch_size
            config["image_size"] = image_size

        # 保存更新后的配置
        with open(processor_config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"已更新处理器配置: patch_size={patch_size}, image_size={image_size}")

    # 检查并更新模型配置
    model_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            config = json.load(f)

        # 更新视觉配置
        if "vision_config" in config:
            config["vision_config"]["patch_size"] = patch_size
            config["vision_config"]["image_size"] = image_size

        # 保存更新后的配置
        with open(model_config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"已更新模型配置: vision_config.patch_size={patch_size}")


def main():
    args = parse_args()

    # 初始化分布式训练
    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Initialized distributed training. Local rank: {local_rank}")

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 只在主进程打印配置信息
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"使用GPU: {args.cuda_visible_devices}")
        print(f"模型路径: {args.model}")
        print(f"数据集路径: {args.dataset}")
        print(f"输出目录: {args.output_dir}")
        print(f"训练轮数: {args.num_train_epochs}")
        print(f"批大小: {args.per_device_train_batch_size}")
        print(f"学习率: {args.learning_rate}")
        print(f"训练类型: 投影器微调")
        print("-" * 50)

    result = sft_main(
        TrainArguments(
            # 模型配置
            model=args.model,
            train_type="lora" if args.use_lora else "ALL",
            # 数据集配置
            dataset=[args.dataset],
            # 训练配置
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            # 冻结配置 - 只训练投影器
            freeze_llm=True,
            freeze_vit=True,
            freeze_aligner=True if args.use_lora else False,
            # 精度和优化
            torch_dtype="bfloat16",
            fp16=False,
            bf16=True,
            dataloader_num_workers=0,
            # 保存和日志
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            eval_strategy="steps",
            # 分布式训练配置
            ddp_backend="nccl",
            ddp_find_unused_parameters=True,
            # 其他配置
            seed=args.seed,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            run_name=args.run_name,
            dataloader_pin_memory=False,
            model_type=args.model_type,
        )
    )

    # 只在主进程打印结果
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("投影器微调完成!")

        # 处理训练结果
        if isinstance(result, dict):
            checkpoint_dir = result.get("output_dir", args.output_dir)
            print(f"最佳模型保存在: {checkpoint_dir}")

            if "train_runtime" in result:
                print(f"训练用时: {result['train_runtime']:.2f} 秒")
            if "train_loss" in result:
                print(f"训练损失: {result['train_loss']:.4f}")
            if "eval_loss" in result:
                print(f"验证损失: {result['eval_loss']:.4f}")
        else:
            print(f"最佳模型保存在: {args.output_dir}")

    return result


if __name__ == "__main__":
    result = main()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"训练结果: {result}")
