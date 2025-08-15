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
        default="./results/llava-1.5-7b-hf-swift-qformer",
        help="Output directory",
    )

    # Q-former配置
    parser.add_argument(
        "--use_qformer",
        type=bool,
        default=False,
        help="Use Q-former instead of MLP projector",
    )
    parser.add_argument(
        "--num_query_tokens",
        type=int,
        default=32,
        help="Number of query tokens for Q-former",
    )
    parser.add_argument(
        "--qformer_hidden_size",
        type=int,
        default=768,
        help="Hidden size for Q-former",
    )
    parser.add_argument(
        "--qformer_num_layers",
        type=int,
        default=6,
        help="Number of transformer layers in Q-former",
    )
    parser.add_argument(
        "--qformer_num_heads",
        type=int,
        default=12,
        help="Number of attention heads in Q-former",
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
        "--learning_rate", type=float, default=1e-4, help="Learning rate for Q-former"
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
        default="llava-1.5-7b-lidc-qformer",
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


def setup_qformer_config(model_path, args):
    """设置Q-former配置"""
    import json

    # 读取模型配置
    model_config_path = os.path.join(model_path, "config.json")
    if os.path.exists(model_config_path):
        with open(model_config_path, "r") as f:
            config = json.load(f)

        # 添加Q-former配置
        qformer_config = {
            "use_qformer": args.use_qformer,
            "num_query_tokens": args.num_query_tokens,
            "hidden_size": args.qformer_hidden_size,
            "num_hidden_layers": args.qformer_num_layers,
            "num_attention_heads": args.qformer_num_heads,
            "intermediate_size": args.qformer_hidden_size * 4,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
        }
        
        config["qformer_config"] = qformer_config
        
        # 修改projector配置
        if "projector_config" in config:
            config["projector_config"]["projector_type"] = "qformer"
        else:
            config["projector_config"] = {"projector_type": "qformer"}

        # 保存更新后的配置
        with open(model_config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"已更新模型配置为Q-former:")
        print(f"  - 查询token数量: {args.num_query_tokens}")
        print(f"  - 隐藏层大小: {args.qformer_hidden_size}")
        print(f"  - Transformer层数: {args.qformer_num_layers}")
        print(f"  - 注意力头数: {args.qformer_num_heads}")


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

    # 只在主进程配置模型
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        # 设置Q-former配置
        if args.use_qformer:
            setup_qformer_config(args.model, args)
        
        print(f"使用GPU: {args.cuda_visible_devices}")
        print(f"模型路径: {args.model}")
        print(f"数据集路径: {args.dataset}")
        print(f"输出目录: {args.output_dir}")
        print(f"训练轮数: {args.num_train_epochs}")
        print(f"批大小: {args.per_device_train_batch_size}")
        print(f"学习率: {args.learning_rate}")
        print(f"训练类型: {'Q-former' if args.use_qformer else 'MLP'} 投影器微调")
        print("-" * 50)


    print("=============================")
    print("Model configuration:")
    print(args.model)
    print("=============================")

    # 构建训练参数
    train_args = TrainArguments(
        # 模型配置
        model=args.model,
        model_type=args.model_type,
        train_type="lora" if args.use_lora else "full",
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
        # 冻结配置
        freeze_llm=False,
        freeze_vit=True,
        # 精度配置
        bf16=True,
        fp16=False,
        # 保存和日志
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        # 分布式训练
        ddp_backend="nccl",
        ddp_find_unused_parameters=True,
        # 其他配置
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name=args.run_name,
        # LoRA配置（如果使用）
        lora_rank=args.lora_rank if args.use_lora else None,
        lora_alpha=args.lora_alpha if args.use_lora else None,
        lora_dropout=args.lora_dropout if args.use_lora else None,
    )
    
    # 如果使用Q-former，设置特定的训练目标
    if args.use_qformer:
        # 对于Q-former，我们主要训练投影器部分
        train_args.freeze_aligner = False
    else:
        # 对于MLP投影器
        train_args.freeze_aligner = False

    result = sft_main(train_args)

    # 只在主进程打印结果
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        projector_type = "Q-former" if args.use_qformer else "MLP"
        print(f"{projector_type} 投影器微调完成!")

        # 处理训练结果
        if hasattr(result, 'log_history'):
            print(f"最佳模型保存在: {args.output_dir}")
            if result.log_history:
                last_log = result.log_history[-1]
                if "train_loss" in last_log:
                    print(f"最终训练损失: {last_log['train_loss']:.4f}")
                if "eval_loss" in last_log:
                    print(f"最终验证损失: {last_log['eval_loss']:.4f}")
        else:
            print(f"训练完成，模型保存在: {args.output_dir}")

    return result


if __name__ == "__main__":
    result = main()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("训练流程结束")
