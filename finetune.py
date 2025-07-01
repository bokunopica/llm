import os
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from data import LIDCClassificationDataset, TrainLlavaModelCollator

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="微调图像文本到文本模型")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="要微调的预训练模型路径或名称",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据集目录，应包含train.jsonl和test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="输出目录，用于保存模型和训练记录",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="训练的总轮次",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="每个设备的训练批次大小",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="每个设备的评估批次大小",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="初始学习率")
    parser.add_argument("--warmup_steps", type=int, default=0, help="学习率预热的步数")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="日志记录步数间隔"
    )
    parser.add_argument("--eval_steps", type=int, default=500, help="评估间隔步数")
    parser.add_argument("--save_steps", type=int, default=500, help="模型保存间隔步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful medical assistant.",
        help="系统提示语",
    )
    parser.add_argument(
        "--use_tensorboard", action="store_true", help="是否使用tensorboard进行实验跟踪"
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default=None,
        help="Tensorboard日志目录，若为None则使用output_dir/runs",
    )
    parser.add_argument(
        "--lora", action="store_true", help="是否使用LoRA进行参数高效微调"
    )
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果没有指定tensorboard目录，则设置默认目录
    if args.use_tensorboard and args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.output_dir, "runs")

    return args


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载处理器和模型
    logger.info(f"加载处理器和模型: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # 准备用于LoRA微调的配置
    if args.lora:
        from peft import (
            LoraConfig,
            get_peft_model,
            TaskType,
            prepare_model_for_kbit_training,
        )

        lora_config = LoraConfig(
            r=16,  # LoRA的秩
            lora_alpha=32,  # LoRA的alpha参数
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],  # 需要微调的模块名称
            lora_dropout=0.05,  # LoRA的dropout率
            bias="none",  # 是否对偏置进行微调
            task_type=TaskType.CAUSAL_LM,  # 任务类型
        )

        # 加载模型，并应用LoRA配置
        logger.info("加载基础模型并应用LoRA配置")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )
        print(model)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数占比
    else:
        # 常规加载模型
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )

    # 加载数据集
    logger.info(f"加载数据集: {args.dataset_dir}")
    train_dataset = LIDCClassificationDataset(args.dataset_dir, is_train=True)
    eval_dataset = LIDCClassificationDataset(args.dataset_dir, is_train=False)

    logger.info(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    # 数据整理器
    data_collator = TrainLlavaModelCollator(
        processor=processor, system_prompt=args.system_prompt
    )

    # 创建tensorboard日志目录
    if args.use_tensorboard:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard日志将保存到 {args.tensorboard_dir}")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to="tensorboard" if args.use_tensorboard else "none",
        logging_dir=args.tensorboard_dir if args.use_tensorboard else None,
        push_to_hub=False,
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存最终模型
    logger.info(f"保存模型到 {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()