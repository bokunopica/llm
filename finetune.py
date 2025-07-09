import os
import argparse
import logging
import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from data import TrainLlavaModelCollator

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
        "--dataset_type",
        type=str,
        required=True,
        help="数据集类型，LIDC-Classification or CC3M",
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
    
    # 训练策略相关参数
    parser.add_argument(
        "--train_type",
        type=str,
        default="lora_with_projector",
        choices=["full", "lora", "lora_with_projector", "projector_only", "freeze_vision"],
        help="训练策略：full(全量), lora(仅LoRA), lora_with_projector(LoRA+投影层), projector_only(仅投影层), freeze_vision(冻结视觉层)",
    )
    parser.add_argument("--fp16", action="store_true", help="是否使用fp16混合精度训练")
    parser.add_argument("--bf16", action="store_true", help="是否使用bf16混合精度训练")
    
    # LoRA 相关参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA的秩")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA的alpha参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA的dropout率")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs='+',
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="LoRA目标模块列表",
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=1, 
        help="保存检查点的数量限制，超过限制会删除旧的检查点"
    )
    parser.add_argument(
        "--load_best_model_at_end", 
        action="store_true", 
        help="训练结束时加载最佳模型"
    )
    parser.add_argument(
        "--metric_for_best_model", 
        type=str, 
        default="eval_loss", 
        help="用于确定最佳模型的指标"
    )
    parser.add_argument(
        "--greater_is_better", 
        action="store_true", 
        help="指标值越大越好（如accuracy），否则越小越好（如loss）"
    )
    parser.add_argument(
        "--deepspeed", 
        type=str, 
        default=None, 
        help="DeepSpeed配置文件路径"
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="本地进程rank，用于分布式训练"
    )

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 如果没有指定tensorboard目录，则设置默认目录
    if args.use_tensorboard and args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.output_dir, "runs")

    return args


def setup_model_training(model, args):
    """根据训练类型配置模型"""
    
    if args.train_type == "full":
        logger.info("使用全量参数进行训练")
        return model
        
    elif args.train_type == "lora":
        logger.info("使用LoRA进行参数高效微调（不包含投影层）")
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        
    elif args.train_type == "lora_with_projector":
        logger.info("使用LoRA + 多模态投影层进行微调")
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=["multi_modal_projector"],  # 同时训练投影层
        )
        model = get_peft_model(model, lora_config)
        
    elif args.train_type == "projector_only":
        logger.info("仅训练多模态投影层，冻结其他参数")
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻投影层
        if hasattr(model, 'multi_modal_projector'):
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = True
        else:
            logger.warning("未找到multi_modal_projector层")
            
    elif args.train_type == "freeze_vision":
        logger.info("冻结视觉编码器，训练语言模型和投影层")
        # 冻结视觉编码器
        if hasattr(model, 'vision_tower'):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
        else:
            logger.warning("未找到vision_tower层")
    
    return model


def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    logger.info(f"可训练参数: {trainable_params:,} || 总参数: {all_param:,} || 可训练比例: {100 * trainable_params / all_param:.4f}%")
    
    # 打印具体哪些层是可训练的
    trainable_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layers.append(name)
    
    if len(trainable_layers) <= 20:  # 如果层数不多，打印所有层
        logger.info("可训练的层:")
        for layer in trainable_layers:
            logger.info(f"  - {layer}")
    else:  # 如果层数很多，只打印前10个和后10个
        logger.info("可训练的层（前10个和后10个）:")
        for layer in trainable_layers[:10]:
            logger.info(f"  - {layer}")
        logger.info("  ...")
        for layer in trainable_layers[-10:]:
            logger.info(f"  - {layer}")

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)

    # 加载处理器和模型
    logger.info(f"加载处理器和模型: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    # 确保处理器有patch_size属性
    if not hasattr(processor, 'patch_size') or processor.patch_size is None:
        processor.patch_size = 14

    # 加载模型
    logger.info("加载基础模型")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 or args.deepspeed else torch.float32,
        low_cpu_mem_usage=True,
    )

    # 根据训练类型配置模型
    model = setup_model_training(model, args)
    
    # 打印可训练参数统计
    print_trainable_parameters(model)

    # 加载数据集
    
    if args.dataset_type == "LIDC-Classification":
        from data import LIDCClassificationDataset
        train_dataset = LIDCClassificationDataset(args.dataset_dir, is_train=True)
        eval_dataset = LIDCClassificationDataset(args.dataset_dir, is_train=False)
        logger.info("使用LIDC-Classification数据集")
    elif args.dataset_type == "CC3M":
        from data import LlavaDatasetWithSplit
        train_dataset = LlavaDatasetWithSplit(args.dataset_dir, is_train=True)
        eval_dataset = LlavaDatasetWithSplit(args.dataset_dir, is_train=False)
        logger.info("使用CC3M数据集")

    logger.info(f"加载数据集: {args.dataset_dir}")



    logger.info(f"训练集大小: {len(train_dataset)}, 评估集大小: {len(eval_dataset)}")

    # 数据整理器
    # data_collator = TrainLlavaModelCollator(
        # processor=processor, system_prompt=args.system_prompt
    # )
    data_collator = TrainLlavaModelCollator(
        processor=processor,
        IGNORE_INDEX=-100,  # 忽略索引
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
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        fp16=args.fp16 if not args.deepspeed else False,  # DeepSpeed配置中处理FP16
        bf16=args.bf16 if not args.deepspeed else False,
        deepspeed=args.deepspeed,  # DeepSpeed配置文件
        report_to="tensorboard" if args.use_tensorboard else "none",
        logging_dir=args.tensorboard_dir if args.use_tensorboard else None,
        push_to_hub=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
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