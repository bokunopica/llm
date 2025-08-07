import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from model import LlavaQformerForConditionalGeneration

# 导入现有的数据处理类
from data import LIDCClassificationDataset, TrainLlavaModelCollator

logger = logging.get_logger(__name__)

class ProjectorTrainer(Trainer):
    """自定义训练器，只训练投影器部分"""
    
    def __init__(self, freeze_llm=True, freeze_vit=True, **kwargs):
        super().__init__(**kwargs)
        self.freeze_llm = freeze_llm
        self.freeze_vit = freeze_vit
        
    def create_optimizer(self):
        """创建优化器，只优化投影器参数"""
        if self.optimizer is None:
            trainable_params = []
            frozen_params = []
            
            for name, param in self.model.named_parameters():
                # 根据参数名判断是否训练
                if self.freeze_llm and ('language_model' in name or 'lm_head' in name or 'embed_tokens' in name):
                    param.requires_grad = False
                    frozen_params.append(name)
                elif self.freeze_vit and ('vision_tower' in name or 'vision_model' in name or 'vision_encoder' in name):
                    param.requires_grad = False
                    frozen_params.append(name)
                elif any(keyword in name for keyword in ['multi_modal_projector', 'mm_projector', 'projector', 'qformer']):
                    param.requires_grad = True
                    trainable_params.append(param)
                    print(f"训练参数: {name}")
                else:
                    # 默认训练其他参数（如Q-former相关）
                    param.requires_grad = True
                    trainable_params.append(param)
                    print(f"训练参数: {name}")
            
            print(f"冻结参数示例: {frozen_params[:5]}...")
            print(f"可训练参数数量: {len(trainable_params)}")
            print(f"总参数数量: {sum(p.numel() for p in trainable_params):,}")
            
            if len(trainable_params) == 0:
                raise ValueError("没有找到可训练的参数！请检查参数冻结逻辑。")
            
            # 创建优化器
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(trainable_params, **optimizer_kwargs)
        
        return self.optimizer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA model projector with Transformers")
    
    # 环境配置
    parser.add_argument("--cuda_visible_devices", type=str, default="1,3", help="CUDA visible devices")
    
    # 模型和数据配置
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # Q-former配置
    parser.add_argument("--use_qformer", type=str, default="true", help="Use Q-former")
    # parser.add_argument("--num_query_tokens", type=int, default=32, help="Number of query tokens")
    # parser.add_argument("--qformer_hidden_size", type=int, default=768, help="Q-former hidden size")
    # parser.add_argument("--qformer_num_layers", type=int, default=6, help="Q-former layers")
    # parser.add_argument("--qformer_num_heads", type=int, default=12, help="Q-former attention heads")
    
    # 训练配置
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Eval batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    
    # LoRA配置
    parser.add_argument("--use_lora", type=str, default="false", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    
    # 保存和日志
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Eval steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Save limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_name", type=str, default="llava-projector", help="Run name")
    
    return parser.parse_args()

def str_to_bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def setup_qformer_config(model_path, args):
#     """设置Q-former配置"""
#     config_path = os.path.join(model_path, "config.json")
#     if os.path.exists(config_path):
#         with open(config_path, "r") as f:
#             config = json.load(f)
        
#         # 添加Q-former配置
#         qformer_config = {
#             "use_qformer": str_to_bool(args.use_qformer),
#             "num_query_tokens": args.num_query_tokens,
#             "hidden_size": args.qformer_hidden_size,
#             "num_hidden_layers": args.qformer_num_layers,
#             "num_attention_heads": args.qformer_num_heads,
#             "intermediate_size": args.qformer_hidden_size * 4,
#             "hidden_dropout_prob": 0.1,
#             "attention_probs_dropout_prob": 0.1,
#             "initializer_range": 0.02,
#             "layer_norm_eps": 1e-12,
#         }
        
#         config["qformer_config"] = qformer_config
        
#         # 更新projector配置
#         if "projector_config" in config:
#             config["projector_config"]["projector_type"] = "qformer"
#         else:
#             config["projector_config"] = {"projector_type": "qformer"}
        
#         # 保存配置
#         with open(config_path, "w") as f:
#             json.dump(config, f, indent=2)
        
#         print(f"已更新Q-former配置:")
#         print(f"  - 查询token数量: {args.num_query_tokens}")
#         print(f"  - 隐藏层大小: {args.qformer_hidden_size}")
#         print(f"  - Transformer层数: {args.qformer_num_layers}")

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # 初始化分布式训练
    local_rank = -1
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置Q-former配置（只在主进程）
    if local_rank <= 0:
        # if str_to_bool(args.use_qformer):
        #     setup_qformer_config(args.model, args)
        
        print(f"模型路径: {args.model}")
        print(f"数据集路径: {args.dataset}")
        print(f"输出目录: {args.output_dir}")
        print(f"训练类型: {'Q-former' if str_to_bool(args.use_qformer) else 'MLP'} 投影器微调")
        print("-" * 50)
    
    # 加载模型和tokenizer
    print("加载模型和tokenizer...")
    if str_to_bool(args.use_qformer):
        model = LlavaQformerForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if local_rank == -1 else None,
        )
    else:  
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if local_rank == -1 else None,
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        trust_remote_code=True,
    )
    
    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载processor
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        print("成功加载processor")
    except Exception as e:
        print(f"加载processor失败: {e}")
        raise
    
    # 创建训练数据集
    print("创建训练数据集...")
    train_dataset = LIDCClassificationDataset(
        dataset_dir=args.dataset,
        is_train=True
    )
    
    # 创建验证数据集
    print("创建验证数据集...")
    try:
        eval_dataset = LIDCClassificationDataset(
            dataset_dir=args.dataset,
            is_train=False
        )
        print(f"验证数据集大小: {len(eval_dataset)}")
    except:
        print("未找到验证数据集，将使用训练数据集的一部分作为验证集")
        # 简单分割训练数据集
        train_size = int(0.9 * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(eval_dataset)}")
    
    # 创建数据整理器 - 使用现有的实现
    IGNORE_INDEX = -100
    data_collator = TrainLlavaModelCollator(
        processor=processor, 
        IGNORE_INDEX=IGNORE_INDEX,
        # enable_augmentation=str_to_bool(args.enable_augmentation)
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=["tensorboard"],
        run_name=args.run_name,
        ddp_find_unused_parameters=True,
        local_rank=local_rank,
        gradient_checkpointing=True,  # 节省内存
        dataloader_pin_memory=True,
    )
    
    # 创建训练器
    trainer = ProjectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        freeze_llm=True,
        freeze_vit=True,
    )
    
    # 检查是否有checkpoint可以恢复
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None and local_rank <= 0:
            print(f"发现checkpoint: {last_checkpoint}")
    
    # 打印模型信息
    if local_rank <= 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")
        print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")
    
    # 开始训练
    print("开始训练...")
    try:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # 保存模型
        trainer.save_model()
        trainer.save_state()
        
        # 保存训练指标
        if local_rank <= 0:
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            print("训练完成!")
            print(f"模型保存在: {args.output_dir}")
            print(f"最终训练损失: {metrics.get('train_loss', 'N/A')}")
            
            # 如果有验证集，进行最终评估
            if eval_dataset:
                print("进行最终评估...")
                eval_result = trainer.evaluate()
                print(f"最终验证损失: {eval_result.get('eval_loss', 'N/A')}")
                trainer.log_metrics("eval", eval_result)
                trainer.save_metrics("eval", eval_result)
    
    except Exception as e:
        print(f"训练过程中出错: {e}")
        # 保存当前状态
        if hasattr(trainer, 'save_state'):
            trainer.save_state()
        raise

if __name__ == "__main__":
    result = main()
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("训练流程结束")
