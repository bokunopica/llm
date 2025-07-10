import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'  # 设置使用多个GPU
from swift.llm import sft_main, TrainArguments

def main():
    # 设置输出目录
    output_dir = "./results/llava-1.5-7b-hf-swift-lora"
    os.makedirs(output_dir, exist_ok=True)
    
    result = sft_main(
        TrainArguments(
            # 模型配置
            model="/home/qianq/model/llava-1.5-7b-hf",
            train_type="ALL",  # 使用sft_type而不是train_type
            
            # 数据集配置
            dataset=[
                "/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
            ],
            
            # 训练配置
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            
            # LoRA 配置 - 手动指定目标模块
            lora_rank=64,
            lora_alpha=128,
            lora_dropout=0.1,
            # lora_modules="ALL",  # 让Swift自动选择所有可训练模块
            freeze_llm=True,
            freeze_vit=True,
            freeze_aligner=False,
            # 或者明确指定目标模块
            # lora_target_modules=["language_model.model.layers.*.self_attn.q_proj", 
            #                     "language_model.model.layers.*.self_attn.k_proj",
            #                     "language_model.model.layers.*.self_attn.v_proj",
            #                     "language_model.model.layers.*.self_attn.o_proj"],
            
            # 精度和优化
            torch_dtype="bfloat16",
            fp16=False,
            bf16=True,
            dataloader_num_workers=0,  # 修复: 设置为0避免多进程问题
            
            # 保存和日志
            output_dir=output_dir,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            eval_strategy="steps",
            
            # 其他配置
            seed=42,
            remove_unused_columns=False,
            report_to=["tensorboard"],
            run_name="llava-1.5-7b-lidc-lora",
            
            # 添加这些参数来避免序列化问题
            dataloader_pin_memory=False,  # 禁用pin_memory
        )
    )
    
    print("训练完成!")
    print(f"最佳模型保存在: {result.checkpoint_dir}")

    # 修复：安全地访问结果
    if isinstance(result, dict):
        # 如果 result 是字典，尝试获取输出目录
        checkpoint_dir = result.get('output_dir', output_dir)
        print(f"最佳模型保存在: {checkpoint_dir}")
        
        # 打印训练结果信息
        if 'train_runtime' in result:
            print(f"训练用时: {result['train_runtime']:.2f} 秒")
        if 'train_loss' in result:
            print(f"训练损失: {result['train_loss']:.4f}")
        if 'eval_loss' in result:
            print(f"验证损失: {result['eval_loss']:.4f}")
            
        # 打印所有可用的键以便调试
        print("训练结果包含的信息:")
        for key, value in result.items():
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
                
    elif hasattr(result, 'checkpoint_dir'):
        # 如果 result 有 checkpoint_dir 属性
        print(f"最佳模型保存在: {result.checkpoint_dir}")
    else:
        # 回退选项
        print(f"最佳模型保存在: {output_dir}")
        print(f"结果对象类型: {type(result)}")
        if hasattr(result, '__dict__'):
            print(f"结果对象属性: {list(result.__dict__.keys())}")
    return result

if __name__ == "__main__":
    result = main()
    print(result)