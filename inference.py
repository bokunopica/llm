import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Swift")
    
    # 设备配置
    parser.add_argument("--cuda_visible_devices", type=str, default="1,3", 
                       help="CUDA visible devices (default: 1,3)")
    parser.add_argument("--infer_backend", type=str, default="pt",
                       choices=["pt", "vllm", "lmdeploy", "sglang"],
                       help="Inference backend")
    
    # 模型配置
    parser.add_argument("--model", type=str,
                       default="/home/qianq/mycodes/llm/results/llava-1.5-7b-hf-swift-lora/v5-20250709-194154/checkpoint-500",
                       help="Model path")
    parser.add_argument("--dataset", type=str,
                       default="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN",
                       help="Dataset path")
    
    # 推理配置
    parser.add_argument("--max_batch_size", type=int, default=8,
                       help="Max batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top p")
    
    # 数据集配置
    parser.add_argument("--val_dataset_sample", type=int, default=-1,
                       help="Number of samples for validation, -1 means all")
    parser.add_argument("--write_batch_size", type=int, default=32,
                       help="Write batch size")
    
    # 分布式配置
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallel size (for vllm)")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                       help="Pipeline parallel size (for vllm)")
    
    # 输出配置
    parser.add_argument("--result_path", type=str, default=None,
                       help="Result path")
    parser.add_argument("--metric", type=str, default=None,
                       choices=["acc", "rouge"],
                       help="Evaluation metric")
    
    return parser.parse_args()

def setup_environment(cuda_visible_devices):
    """设置环境变量"""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["MAX_PIXELS"] = "1003520"
    os.environ["VIDEO_MAX_PIXELS"] = "50176"
    os.environ["FPS_MAX_FRAMES"] = "12"

def prepare_dataset_config(dataset_path, val_dataset_sample):
    """准备数据集配置，避免重复代码"""
    # 确定数据集路径
    if dataset_path.endswith('.jsonl'):
        val_dataset_path = dataset_path
    else:
        val_dataset_path = dataset_path
    
    # 确定采样数量
    if val_dataset_sample == -1:
        # 如果设置为-1，则使用所有样本
        try:
            with open(f"{dataset_path}/test.jsonl", 'r') as f:
                dataset_sample = len(f.readlines())
        except:
            dataset_sample = 100  # 默认值
    else:
        dataset_sample = val_dataset_sample
    
    return val_dataset_path, dataset_sample

def inference_multi_gpu_pt(args):
    """使用PT backend进行多GPU推理"""
    from swift.llm import InferArguments, infer_main
    
    # 计算GPU数量
    gpu_count = len(args.cuda_visible_devices.split(','))
    val_dataset_path, dataset_sample = prepare_dataset_config(args.dataset, args.val_dataset_sample)
    
    infer_args = InferArguments(
        model=args.model,
        infer_backend="pt",
        
        # 批处理配置 - 根据GPU数量调整
        max_batch_size=args.max_batch_size * gpu_count,
        
        # 生成配置
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        
        # 数据集配置
        val_dataset=[val_dataset_path],
        val_dataset_sample=dataset_sample,
        dataset_shuffle=False,
        
        # 输出配置
        result_path=args.result_path or f"{args.model}/inference_results.jsonl",
        write_batch_size=args.write_batch_size,
        
        # 评估配置
        metric=args.metric,
        
        # 分布式配置
        ddp_backend="nccl",
    )
    
    print(f"使用PT backend在{gpu_count}个GPU上进行推理")
    print(f"调整后的批大小: {infer_args.max_batch_size}")
    
    return infer_main(infer_args)

def inference_multi_gpu_vllm(args):
    """使用VLLM backend进行多GPU推理"""
    from swift.llm import InferArguments, infer_main
    
    # 计算GPU数量
    gpu_count = len(args.cuda_visible_devices.split(','))
    tensor_parallel_size = min(args.tensor_parallel_size, gpu_count)
    val_dataset_path, dataset_sample = prepare_dataset_config(args.dataset, args.val_dataset_sample)

    
    infer_args = InferArguments(
        model=args.model,
        infer_backend="vllm",
        
        # VLLM特定配置
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        
        # 批处理配置
        max_batch_size=args.max_batch_size,
        
        # 生成配置
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        
        # 数据集配置
        val_dataset=[val_dataset_path],
        val_dataset_sample=dataset_sample,
        dataset_shuffle=False,
        
        # 输出配置
        result_path=args.result_path or f"{args.model}/inference_results_vllm.jsonl",
        write_batch_size=args.write_batch_size,
        
        # 评估配置
        metric=args.metric,
        
        # VLLM优化配置
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )
    
    print(f"使用VLLM backend在{gpu_count}个GPU上进行推理")
    print(f"张量并行大小: {tensor_parallel_size}")
    
    return infer_main(infer_args)

def inference_on_dataset(args):
    """在数据集上进行推理"""
    
    if args.infer_backend == "pt":
        return inference_multi_gpu_pt(args)
    elif args.infer_backend == "vllm":
        return inference_multi_gpu_vllm(args)
    else:
        # 其他backend的通用配置
        from swift.llm import InferArguments, infer_main

        val_dataset_path, dataset_sample = prepare_dataset_config(args.dataset, args.val_dataset_sample)

        
        infer_args = InferArguments(
            model=args.model,
            infer_backend=args.infer_backend,
            max_batch_size=args.max_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            val_dataset=[val_dataset_path],
            val_dataset_sample=dataset_sample,
            dataset_shuffle=False,
            result_path=args.result_path or f"{args.model}/inference_results_{args.infer_backend}.jsonl",
            write_batch_size=args.write_batch_size,
            metric=args.metric,
        )
        
        return infer_main(infer_args)

def main():
    args = parse_args()
    
    # 设置环境变量
    setup_environment(args.cuda_visible_devices)
    
    # 打印配置信息
    print("=== 推理配置 ===")
    print(f"CUDA设备: {args.cuda_visible_devices}")
    print(f"推理后端: {args.infer_backend}")
    print(f"模型路径: {args.model}")
    print(f"数据集路径: {args.dataset}")
    print(f"批大小: {args.max_batch_size}")
    print(f"最大新token数: {args.max_new_tokens}")
    print(f"温度: {args.temperature}")
    print(f"采样数量: {args.val_dataset_sample}")
    if args.infer_backend == "vllm":
        print(f"张量并行大小: {args.tensor_parallel_size}")
    print("=" * 20)
    
    # 开始推理
    print("开始推理...")
    results = inference_on_dataset(args)
    
    print("推理完成!")
    return results

if __name__ == "__main__":
    main()
    # print(f"推理结果: {results}")
