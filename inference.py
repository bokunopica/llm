import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Swift")

    # 设备配置
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="1,3",
        help="CUDA visible devices (default: 1,3)",
    )
    parser.add_argument(
        "--infer_backend",
        type=str,
        default="pt",
        choices=["pt", "vllm", "lmdeploy", "sglang"],
        help="Inference backend",
    )

    # 模型配置
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="",
        help="Model type for inference",
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default="/home/qianq/mycodes/llm/results/llava-1.5-7b-hf-swift-lora/v5-20250709-194154/checkpoint-500",
        help="adapters path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN",
        help="Dataset path",
    )

    # 推理配置
    parser.add_argument("--max_batch_size", type=int, default=8, help="Max batch size")
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max new tokens"
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top p")

    # 数据集配置
    parser.add_argument(
        "--val_dataset_sample",
        type=int,
        default=-1,
        help="Number of samples for validation, -1 means all",
    )
    parser.add_argument(
        "--write_batch_size", type=int, default=1024, help="Write batch size"
    )

    # 分布式配置
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (for vllm)",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel size (for vllm)",
    )

    # 输出配置
    parser.add_argument("--result_path", type=str, default=None, help="Result path")
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=["acc", "rouge"],
        help="Evaluation metric",
    )

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
    if dataset_path.endswith(".jsonl"):
        val_dataset_path = dataset_path
    else:
        val_dataset_path = dataset_path

    # 确定采样数量
    if val_dataset_sample == -1:
        # 如果设置为-1，则使用所有样本
        try:
            with open(f"{dataset_path}/test.jsonl", "r") as f:
                dataset_sample = len(f.readlines())
        except:
            dataset_sample = 100  # 默认值
    else:
        dataset_sample = val_dataset_sample

    return val_dataset_path, dataset_sample


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
    from swift.llm import InferArguments, infer_main

    val_dataset_path, dataset_sample = prepare_dataset_config(
        args.dataset, args.val_dataset_sample
    )

    if args.model == "":
        infer_args = InferArguments(
            adapters=args.adapters,
            infer_backend=args.infer_backend,
            max_batch_size=args.max_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            val_dataset=[val_dataset_path],
            val_dataset_sample=dataset_sample,
            dataset_shuffle=False,
            result_path=args.result_path,
            write_batch_size=args.write_batch_size,
            metric=args.metric,
        )
    else:
        infer_args = InferArguments(
            model=args.model,
            model_type=args.model_type,
            # adapters=args.adapters,
            infer_backend=args.infer_backend,
            max_batch_size=args.max_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            val_dataset=[val_dataset_path],
            val_dataset_sample=dataset_sample,
            dataset_shuffle=False,
            result_path=args.result_path,
            write_batch_size=args.write_batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            metric=args.metric,
        )

    results = infer_main(infer_args)

    print("推理完成!")
    return results


if __name__ == "__main__":
    main()
    # print(f"推理结果: {results}")
