import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDEO_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "12"

from swift.llm import InferArguments, infer_main, ModelType
import json
from pathlib import Path


def inference_on_dataset():
    """在数据集上进行推理"""
    # 创建推理参数
    model_path = "/home/qianq/mycodes/llm/results/llava-1.5-7b-hf-swift-lora/v5-20250709-194154/checkpoint-500"
    infer_args = InferArguments(
        model=model_path,
        infer_backend="pt",
        max_batch_size=4,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        # 数据集配置
        # dataset=["/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"],
        val_dataset=[
            "/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN/test.jsonl",
        ],
        # model_type=ModelType.llava1_5_hf,
        result_path=f"{model_path}/inference_results.jsonl",
        # 数据集采样
        dataset_shuffle=False,
        # 批处理配置
        write_batch_size=256,
        # 评估指标
        # metric='acc',  # 或者 'acc'
    )
    # 使用SwiftInfer进行推理
    infer_main(infer_args)


if __name__ == "__main__":
    # 直接运行批量推理示例
    print("开始推理...")
    results = inference_on_dataset()
    print(results)

    # 如果要运行其他类型的推理，可以调用main()
    # main()
