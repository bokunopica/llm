#!/usr/bin/env python3
import subprocess
import os
import time


class TrainPipeline:
    """训练管道类"""

    def __init__(
        self,
        base_model,
        epoch=1,
        lr="1e-4",
        dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        dataset_name="attr-lidc",  # attr-lidc | lidc
        cuda_devices="1,3",
    ):
        # ===== 配置区域 =====
        self.max_pixels = 1003520
        self.model = base_model
        self.epoch = epoch
        self.lr = lr
        self.dataset_prefix = dataset_prefix
        self.dataset_name = dataset_name
        self.cuda_devices = cuda_devices

        # 推理相关
        self.infer_backend = "vllm"
        self.max_batch_size = 16
        self.max_new_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.write_batch_size = 64
        self.tensor_parallel_size = 1
        self.metric = "acc"

    def run_train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        # 用 dict 管理参数
        train_params = {
            "--model": self.model,
            "--dataset": f"{self.dataset_prefix}/{self.dataset_name}",
            "--split_dataset_ratio": "0.01",
            "--train_type": "lora",
            "--torch_dtype": "bfloat16",
            "--num_train_epochs": str(self.epoch),
            "--per_device_train_batch_size": "4",
            "--per_device_eval_batch_size": "4",
            "--learning_rate": self.lr,
            "--lora_rank": "8",
            "--lora_alpha": "32",
            "--target_modules": "all-linear",
            "--freeze_vit": "true",
            "--gradient_accumulation_steps": "16",
            "--eval_steps": "100",
            "--save_steps": "100",
            "--save_total_limit": "2",
            "--logging_steps": "5",
            "--max_length": "2048",
            "--warmup_ratio": "0.05",
            "--dataloader_num_workers": "0",
            "--model_type": "llava1_5_hf",
            "--output_dir": f"results/{os.path.basename(self.model)}-EPOCH={self.epoch}-LR={self.lr}-DATASET={self.dataset_name}",
        }
        # 展开成列表
        train_cmd = ["swift", "sft"]
        for k, v in train_params.items():
            train_cmd += [k, v]

        print("=== 开始训练 ===")
        subprocess.run(train_cmd, check=True)
        return train_params["output_dir"]

    def run_infer(self, output_dir):
        """
        args:
            output_dir: 训练输出目录-
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        dataset_path = f"{self.dataset_prefix}/{self.dataset_name}"
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 找最新的 checkpoint
        checkpoints = sorted(
            [
                os.path.join(output_dir, d)
                for d in os.listdir(output_dir)
                if "checkpoint" in d
            ],
            key=os.path.getmtime,
        )
        if not checkpoints:
            raise RuntimeError("未找到 checkpoint，请检查训练是否完成")
        latest_ckpt = checkpoints[-1]

        result_path = f"{latest_ckpt}/inference_{self.dataset_name}.jsonl"

        print("=== 开始推理 ===")
        infer_params = {
            "--cuda_visible_devices": self.cuda_devices,
            "--infer_backend": self.infer_backend,
            "--model": latest_ckpt,
            "--dataset": dataset_path,
            "--max_batch_size": str(self.max_batch_size),
            "--max_new_tokens": str(self.max_new_tokens),
            "--temperature": str(self.temperature),
            "--top_p": str(self.top_p),
            "--write_batch_size": str(self.write_batch_size),
            "--tensor_parallel_size": str(self.tensor_parallel_size),
            "--result_path": result_path,
            "--metric": self.metric,
        }

        infer_cmd = ["python", "inference.py"]
        for k, v in infer_params.items():
            infer_cmd += [k, v]

        subprocess.run(infer_cmd, check=True)

    def run_eval(self, result_path):
        """
        args:
            result_path: 推理结果文件路径
        """
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f"推理结果文件不存在: {result_path}")

        print("=== 开始评估 ===")
        eval_cmd = ["python", "evaluate_results.py", "--input_file", result_path]
        subprocess.run(eval_cmd, check=True)



# ===== 主流程 =====
if __name__ == "__main__":
    start_time = time.time()
    pipeline = TrainPipeline(
        base_model="llava-1.5-hf/llava-1.5-7b",
        epoch=1,
        lr="1e-4",
        dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        dataset_name="attr-lidc",
        cuda_devices="0,1,2,3",
    )
    pipeline.run_infer("/home/qianq/mycodes/llm/results/llava-med-v1.5-mistral-7b-EPOCH=1-LR=-DATASET=lidc")

    print(f"=== Pipeline 完成，总耗时: {time.time() - start_time:.1f} 秒 ===")
