import os
import sys
import time
import shlex
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class Tee:
    """让 stdout/stderr 同时写到文件和终端"""

    def __init__(self, filename: str, mode: str = "w"):
        # 行缓冲，便于实时落盘
        self.file = open(filename, mode, buffering=1)
        self.stdout = sys.stdout

    def write(self, data: str):
        self.stdout.write(data)  # 打印到终端
        self.file.write(data)    # 写到文件

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ========== 子进程日志实时打印（关键改造） ==========

def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def stream_run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None):
    """
    以流式的方式运行命令，实时输出 stdout/stderr 到当前 stdout（已经被 Tee 接管），
    并在结束时检查返回码。
    """
    # 打印命令
    print(f"\n$ {_format_cmd(cmd)}")
    sys.stdout.flush()

    # 启用无缓冲，避免 Python 子进程日志被吞
    _env = os.environ.copy()
    if env:
        _env.update(env)
    _env.setdefault("PYTHONUNBUFFERED", "1")
    # 一些训练脚本会显示进度条，可按需禁用
    _env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # _env.setdefault("HF_DISABLE_PROGRESS_BARS", "1")  # 如果不想要 tqdm 进度条可打开

    start = time.time()
    # 使用 line-buffered 文本模式，合并 stderr
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        env=_env,
        text=True,
        bufsize=1,
        universal_newlines=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            # 直接写入（Tee 会同步到文件）
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = proc.wait()

    duration = time.time() - start
    print(f"[完成] 退出码={returncode}, 耗时={duration:.1f}s")
    sys.stdout.flush()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)



def setup_logging(log_dir="logs"):
    """初始化双输出日志"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    tee = Tee(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return log_file


class TrainPipeline:
    """训练管道类"""

    def __init__(
        self,
        base_model,
        model_type,
        epoch=1,
        lr="1e-4",
        dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        dataset_name="attr-lidc",  # attr-lidc | lidc
        # 默认全部gpu
        cuda_devices=None,
        is_raw_model=False,  # 是否直接使用原始模型进行推理
    ):
        # ===== 配置区域 =====
        self.max_pixels = 1003520
        self.model = base_model
        self.model_type = model_type
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
        self.is_raw_model = is_raw_model

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.cuda_devices is not None and self.cuda_devices != "":
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        else:
            # 不设置表示使用全部 GPU
            env.pop("CUDA_VISIBLE_DEVICES", None)
        return env

    def run_train(self) -> str:
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env = self._build_env()
        # 用 dict 管理参数
        train_params = {
            "--model": self.model,
            # "--model_type": self.model_type,
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

        # subprocess.run(train_cmd, check=True)
        stream_run(train_cmd, env=env)
        return train_params["--output_dir"]

    def run_infer(self, output_dir, is_raw_model=False):
        """
        args:
            output_dir: 训练输出目录-
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        env = self._build_env()
        dataset_path = f"{self.dataset_prefix}/{self.dataset_name}"
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        # 找最新的 checkpoint
        if is_raw_model:
            version_folders = []
            checkpoints = [self.model]
        else:
            version_folders: list = os.listdir(output_dir)
            if not version_folders:
                raise RuntimeError(f"输出目录不存在任何版本文件: {output_dir}")
            else:
                version_folders.sort(reverse=True)
                base_folder = os.path.join(output_dir, version_folders[0])
                checkpoints = sorted(
                    [
                        os.path.join(base_folder, d)
                        for d in os.listdir(base_folder)
                        if "checkpoint" in d
                    ],
                    key=os.path.getmtime,
                )

        if not checkpoints:
            raise RuntimeError("未找到 checkpoint，请检查训练是否完成")

        latest_ckpt = checkpoints[-1]


        result_path = f"{latest_ckpt}/inference_{self.dataset_prefix.split('/')[-1]}_{self.dataset_name}.jsonl"

        print("=== 开始推理 ===")
        infer_params = {
            "--cuda_visible_devices": self.cuda_devices,
            "--dataset": dataset_path,
            "--result_path": result_path,
            "--infer_backend": "pt",
            "--max_batch_size": "8",
            "--max_new_tokens": "512",
            "--temperature": "0",
            "--top_p": "0.9",
            "--val_dataset_sample": "-1",
            "--write_batch_size": "32",
            "--tensor_parallel_size": "1",
            "--pipeline_parallel_size": "1",
            "--metric": "acc",
        }
        if is_raw_model:
            infer_params.update(
                {
                    "--model": self.model,
                    "--model_type": self.model_type,
                }
            )
        else:
            infer_params.update({"--adapters": latest_ckpt})
        infer_cmd = ["python", "inference.py"]
        for k, v in infer_params.items():
            infer_cmd += [k, v]

        # subprocess.run(infer_cmd, check=True)
        stream_run(infer_cmd, env=env)
        return result_path

    def run_eval(self, result_path):
        """
        args:
            result_path: 推理结果文件路径
        """
        env = self._build_env()
        if not os.path.isfile(result_path):
            raise FileNotFoundError(f"推理结果文件不存在: {result_path}")

        print("=== 开始评估 ===")
        eval_cmd = ["python", "evaluate_results.py", "--input_file", result_path]
        # subprocess.run(eval_cmd, check=True)
        stream_run(eval_cmd, env=env)

    def run(self):
        if self.is_raw_model:
            print("=== 直接使用原始模型进行推理 ===")
            self.run_without_train()
        else:
            output_dir = self.run_train()
            result_path = self.run_infer(output_dir)
            self.run_eval(result_path)

    def run_without_train(self):
        result_path = self.run_infer(self.model, is_raw_model=True)
        self.run_eval(result_path)

    def run_on_schedule(self, next_run_time):
        """
        定时运行训练、推理和评估, 只运行一次
        args:
            next_run_time: 下次运行的时间戳
        """
        while True:
            current_time = time.time()

            print(
                f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}, "
                f"计划运行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_run_time))}"
            )
            if current_time >= next_run_time:
                print(f"=== 定时任务开始，当前时间: {time.ctime(current_time)} ===")
                try:
                    self.run()
                except Exception as e:
                    print(f"❌ 运行出错: {e}")
                print(f"=== 定时任务结束，当前时间: {time.ctime(time.time())} ===")
                break
            time.sleep(300)


def construct_train_time(hour, minute=0, second=0):
    """构造下一个训练时间的时间戳"""
    from datetime import datetime, timedelta

    now = datetime.now()
    next_run_time = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if next_run_time <= now:
        next_run_time += timedelta(days=1)
    return next_run_time.timestamp()


def wait_until(next_run_time):
    while True:
        current_time = time.time()
        print(
            f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))}, "
            f"计划运行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(next_run_time))}"
        )
        if current_time >= next_run_time:
            break
        time.sleep(300)  # 每5分钟检查一次


def main_0813():
    log_file = setup_logging()
    print(f"日志已保存到: {log_file}")

    # wait_until(construct_train_time(hour=7))

    pipelines = [
        #####
        # llava-med
        #####
        # 数据类型：肺结节放大图像
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：肺结节放大图像
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：肺结节放大图像
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="lidc",
            cuda_devices="2,3",
        ),
        # 数据类型：肺结节放大图像
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="lidc",
            cuda_devices="2,3",
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-med-v1.5-mistral-7b
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
        ),
        #####
        # llava-1.5-7b-hf
        #####
        # 数据类型：肺结节放大图像
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：肺结节放大图像
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="LIDC-IDRI-MLLM-CLF-EN-ATTRS",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="LIDC-IDRI-MLLM-CLF-EN-ATTRS",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),


        # # 数据类型：肺结节放大图像
        # # 训练模型：llava-1.5-7b-hf
        # # 结节特征：×
        # TrainPipeline(
        #     base_model="/home/qianq/model/llava-1.5-7b-hf",
        #     model_type="llava1_5_hf",
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
        #     dataset_name="LIDC-IDRI-MLLM-CLF-EN",
        #     cuda_devices="2,3",
        # ),
        # # 数据类型：肺结节放大图像
        # # 训练模型：llava-1.5-7b-hf
        # # 结节特征：√
        # TrainPipeline(
        #     base_model="/home/qianq/model/llava-1.5-7b-hf",
        #     model_type="llava1_5_hf",
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
        #     dataset_name="LIDC-IDRI-MLLM-CLF-EN-ATTRS",
        #     cuda_devices="2,3",
        # ),



        # # 数据类型：CT切片+肺结节目标框
        # # 训练模型：llava-1.5-7b-hf
        # # 结节特征：×
        # TrainPipeline(
        #     base_model="/home/qianq/model/llava-1.5-7b-hf",
        #     model_type="llava1_5_hf",
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        #     dataset_name="LIDC-IDRI-MLLM-CLF-EN",
        #     cuda_devices="2,3",
        # ),
        # # 数据类型：CT切片+肺结节目标框
        # # 训练模型：llava-1.5-7b-hf
        # # 结节特征：√
        # TrainPipeline(
        #     base_model="/home/qianq/model/llava-1.5-7b-hf",
        #     model_type="llava1_5_hf",
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        #     dataset_name="LIDC-IDRI-MLLM-CLF-EN-ATTRS",
        #     cuda_devices="2,3",
        # ),
    ]

    for i, pipeline in enumerate(pipelines, start=1):
        print(f"=== 开始执行 Pipeline {i} ===")
        try:
            pipeline.run()
        except Exception as e:
            print(f"❌ Pipeline {i} 出错: {e}")
        print(f"=== Pipeline {i} 完成 ===")


def main_0813_night():
    log_file = setup_logging()
    print(f"日志已保存到: {log_file}")

    # wait_until(construct_train_time(hour=7))

    pipelines = [
        #####
        # llava-1.5-7b-hf

        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf（原始模型不训练）
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
            is_raw_model=True,  # 直接使用原始模型进行推理
        ),
    ]

    for i, pipeline in enumerate(pipelines, start=1):
        print(f"=== 开始执行 Pipeline {i} ===")
        try:
            pipeline.run()
        except Exception as e:
            print(f"❌ Pipeline {i} 出错: {e}")
        print(f"=== Pipeline {i} 完成 ===")


def main_0814():
    log_file = setup_logging()
    print(f"日志已保存到: {log_file}")

    wait_until(construct_train_time(hour=7))

    pipelines = [
        
        #####
        # llava-1.5-7b-hf
        #####

        # 数据类型：肺结节放大图像
        # 训练模型：llava-1.5-7b-hf
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-1.5-7b-hf",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="lidc",
            cuda_devices="2,3",
        ),
        # 数据类型：肺结节放大图像
        # 训练模型：llava-1.5-7b-hf
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-1.5-7b-hf",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
        ),



        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf
        # 结节特征：×
        TrainPipeline(
            base_model="/home/qianq/model/llava-1.5-7b-hf",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="lidc",
            cuda_devices="2,3",
        ),
        # 数据类型：CT切片+肺结节目标框
        # 训练模型：llava-1.5-7b-hf
        # 结节特征：√
        TrainPipeline(
            base_model="/home/qianq/model/llava-1.5-7b-hf",
            model_type="llava1_5_hf",
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc",
            cuda_devices="2,3",
        ),
    ]

    for i, pipeline in enumerate(pipelines, start=1):
        print(f"=== 开始执行 Pipeline {i} ===")
        try:
            pipeline.run()
        except Exception as e:
            print(f"❌ Pipeline {i} 出错: {e}")
        print(f"=== Pipeline {i} 完成 ===")


if __name__ == "__main__":
    main_0813_night()
    main_0814()
