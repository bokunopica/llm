from logger import (
    setup_logging,
    with_logging,
)
from pipeline import (
    TrainPipeline,
    run_pipelines,
)
from typing import List
from utils import (
    wait_until,
    construct_train_time,
)


@with_logging("logs/llava-med-sft-1epoch-bone")
def llava_med_sft_one_epoch_detail_enforce_bugfix():
    global CUDA_VISIBLE_DEVICES
    base_model = "/home/qianq/model/llava-med-v1.5-mistral-7b"
    model_type = "llava1_5_hf"
    pipeline = TrainPipeline(
        base_model=base_model,
        model_type=model_type,
        epoch=1,
        dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        dataset_name="lidc-detail",
        cuda_devices=CUDA_VISIBLE_DEVICES,
        train_type="bone",
    )
    pipeline.run()


if __name__ == "__main__":
    # main_0813_night()
    # main_0814()logs
    # main_0815_01()
    # main_0815_03()
    # main_0815_04()
    # 在pipeline.txt文件中写入进程id
    import os

    open("train_pipeline.pid", "w").write(str(os.getpid()))
    # wait_until(construct_train_time(hour=5), check_interval=600)  # 每10分钟检查一次
    CUDA_VISIBLE_DEVICES = "1"
    # qoq_sft_one_epoch()
    # llava_med_sft_one_epoch_detail_enforce()
    # qoq_sft_one_epoch_detail_enforce()
    llava_med_sft_one_epoch_detail_enforce_bugfix()
