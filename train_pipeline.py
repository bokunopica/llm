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


@with_logging("logs/llava-med-sft-1epoch-adalora")
def llava_med_sft_one_epoch_adalora_01():
    global CUDA_VISIBLE_DEVICES
    base_model = "/home/qianq/model/llava-med-v1.5-mistral-7b"
    model_type = "llava1_5_hf"
    pipelines = [
        TrainPipeline(
            base_model=base_model,
            model_type=model_type,
            epoch=1,
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc",
            cuda_devices=CUDA_VISIBLE_DEVICES,
            train_type="adalora",
        ),
        # TrainPipeline(
        #     base_model=base_model,
        #     model_type=model_type,
        #     epoch=1,
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        #     dataset_name="attr-lidc-detail",
        #     cuda_devices=CUDA_VISIBLE_DEVICES,
        #     train_type="adalora",
        # )
    ]
    run_pipelines(pipelines)


@with_logging("logs/llava-med-sft-1epoch-adalora")
def llava_med_sft_one_epoch_adalora_02():
    global CUDA_VISIBLE_DEVICES
    base_model = "/home/qianq/model/llava-med-v1.5-mistral-7b"
    model_type = "llava1_5_hf"
    pipelines = [
        # TrainPipeline(
        #     base_model=base_model,
        #     model_type=model_type,
        #     epoch=1,
        #     dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
        #     dataset_name="attr-lidc",
        #     cuda_devices=CUDA_VISIBLE_DEVICES,
        #     train_type="adalora",
        # ),
        TrainPipeline(
            base_model=base_model,
            model_type=model_type,
            epoch=1,
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc-detail",
            cuda_devices=CUDA_VISIBLE_DEVICES,
            train_type="adalora",
        )
    ]
    run_pipelines(pipelines)


@with_logging("logs/QoQ-Med-VL-7B-sft-1epoch-adalora")
def qoq_med_vl_7b_sft_one_epoch_adalora():
    global CUDA_VISIBLE_DEVICES
    base_model = "/home/qianq/model/QoQ-Med-VL-7B"
    model_type = "qwen2_5_vl"
    pipelines = [
        TrainPipeline(
            base_model=base_model,
            model_type=model_type,
            epoch=1,
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc",
            cuda_devices=CUDA_VISIBLE_DEVICES,
            train_type="adalora",
        ),
        TrainPipeline(
            base_model=base_model,
            model_type=model_type,
            epoch=1,
            dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-ct-slice",
            dataset_name="attr-lidc-detail",
            cuda_devices=CUDA_VISIBLE_DEVICES,
            train_type="adalora",
        )
    ]
    run_pipelines(pipelines)


if __name__ == "__main__":
    import os
    open("train_pipeline.pid", "w").write(str(os.getpid()))
    CUDA_VISIBLE_DEVICES = "1,3"
    qoq_med_vl_7b_sft_one_epoch_adalora()