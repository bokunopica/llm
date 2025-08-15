import time
from logger import setup_logging
from pipeline import TrainPipeline
import os
import sys







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


def main_0815_00():
    # llava-med推理结果再次推理
    setup_logging(log_dir="logs/llava-med-eval")
    llava_med_eval_pipeline = TrainPipeline(
        base_model="/home/qianq/model/llava-med-v1.5-mistral-7b",
        model_type="llava1_5_hf",
        dataset_prefix="/home/qianq/data/image-text-to-text/lidc-clf-nodule-img",
        dataset_name="lidc",
        cuda_devices="0,1",
    )
    llava_med_eval_pipeline.run_eval('/home/qianq/model/llava-med-v1.5-mistral-7b/inference_lidc-clf-nodule-img_lidc.jsonl')
    llava_med_eval_pipeline.run_eval('/home/qianq/model/llava-med-v1.5-mistral-7b/inference_lidc-clf-nodule-img_attr-lidc.jsonl')
    llava_med_eval_pipeline.run_eval('/home/qianq/model/llava-med-v1.5-mistral-7b/inference_lidc-clf-nodule-ct-slice_lidc.jsonl')
    llava_med_eval_pipeline.run_eval('/home/qianq/model/llava-med-v1.5-mistral-7b/inference_lidc-clf-nodule-ct-slice_attr-lidc.jsonl')

    

if __name__ == "__main__":
    # main_0813_night()
    # main_0814()
    main_0815_00()
