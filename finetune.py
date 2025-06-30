import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)

# from custom_trainer import WebTrainer
from data import LlavaDataset, TrainLlavaModelCollator, LIDCClassificationDataset

# from train_llava.data_websend import DatasetReceiveByWeb, TrainLlavaModelCollatorByWeb
from util import print_trainable_parameters

logger = logging.getLogger(__name__)

# If using LlavaInsight
from modeling_llava_insight import LlavaInsight, LlavaInsightProcessor


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="test_model/model001")
    model_class: Optional[str] = field(default="LlavaForConditionalGeneration", metadata={
        "help": "Choose model class. Options: LlavaForConditionalGeneration, LlavaInsight."
    })
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            4. train_projector:只训练投影层
            """
        },
    )
    # use_patches: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Whether to use patches in the model. If True, num_patches must be specified."
    #     },
    # )
    # num_patches: int = field(
    #     default=0,
    #     metadata={
    #         "help": "Number of patches to use in the model. Required if use_patches is True."
    #     },
    # )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)


def load_model_processor(modelargs: ModelArguments):
    init_kwargs = {}
    if modelargs.model_class == "LlavaInsight":
        # Use LlavaInsight model and processor
        model_cls = LlavaInsight
        processor_cls = LlavaInsightProcessor
        init_kwargs['use_patches'] = modelargs.use_patches
        init_kwargs['num_patches'] = modelargs.num_patches
    elif modelargs.model_class == "LlavaNextForConditionalGeneration":
        # Use LlavaNext model and processor
        model_cls = LlavaNextForConditionalGeneration
        processor_cls = LlavaNextProcessor
        # init_kwargs['use_patches'] = modelargs.use_patches
        # init_kwargs['num_patches'] = modelargs.num_patches
    else:
        # Default to LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
        processor_cls = LlavaProcessor

    # Load the model with the selected class
    model = model_cls.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **init_kwargs,
    )

    # Load the processor
    processor = processor_cls.from_pretrained(
        modelargs.model_name_or_path,
        patch_size=model.vision_tower.config.patch_size,
        **init_kwargs,
    )

    processor.patch_size = getattr(model.vision_tower.config, "patch_size", 14)
    processor.vision_feature_select_strategy = getattr(model.config, "vision_feature_select_strategy", "default")


    # Apply training strategy (LoRA, freeze vision, etc.)
    if modelargs.train_type == "use_lora":
        logging.warning("Loading model with LoRA")

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)

    elif modelargs.train_type == "none":
        logging.warning("Training all parameters")

    elif modelargs.train_type == "freeze_vision":
        logging.warning("Freezing vision_tower parameters")
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    elif modelargs.train_type == "train_projector":
        logging.warning("Freezing all but the projector parameters")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = False

    print_trainable_parameters(model)
    # processor.patch_size = model.vision_tower.config.patch_size
    # processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy


    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):
    llava_dataset = LIDCClassificationDataset(
        dataargs.data_path  # Example: "data/LIDC_train_data"
    )
    data_collator = TrainLlavaModelCollator(
        processor,
        -100,
        'You are a medical image assistant specializing in lung CT scans. Based on the image, determine whether the lung nodule is malignant or benign. Respond with only one word: \"malignant\" or \"benign\". Do not explain your answer.'
    )
    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
