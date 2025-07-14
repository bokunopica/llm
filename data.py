# Standard library imports
import os
import pwd
from dataclasses import dataclass
from pathlib import Path
import random

# Third-party imports
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import trange
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
from typing import List, Tuple, Dict
import torchvision.transforms as transforms


@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()

        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images_dl")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data.get("conversations")

        human_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))
        return human_input, chatbot_output, image_path


class LIDCClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        is_train: bool = True,
    ) -> None:
        super().__init__()
        self.is_train = is_train
        self.chat_data = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> List[Dict]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("train.jsonl" if self.is_train else "test.jsonl")
        print(chat_file)
        chat_data = pd.read_json(chat_file, lines=True).to_dict(orient="records")
        return chat_data

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, str]:
        cur_data = self.chat_data[index]
        human_input = "<image>\n" + cur_data.get("query", "")
        chatbot_output = cur_data.get("response")
        current_user = pwd.getpwuid(os.getuid()).pw_name
        image_path = cur_data.get("images")
        if current_user == "pico":
            image_path = image_path.replace("/home/qianq/", "/home/pico/")
        return human_input, chatbot_output, image_path


class LlavaDatasetWithSplit(Dataset):
    def __init__(self, dataset_dir: str, is_train: bool = True) -> None:
        super().__init__()

        self.chat_data = self.build_dataset(dataset_dir, is_train)

    def build_dataset(
        self, data_dir: str, is_train: bool = True
    ) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("train.jsonl" if is_train else "test.jsonl")
        chat_data = pd.read_json(chat_file, lines=True).to_dict(orient="records")

        return chat_data

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        human_input = cur_data.get("human_input")
        chatbot_output = cur_data.get("chatbot_output")
        image_path = cur_data.get("image_path")
        return human_input, chatbot_output, image_path


def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path):
    prompt = f"USER: {q_text}\nASSISTANT:"
    image_file = image_path  # "000000039769.jpg"

    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors="pt")  # .to(0, torch.float16)

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=256,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res


class TrainLlavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int, enable_augmentation: bool = True) -> None:
        self.processor = processor
        self.ingnore_index = IGNORE_INDEX
        self.enable_augmentation = enable_augmentation
        
        # 定义图像增强变换
        self.augmentation_transforms = self._build_augmentation_transforms()

    def _build_augmentation_transforms(self):
        """构建图像增强变换"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.RandomVerticalFlip(p=0.3),    # 垂直翻转
            transforms.RandomRotation(degrees=30),    # 随机旋转 ±30度
            transforms.RandomResizedCrop(
                size=(224, 224),  # 根据您的模型输入尺寸调整
                scale=(0.8, 1.2),  # 缩放范围
                ratio=(0.8, 1.2),  # 宽高比范围
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomAffine(
                degrees=0,          # 已经通过RandomRotation处理
                translate=(0.1, 0.1),  # 平移 ±10%
                scale=(0.9, 1.1),   # 缩放
                shear=(-10, 10)     # 剪切变换
            ),
            transforms.ColorJitter(
                brightness=0.2,     # 亮度调整
                contrast=0.2,       # 对比度调整
                saturation=0.2,     # 饱和度调整
                hue=0.1            # 色调调整
            ),
        ])

    def apply_image_augmentation(self, image: Image.Image) -> Image.Image:
        """应用图像增强"""
        if not self.enable_augmentation:
            return image
            
        try:
            # 应用增强变换
            augmented_image = self.augmentation_transforms(image)
            return augmented_image
        except Exception as e:
            print(f"图像增强失败，使用原图像: {e}")
            return image

    def convert_one_piece(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
        # pixel_values: torch.Tensor,
    ):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(
                    q_input_ids.shape, self.ingnore_index
                ),  # query用-100[ignore_index]替代，计算loss时忽略
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            # 加载并增强图像
            image_path = feature[2]
            raw_image = Image.open(image_path)
            
            # 应用图像增强
            augmented_image = self.apply_image_augmentation(raw_image)
            
            # 使用增强后的图像构建输入
            qaimage_output = self.build_qaimage_with_augmented_image(
                self.processor, feature[0], feature[1], augmented_image
            )
            
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )
        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingnore_index,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )
        final_pixel_values = torch.concat(pixel_values, axis=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }

    def build_qaimage_with_augmented_image(self, processor: AutoProcessor, q_text: str, a_text: str, image: Image.Image):
        """使用增强后的图像构建QA输入"""
        prompt = f"USER: {q_text}\nASSISTANT:"
        
        inputs = processor(prompt, image, return_tensors="pt")

        a_input_ids = processor.tokenizer(
            a_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=256,
        )["input_ids"]

        res = QaImageOutput(
            q_input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            a_input_ids=a_input_ids,
        )
        return res


def safe_cat(tensors, dim=0, pad_value=0):
    """
    自动对齐除指定维度外的所有维度，对输入 tensor 列表进行拼接。
    """
    # 计算目标 shape
    max_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        for i in range(len(max_shape)):
            if i != dim:
                max_shape[i] = max(max_shape[i], t.shape[i])

    padded = []
    for t in tensors:
        pad_dims = []
        for i in reversed(range(len(max_shape))):
            if i == dim:
                pad_dims.extend([0, 0])
            else:
                diff = max_shape[i] - t.shape[i]
                pad_dims.extend([0, diff])
        padded_tensor = F.pad(t, pad_dims, value=pad_value)
        padded.append(padded_tensor)

    return torch.cat(padded, dim=dim)


def main(is_train: bool = True):
    data_dir = "data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
    llavadataset = LIDCClassificationDataset(data_dir, is_train=is_train)
    print(len(llavadataset))
    item = llavadataset[0]
    print("human_input:", item[0])
    print("chatbot_output:", item[1])
    print("image_path:", item[2])


def test_llava_cc3m():
    llavadataset = LlavaDatasetWithSplit(
        "data/image-text-to-text/LLaVA-CC3M-Pretrain-595K", is_train=False
    )
    print(len(llavadataset))
    item = llavadataset[0]
    print("human_input:", item[0])
    print("chatbot_output:", item[1])
    print("image_path:", item[2])


import json


def split_cc3m():
    data_path = "data/image-text-to-text/LLaVA-CC3M-Pretrain-595K"
    llavadataset = LlavaDataset(data_path)
    print(len(llavadataset))
    random_seed = 42
    random.seed(random_seed)
    random.shuffle(llavadataset.chat_data)
    ratio = 0.8
    split_index = int(len(llavadataset) * ratio)

    for i in trange(split_index):
        item = llavadataset[i]
        human_input = item[0]
        chatbot_output = item[1]
        image_path = item[2]

        # 处理human_input和chatbot_output
        human_input = human_input.replace("\n", " ")
        chatbot_output = chatbot_output.replace("\n", " ")

        # 保存到文件
        with open(f"{data_path}/train.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "human_input": human_input,
                        "chatbot_output": chatbot_output,
                        "image_path": str(image_path),
                    }
                )
                + "\n"
            )

    for i in trange(split_index, len(llavadataset)):
        item = llavadataset[i]
        human_input = item[0]
        chatbot_output = item[1]
        image_path = item[2]

        # 处理human_input和chatbot_output
        human_input = human_input.replace("\n", " ")
        chatbot_output = chatbot_output.replace("\n", " ")

        # 保存到文件
        with open(f"{data_path}/test.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "human_input": human_input,
                        "chatbot_output": chatbot_output,
                        "image_path": str(image_path),
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    # main(is_train=True)
    # main(is_train=False)
    # main()
    # test_llava_cc3m()
    # split_cc3m()
    test_llava_cc3m()
