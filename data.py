from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
import os
import pwd
from jinja2.exceptions import TemplateError


@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str, is_train: bool = True) -> None:
        super().__init__()
        self.is_train = is_train
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("train.jsonl" if self.is_train else "test.jsonl")
        image_dir = data_dir.joinpath("images_dl")
        print(f"Loading file: {chat_file}, exists: {chat_file.exists()}")
        chat_data = pd.read_json(chat_file, lines=True).to_dict(orient="records")
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


def build_qaimage(
    processor: AutoProcessor,
    q_text: str,
    a_text: str,
    image_path: Path,
    sys_prompt: str = "You are a helpful assistant.",
) -> QaImageOutput:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": q_text},
    ]
    try:
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TemplateError:
        messages = [
            {"role": "user", "content": sys_prompt + "\n" + q_text},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{sys_prompt}\nUser: {q_text}\nAssistant:"

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, prompt, return_tensors="pt")
    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=2048,
    )["input_ids"]

    return QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )


class TrainLlavaModelCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        ignore_index: int = -100,
        system_prompt: Optional[str] = None,
    ):
        self.processor = processor
        self.ignore_index = ignore_index
        self.system_prompt = system_prompt or "You are a helpful assistant."
        if not hasattr(self.processor, 'patch_size') or self.processor.patch_size is None:
            self.processor.patch_size = 14  # LLaVA 通常使用的 patch_size 值
        

    def convert_sample(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_tensor = torch.tensor([[eos_token_id]], dtype=torch.long)
        input_ids = torch.cat([q_input_ids, a_input_ids, eos_tensor], dim=1)
        labels = torch.cat(
            [
                torch.full_like(q_input_ids, self.ignore_index),
                a_input_ids,
                eos_tensor,
            ],
            dim=1,
        )
        return input_ids, labels

    def __call__(self, features: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_length = 0

        for query, answer, image_path in features:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ]
            try:
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = f"{self.system_prompt}\n{query}"

            image = Image.open(image_path).convert("RGB")
            self.processor.image_processor.patch_size = 14  # 如果模型是基于 ViT 或类似结构
            # print("Patch size:", self.processor.image_processor.patch_size)
            model_inputs = self.processor(image, prompt, return_tensors="pt")
            answer_ids = self.processor.tokenizer(
                answer,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=False,
            )["input_ids"]

            input_ids, labels = self.convert_sample(
                model_inputs["input_ids"], answer_ids
            )
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(model_inputs["pixel_values"])
            max_length = max(max_length, input_ids.shape[1])

        batch_input_ids = []
        batch_labels = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_length - input_ids.shape[1]
            pad_input_ids = torch.full(
                (1, pad_len), self.processor.tokenizer.pad_token_id, dtype=torch.long
            )
            pad_labels = torch.full((1, pad_len), self.ignore_index, dtype=torch.long)
            batch_input_ids.append(torch.cat([pad_input_ids, input_ids], dim=1))
            batch_labels.append(torch.cat([pad_labels, labels], dim=1))

        batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }
        batch["attention_mask"] = (
            batch["input_ids"] != self.processor.tokenizer.pad_token_id
        ).long()
        return batch


if __name__ == "__main__":
    data_dir = "data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN"
    llavadataset = LIDCClassificationDataset(data_dir)
    print(len(llavadataset))
    item = llavadataset[0]
    print("human_input:", item[0])
    print("chatbot_output:", item[1])
    print("image_path:", item[2])
    
    print("\n=== 数据集示例 ===")
    for i, item in enumerate(llavadataset):
        if i >= 3:  # 只显示前3个样本
            break
        print(f"样本 {i}:", item)

    # 测试 TrainLlavaModelCollator
    print("\n=== 测试 Collator ===")
    from transformers import AutoProcessor
    
    # 加载处理器 - 使用实际的LLaVA处理器，你需要根据实际情况调整模型名称
    try:
        processor_name = "llava-hf/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(processor_name)
        
        # 创建 collator
        collator = TrainLlavaModelCollator(processor=processor)
        print("创建了 collator:", collator)
        
        # 准备批处理样本 - 只取前2个样本作为示例
        batch_samples = [llavadataset[i] for i in range(min(2, len(llavadataset)))]
        print(f"准备处理 {len(batch_samples)} 个样本")
        
        # 应用 collator 进行批处理
        try:
            batch = collator(batch_samples)
            
            # 输出批处理结果的形状和内容摘要
            print("\n批处理结果:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} 形状: {value.shape}")
                    if key == "input_ids":
                        # 显示第一个样本的部分标记ID
                        print(f"第一个样本的前10个标记: {value[0, :10]}")
                    elif key == "labels":
                        # 计算非忽略标记的数量
                        valid_tokens = (value != collator.ignore_index).sum().item()
                        print(f"有效标签数量: {valid_tokens}")
                else:
                    print(f"{key}: {type(value)}")
            
        except Exception as e:
            print(f"批处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"加载处理器失败: {e}")
        print("请确保已安装所需的依赖库，例如: pip install transformers")
    except Exception as e:
        print(f"测试 collator 时出错: {e}")
        import traceback
        traceback.print_exc()
