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
import traceback
import torch.nn.functional as F


# @dataclass
# class QaImageOutput:
#     q_input_ids: torch.Tensor
#     pixel_values: torch.Tensor
#     a_input_ids: torch.Tensor


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


# def build_qaimage(
#     processor: AutoProcessor,
#     q_text: str,
#     a_text: str,
#     image_path: Path,
#     sys_prompt: str = "You are a helpful assistant.",
# ) -> QaImageOutput:
#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": q_text},
#     ]
#     try:
#         prompt = processor.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#     except TemplateError:
#         messages = [
#             {"role": "user", "content": sys_prompt + "\n" + q_text},
#         ]
#         prompt = processor.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#     except Exception:
#         prompt = f"{sys_prompt}\nUser: {q_text}\nAssistant:"

#     raw_image = Image.open(image_path).convert("RGB")
#     inputs = processor(raw_image, prompt, return_tensors="pt")
#     a_input_ids = processor.tokenizer(
#         a_text,
#         return_tensors="pt",
#         padding="longest",
#         truncation=True,
#         max_length=2048,
#     )["input_ids"]

#     return QaImageOutput(
#         q_input_ids=inputs.get("input_ids"),
#         pixel_values=inputs.get("pixel_values"),
#         a_input_ids=a_input_ids,
#     )


# class TrainLlavaModelCollator:
#     def __init__(
#         self,
#         processor: AutoProcessor,
#         ignore_index: int = -100,
#         system_prompt: Optional[str] = None,
#     ):
#         self.processor = processor
#         self.ignore_index = ignore_index
#         self.system_prompt = system_prompt or "You are a helpful assistant."
#         if not hasattr(self.processor, 'patch_size') or self.processor.patch_size is None:
#             self.processor.patch_size = 14  # LLaVA 通常使用的 patch_size 值


#     def convert_sample(
#         self,
#         q_input_ids: torch.Tensor,
#         a_input_ids: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         eos_token_id = self.processor.tokenizer.eos_token_id
#         eos_tensor = torch.tensor([[eos_token_id]], dtype=torch.long)
#         input_ids = torch.cat([q_input_ids, a_input_ids, eos_tensor], dim=1)
#         labels = torch.cat(
#             [
#                 torch.full_like(q_input_ids, self.ignore_index),
#                 a_input_ids,
#                 eos_tensor,
#             ],
#             dim=1,
#         )
#         return input_ids, labels

#     def __call__(self, features: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
#         input_ids_list = []
#         labels_list = []
#         pixel_values_list = []
#         max_length = 0

#         # TODO collator的问题需要排查一下？训练出来的永远都是同一个答案
#         # 可能是因为没有正确处理输入的tokenization和padding

#         for query, answer, image_path in features:
#             messages = [
#                 {"role": "system", "content": self.system_prompt},
#                 {"role": "user", "content": query},
#             ]
#             try:
#                 prompt = self.processor.tokenizer.apply_chat_template(
#                     messages,
#                     tokenize=False,
#                     add_generation_prompt=True,
#                 )
#             except Exception:
#                 prompt = f"{self.system_prompt}\n{query}"

#             image = Image.open(image_path).convert("RGB")
#             self.processor.image_processor.patch_size = 14  # 如果模型是基于 ViT 或类似结构
#             # print("Patch size:", self.processor.image_processor.patch_size)
#             model_inputs = self.processor(image, prompt, return_tensors="pt")
#             answer_ids = self.processor.tokenizer(
#                 answer,
#                 return_tensors="pt",
#                 max_length=2048,
#                 truncation=True,
#                 padding=False,
#             )["input_ids"]

#             input_ids, labels = self.convert_sample(
#                 model_inputs["input_ids"], answer_ids
#             )
#             input_ids_list.append(input_ids)
#             labels_list.append(labels)
#             pixel_values_list.append(model_inputs["pixel_values"])
#             max_length = max(max_length, input_ids.shape[1])

#         batch_input_ids = []
#         batch_labels = []

#         for input_ids, labels in zip(input_ids_list, labels_list):
#             pad_len = max_length - input_ids.shape[1]
#             pad_input_ids = torch.full(
#                 (1, pad_len), self.processor.tokenizer.pad_token_id, dtype=torch.long
#             )
#             pad_labels = torch.full((1, pad_len), self.ignore_index, dtype=torch.long)
#             batch_input_ids.append(torch.cat([pad_input_ids, input_ids], dim=1))
#             batch_labels.append(torch.cat([pad_labels, labels], dim=1))

#         batch = {
#             "input_ids": torch.cat(batch_input_ids, dim=0),
#             "labels": torch.cat(batch_labels, dim=0),
#             "pixel_values": torch.cat(pixel_values_list, dim=0),
#         }
#         batch["attention_mask"] = (
#             batch["input_ids"] != self.processor.tokenizer.pad_token_id
#         ).long()
#         return batch


# class TrainLlavaModelCollator:
#     def __init__(
#         self,
#         processor: AutoProcessor,
#         ignore_index: int = -100,
#         system_prompt: Optional[str] = None,
#         debug: bool = False,  # 添加调试标志
#     ):
#         self.processor = processor
#         self.ignore_index = ignore_index
#         self.system_prompt = system_prompt or "You are a helpful assistant."
#         self.debug = debug
#         self.call_count = 0  # 记录调用次数
#         if (
#             not hasattr(self.processor, "patch_size")
#             or self.processor.patch_size is None
#         ):
#             self.processor.patch_size = 14

#     def convert_sample(
#         self,
#         q_input_ids: torch.Tensor,
#         a_input_ids: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         eos_token_id = self.processor.tokenizer.eos_token_id
#         eos_tensor = torch.tensor([[eos_token_id]], dtype=torch.long)
#         input_ids = torch.cat([q_input_ids, a_input_ids, eos_tensor], dim=1)
#         labels = torch.cat(
#             [
#                 torch.full_like(q_input_ids, self.ignore_index),
#                 a_input_ids,
#                 eos_tensor,
#             ],
#             dim=1,
#         )
#         return input_ids, labels

#     def __call__(self, features: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
#         input_ids_list = []
#         labels_list = []
#         pixel_values_list = []
#         max_length = 0

#         for i, (query, answer, image_path) in enumerate(features):
#             prompt = f"{self.system_prompt}\n{query}"

#             if self.debug:
#                 print(f"Generated prompt: {prompt[:200]}...")

#             image = Image.open(image_path).convert("RGB")
#             self.processor.image_processor.patch_size = 14
#             model_inputs = self.processor(image, prompt, return_tensors="pt")
#             answer_ids = self.processor.tokenizer(
#                 answer,
#                 return_tensors="pt",
#                 max_length=2048,
#                 truncation=True,
#                 padding=True,
#             )["input_ids"]

#             if self.debug:
#                 print(f"Query tokens: {model_inputs['input_ids'].shape}")
#                 print(f"Answer tokens: {answer_ids.shape}")
#                 print(f"Query token ids: {model_inputs['input_ids'][0, :20]}")
#                 print(f"Answer token ids: {answer_ids[0, :20]}")

#                 # 解码部分tokens查看内容
#                 query_text = self.processor.tokenizer.decode(
#                     model_inputs["input_ids"][0, :50]
#                 )
#                 answer_text = self.processor.tokenizer.decode(answer_ids[0, :50])
#                 print(f"Query decoded: {query_text}")
#                 print(f"Answer decoded: {answer_text}")

#             input_ids, labels = self.convert_sample(
#                 model_inputs["input_ids"], answer_ids
#             )
#             input_ids_list.append(input_ids)
#             labels_list.append(labels)
#             pixel_values_list.append(model_inputs["pixel_values"])
#             max_length = max(max_length, input_ids.shape[1])

#         batch_input_ids = []
#         batch_labels = []

#         for input_ids, labels in zip(input_ids_list, labels_list):
#             pad_len = max_length - input_ids.shape[1]
#             pad_input_ids = torch.full(
#                 (1, pad_len), self.processor.tokenizer.pad_token_id, dtype=torch.long
#             )
#             pad_labels = torch.full((1, pad_len), self.ignore_index, dtype=torch.long)
#             batch_input_ids.append(torch.cat([pad_input_ids, input_ids], dim=1))
#             batch_labels.append(torch.cat([pad_labels, labels], dim=1))

#         batch = {
#             "input_ids": torch.cat(batch_input_ids, dim=0),
#             "labels": torch.cat(batch_labels, dim=0),
#             "pixel_values": torch.cat(pixel_values_list, dim=0),
#         }
#         batch["attention_mask"] = (
#             batch["input_ids"] != self.processor.tokenizer.pad_token_id
#         ).long()
#         return batch

from typing import List, Tuple, Dict, Optional
import torch
from PIL import Image
from transformers import AutoProcessor


class TrainLlavaModelCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        ignore_index: int = -100,
        system_prompt: Optional[str] = None,
        debug: bool = False,
    ):
        self.processor = processor
        self.ignore_index = ignore_index
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.debug = debug
        self.call_count = 0

        # 设置 patch size
        if (
            not hasattr(self.processor, "patch_size")
            or self.processor.patch_size is None
        ):
            self.processor.patch_size = 14

    def convert_sample(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 保证在同一 device 上构造 eos_tensor
        eos_token_id = self.processor.tokenizer.eos_token_id
        eos_tensor = torch.tensor(
            [eos_token_id], dtype=torch.long, device=q_input_ids.device
        ).unsqueeze(0)

        # 检查 answer_ids 是否有非法 token（如超出 vocab 范围）
        assert (
            a_input_ids.max() < self.processor.tokenizer.vocab_size
        ), f"❌ Token ID 超出 vocab 范围：最大 {a_input_ids.max()} >= vocab_size={self.processor.tokenizer.vocab_size}"

        # 拼接输入和标签（问句不参与 loss）
        input_ids = torch.cat([q_input_ids, a_input_ids, eos_tensor], dim=1)
        labels = torch.cat(
            [torch.full_like(q_input_ids, self.ignore_index), a_input_ids, eos_tensor],
            dim=1,
        )

        return input_ids, labels

    def __call__(self, features: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []
        pixel_values_list = []
        max_length = 0

        for i, (query, answer, image_path) in enumerate(features):
            prompt = f"{self.system_prompt}\n{query}"

            if self.debug:
                print(f"\n👉 [{i}] Prompt: {prompt[:100]}...")

            # 图像处理
            image = Image.open(image_path).convert("RGB")
            self.processor.image_processor.patch_size = 14
            model_inputs = self.processor(image, prompt, return_tensors="pt")

            # 文本处理
            answer_ids = self.processor.tokenizer(
                answer,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=True,
            )["input_ids"]

            if self.debug:
                print(f"→ query shape: {model_inputs['input_ids'].shape}")
                print(f"→ answer shape: {answer_ids.shape}")
                print(f"→ query tokens: {model_inputs['input_ids'][0, :10]}")
                print(f"→ answer tokens: {answer_ids[0, :10]}")
                print(
                    f"→ query decoded: {self.processor.tokenizer.decode(model_inputs['input_ids'][0, :50])}"
                )
                print(
                    f"→ answer decoded: {self.processor.tokenizer.decode(answer_ids[0, :50])}"
                )
                print(f"→ max token ID in answer: {answer_ids.max().item()}")

            # 拼接 input + label
            input_ids, labels = self.convert_sample(
                model_inputs["input_ids"], answer_ids
            )
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(model_inputs["pixel_values"])
            max_length = max(max_length, input_ids.shape[1])

        # 统一 pad 成 batch
        batch_input_ids = []
        batch_labels = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_length - input_ids.shape[1]

            pad_input_ids = torch.full(
                (1, pad_len),
                self.processor.tokenizer.pad_token_id,
                dtype=torch.long,
                device=input_ids.device,
            )
            pad_labels = torch.full(
                (1, pad_len), self.ignore_index, dtype=torch.long, device=labels.device
            )

            padded_input_ids = torch.cat([pad_input_ids, input_ids], dim=1)
            padded_labels = torch.cat([pad_labels, labels], dim=1)

            batch_input_ids.append(padded_input_ids)
            batch_labels.append(padded_labels)

            if self.debug:
                print(
                    f"✅ Padded input shape: {padded_input_ids.shape}, labels shape: {padded_labels.shape}"
                )

        # 构造 batch 返回
        batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }

        # Attention mask 自动生成
        batch["attention_mask"] = (
            batch["input_ids"] != self.processor.tokenizer.pad_token_id
        ).long()

        return batch


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

    # 测试 TrainLlavaModelCollator
    print("\n=== 测试 Collator ===")
    from transformers import AutoProcessor

    # 加载处理器 - 使用实际的LLaVA处理器，你需要根据实际情况调整模型名称
    try:
        processor_name = "/home/qianq/model/llava-1.5-7b-hf"
        processor = AutoProcessor.from_pretrained(processor_name)

    except ImportError as e:
        print(f"加载处理器失败: {e}")
        print("请确保已安装所需的依赖库，例如: pip install transformers")
    except Exception as e:
        print(f"测试 collator 时出错: {e}")
        traceback.print_exc()

    # TODO 把所有的数据都跑一遍，并解码labels看下结果，谢谢！
    # from tqdm import trange
    from tqdm import tqdm

    print("\n=== 遍历所有数据并解码labels ===")
    # 创建带调试信息的collator
    debug_collator = TrainLlavaModelCollator(processor=processor)
    pad_token_id = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else 0
    )

    # 分批处理数据，避免内存问题
    batch_size = 2
    total_samples = len(llavadataset)

    bar = tqdm(total=total_samples, desc="Processing batches", unit="batch")
    for start_idx in range(0, total_samples, batch_size):  # 我要所有的样本
        bar.update(batch_size)
        end_idx = min(start_idx + batch_size, total_samples)

        # 获取批次数据
        batch_samples = [llavadataset[i] for i in range(start_idx, end_idx)]

        # 打印原始数据
        # print("\n--- 原始数据 ---")
        # for i, (human_input, chatbot_output, image_path) in enumerate(batch_samples):
        # print(f"样本 {start_idx + i}:")
        # print(f"  输入: {human_input[:100]}...")
        # print(f"  期望输出: {chatbot_output[:100]}...")
        # print(f"  图片路径: {image_path}")
        # 应用collator处理

        batch = debug_collator(batch_samples)

        # 详细分析labels
        # print("\n--- Labels分析 ---")
        labels = batch["labels"]
        querys = batch["input_ids"]
        # images = batch['pixel_values']
        # TODO 对所有batch中的images、querys、labels整合到一起进行unique分析， 分析总共有多少数据， 有多少个unique值，并对labels解码后的unique值进行分析
        # 收集所有数据用于unique分析
        if start_idx == 0:
            # all_images = images
            all_querys = querys
            all_labels = labels
        else:
            try:

                # all_images = safe_cat([all_images, images], dim=0, pad_value=0)        # 图像一般 padding 0
                all_querys = safe_cat(
                    [all_querys, querys], dim=0, pad_value=pad_token_id
                )
                all_labels = safe_cat([all_labels, labels], dim=0, pad_value=-100)
            except Exception as e:
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                # print(querys.shape, labels.shape)
                # print(all_querys.shape, all_labels.shape)
                # # tokenizer解码并分析：
                # for i in range(querys.shape[0]):
                #     print("Label shape:", labels.shape)
                #     print("Max token id:", labels[i, :].max())
                #     print("Min token id:", labels[i, :].min())
                #     print("Dtype:", labels.dtype)
                #     print("Label IDs:", labels[i, :])
                #     print(f"Query {i}: [{processor.tokenizer.decode(querys[i, :], skip_special_tokens=True)}]")
                #     print(f"Label {i}: [{processor.tokenizer.decode(labels[i, :], skip_special_tokens=True)}]")
                # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                raise e

        # 对每个样本解码labels
        for i in range(labels.shape[0]):
            sample_valid_labels = labels[i, :][
                labels[i, :] != debug_collator.ignore_index
            ]
            if len(sample_valid_labels) > 0:
                decoded_labels = processor.tokenizer.decode(
                    sample_valid_labels, skip_special_tokens=True
                )
                # print(f"样本 {start_idx + i} 解码标签: {decoded_labels}")
            else:
                print(f"样本 {start_idx + i} 没有有效标签")

        # 如果是最后一个批次，进行整体unique分析
        if end_idx >= total_samples:
            print(f"\n{'='*60}")
            print("整体数据分析")
            print(f"{'='*60}")

            print(f"总样本数: {all_labels.shape[0]}")
            # print(f"总图片数: {all_images.shape[0]}")
            print(f"总查询数: {all_querys.shape[0]}")

            # 分析labels的unique值
            all_valid_labels = all_labels[all_labels != debug_collator.ignore_index]
            unique_labels = torch.unique(all_valid_labels)
            print(f"所有有效标签数量: {len(all_valid_labels)}")
            print(f"unique标签值数量: {len(unique_labels)}")
            print(f"unique标签值: {unique_labels}")

            # 分析每个样本的解码标签
            decoded_labels_list = []
            for i in range(all_labels.shape[0]):
                sample_valid_labels = all_labels[i, :][
                    all_labels[i, :] != debug_collator.ignore_index
                ]
                if len(sample_valid_labels) > 0:
                    decoded_labels = processor.tokenizer.decode(
                        sample_valid_labels, skip_special_tokens=True
                    )
                    decoded_labels_list.append(decoded_labels)
                else:
                    decoded_labels_list.append("")

            # 统计unique解码标签
            unique_decoded_labels = list(set(decoded_labels_list))
            print(f"unique解码标签数量: {len(unique_decoded_labels)}")
            print("unique解码标签内容:")
            for i, label in enumerate(unique_decoded_labels):
                print(f"  {i}: {label}")

            # 统计每个解码标签的出现次数
            from collections import Counter

            label_counts = Counter(decoded_labels_list)
            print("解码标签出现次数:")
            print(label_counts)

            # 分析query的unique值
            print(f"\n{'='*30} Query分析 {'='*30}")
            all_valid_querys = all_querys[all_querys != pad_token_id]
            unique_querys = torch.unique(all_valid_querys)
            print(f"所有有效查询token数量: {len(all_valid_querys)}")
            print(f"unique查询token数量: {len(unique_querys)}")
            print(f"unique查询token前20个: {unique_querys[:20]}")

            # 分析每个样本的解码查询
            decoded_querys_list = []
            for i in range(all_querys.shape[0]):
                sample_valid_querys = all_querys[i, :][all_querys[i, :] != pad_token_id]
                if len(sample_valid_querys) > 0:
                    decoded_querys = processor.tokenizer.decode(
                        sample_valid_querys, skip_special_tokens=True
                    )
                    decoded_querys_list.append(decoded_querys)
                else:
                    decoded_querys_list.append("")

            # 统计unique解码查询
            unique_decoded_querys = list(set(decoded_querys_list))
            print(f"unique解码查询数量: {len(unique_decoded_querys)}")
            print("unique解码查询内容:")
            for i, query in enumerate(unique_decoded_querys):
                print(f"  {i}: {query}...")

            # 统计每个解码查询的出现次数
            query_counts = Counter(decoded_querys_list)
            print("解码查询出现次数:")
            for query, count in query_counts.most_common(5):  # 显示前5个最常见的
                print(f"  出现{count}次: {query}...")


# def test_tokenize():
#     tokenizer = AutoProcessor.from_pretrained("/home/qianq/model/llava-1.5-7b-hf").tokenizer
#     text = "benign"
#     tokens = tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=2048)

if __name__ == "__main__":
    main(is_train=True)
    main(is_train=False)
