from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
import pwd
import os
from jinja2.exceptions import TemplateError


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
        chat_file = data_dir.joinpath("train.jsonl" if self.is_train else "test.jsonl")
        print(f"Loading file: {chat_file}, exists: {chat_file.exists()}")
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
        """
        dataset_dir: 数据集所在路径
        is_train: train.jsonl / test.jsonl
        with_attr: 是否带属性
        """
        super().__init__()
        self.is_train = is_train
        self.chat_data = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("train.jsonl" if self.is_train else "test.jsonl")
        chat_data = pd.read_json(chat_file, lines=True).to_dict(orient="records")
        return chat_data

    def __len__(self):
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        # human_input = "<image>" + cur_data.get("query", "")
        human_input = "<image>\n" + cur_data.get("query", "")
        # human_input = human_input.replace(
        #     "。结合以上肺结节特征信息，请对图中的肺结节给出良恶性的分类结果",
        #     "Based on the above lung nodule feature information, please provide a classification result of the malignancy of the lung nodule in the image. The result should be either <ref>malignant</ref> or <ref>benign</ref>.\n",
        # )
        # human_input = human_input.replace(
        #     "请对图中的肺结节给出良恶性的分类结果",
        #     "Please provide a classification result of the malignancy of the lung nodule in the image. The result should be either <ref>malignant</ref> or <ref>benign</ref>.\n",
        # )

        chatbot_output = cur_data.get("response")
        current_user = pwd.getpwuid(os.getuid()).pw_name
        image_path = cur_data.get("images")
        if current_user == "pico":  # 不同机器保存的数据的前缀路径的修改

            image_path = image_path.replace(
                "/home/qianq/",
                "/home/pico/",
            )

        return human_input, chatbot_output, image_path


def build_qaimage(
    processor: AutoProcessor,
    q_text: str,
    a_text: str,
    image_path: Path,
    sys_prompt="You are a helpful assistant.",
):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": q_text},
    ]

    # 尝试使用 apply_chat_template，如果失败则 fallback
    try:
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TemplateError:
        # Mistral-7B-Instruct不支持system prompt
        messages = [
            {"role": "user", "content": sys_prompt + "\n" + q_text},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except ValueError:
        # fallback to simple concatenation if no chat_template is set
        prompt = f"{sys_prompt}\nUser: {q_text}\nAssistant:"

    image_file = image_path
    raw_image = Image.open(image_file).convert("RGB")  # 防止灰度图或模式不符

    inputs = processor(raw_image, prompt, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=2048,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    # print("input keys:", inputs.keys())
    # print("pixel_values shape:", inputs["pixel_values"].shape)
    return res


# class TrainLlavaModelCollator:
#     def __init__(
#         self, processor: AutoProcessor, IGNORE_INDEX: int, SYS_PROMPT: str = None
#     ) -> None:
#         self.processor = processor
#         self.ignore_index = IGNORE_INDEX
#         self.sys_prompt = SYS_PROMPT

#     def convert_one_piece(
#         self,
#         q_input_ids: torch.Tensor,
#         a_input_ids: torch.Tensor,
#         # pixel_values: torch.Tensor,
#     ):
#         input_ids = torch.concat(
#             [
#                 q_input_ids,
#                 a_input_ids,
#                 torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
#             ],
#             axis=1,
#         )
#         labels = torch.concat(
#             [
#                 torch.full(
#                     q_input_ids.shape, self.ignore_index
#                 ),  # query用-100[ignore_index]替代，计算loss时忽略
#                 a_input_ids,
#                 torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
#             ],
#             axis=1,
#         )

#         return input_ids, labels

#     def __call__(self, features: List) -> Dict[str, torch.Tensor]:
#         input_ids_list = []
#         labels_list = []
#         pixel_values = []
#         max_input_len_list = []

#         for feature in features:
#             qaimage_output = build_qaimage(
#                 self.processor,
#                 feature[0],
#                 feature[1],
#                 feature[2],
#                 self.sys_prompt,
#             )
#             temp_input_ids, temp_labels = self.convert_one_piece(
#                 qaimage_output.q_input_ids,
#                 qaimage_output.a_input_ids,
#             )
#             max_input_len_list.append(temp_input_ids.shape[1])
#             input_ids_list.append(temp_input_ids)
#             labels_list.append(temp_labels)
#             pixel_values.append(qaimage_output.pixel_values)

#         max_input_len = max(max_input_len_list)

#         final_input_ids = torch.concat(
#             [
#                 torch.concat(
#                     [
#                         torch.full(
#                             (1, max_input_len - max_input_len_list[index]),
#                             self.processor.tokenizer.pad_token_id,
#                         ),
#                         value,
#                     ],
#                     axis=1,
#                 )
#                 for index, value in enumerate(input_ids_list)
#             ]
#         )
#         final_labels = torch.concat(
#             [
#                 torch.concat(
#                     [
#                         torch.full(
#                             (1, max_input_len - max_input_len_list[index]),
#                             self.ignore_index,
#                         ),
#                         value,
#                     ],
#                     axis=1,
#                 )
#                 for index, value in enumerate(labels_list)
#             ]
#         )
#         final_pixel_values = torch.concat(pixel_values, axis=0)
#         attention_mask = torch.ones_like(final_input_ids)
#         attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0
#         return {
#             "input_ids": final_input_ids,
#             "labels": final_labels,
#             "pixel_values": final_pixel_values,
#             "attention_mask": attention_mask,
#         }


class TrainLlavaModelCollator:
    def __init__(
        self,
        processor: AutoProcessor,
        ignore_index: int = -100,
        system_prompt: str = None,
    ):
        self.processor = processor
        self.ignore_index = ignore_index
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def convert_sample(
        self,
        q_input_ids: torch.Tensor,
        a_input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        拼接 input_ids 和 labels，labels 中 q 部分替换为 ignore_index
        """
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
            # 构建 prompt
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
                # fallback 简单拼接
                prompt = f"{self.system_prompt}\n{query}"

            # 读取图片并处理
            image = Image.open(image_path).convert("RGB")
            model_inputs = self.processor(image, prompt, return_tensors="pt")

            # tokenize answer
            answer_ids = self.processor.tokenizer(
                answer,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=False,
            )["input_ids"]

            # 拼接 input_ids 和 labels
            input_ids, labels = self.convert_sample(
                model_inputs["input_ids"], answer_ids
            )

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(model_inputs["pixel_values"])
            max_length = max(max_length, input_ids.shape[1])

        # Padding to max_length
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

        # Stack
        batch = {
            "input_ids": torch.cat(batch_input_ids, dim=0),
            "labels": torch.cat(batch_labels, dim=0),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
        }

        # 构建 attention_mask
        batch["attention_mask"] = (
            batch["input_ids"] != self.processor.tokenizer.pad_token_id
        ).long()
        return batch


if __name__ == "__main__":
    # data_dir = "/home/pico/mycodes/lungNodule/datasets/LIDC-IDRI-MLLM-CLF"
    # llavadataset = LIDCClassificationDataset(data_dir)
    # print(len(llavadataset))
    # item = llavadataset[0]
    # print("human_input:", item[0])
    # print("chatbot_output:", item[1])
    # print("image_path:", item[2])

    # for item in llavadataset:
    #     print(item)
    # chat_data = pd.read_json("/home/pico/myCodes/lungNodule/datasets/LIDC-IDRI-MLLM-CLF-EN/train.jsonl", lines=True).to_dict(orient="records")
    # print(chat_data[0])
    import json

    with open("/home/pico/myCodes/lungNodule/datasets/LIDC-IDRI-MLLM-CLF-EN/train.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                print(f"Empty line at {i+1}")
                continue
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"Error parsing JSON on line {i+1}: {e}")

