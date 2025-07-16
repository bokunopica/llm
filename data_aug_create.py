import json
import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """保存数据到jsonl文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def horizontal_flip(image):
    """水平翻转"""
    return cv2.flip(image, 1)

def vertical_flip(image):
    """垂直翻转"""
    return cv2.flip(image, 0)

def rotate_image(image, angle):
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

class MedicalDataAugmentation:
    def __init__(self, image_dir, output_dir, output_jsonl):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_jsonl = output_jsonl
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义医学图像专用的增强方法（仅翻转和旋转）
        self.augmentations = {
            'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip,
            'rotate_90': lambda img: rotate_image(img, 90),
            'rotate_180': lambda img: rotate_image(img, 180),
            'rotate_270': lambda img: rotate_image(img, 270),
        }
    
    def augment_image(self, image_path, augment_type):
        """对单个图像进行数据增强"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"警告: 无法读取图像 {image_path}")
            return None
        
        # 应用增强
        augmented_image = self.augmentations[augment_type](image)
        
        # 生成新的文件名
        original_name = image_path.stem
        extension = image_path.suffix
        new_name = f"{original_name}_{augment_type}{extension}"
        new_path = self.output_dir / new_name
        
        # 保存增强后的图像
        cv2.imwrite(str(new_path), augmented_image)
        
        return new_name
    
    def process_dataset(self, jsonl_path):
        """处理整个数据集，为每张图片生成所有增强类型"""
        
        # 加载原始数据
        original_data = load_jsonl(jsonl_path)
        augmented_data = []
        
        # 医学图像增强类型（仅翻转和旋转）
        medical_augment_types = list(self.augmentations.keys())
        
        print(f"开始处理 {len(original_data)} 个样本...")
        print(f"使用的增强类型: {medical_augment_types}")
        
        for i, item in enumerate(original_data):
            # 获取原始图像路径
            original_image_name = Path(item['images']).name
            original_image_path = self.image_dir / original_image_name
            
            # 检查原始图像是否存在
            if not original_image_path.exists():
                print(f"警告: 原始图像不存在 {original_image_path}")
                continue
            
            # 复制原始图像到输出目录
            shutil.copy2(str(original_image_path), str(self.output_dir / original_image_name))
            
            # 添加原始数据到增强数据集
            original_item = item.copy()
            original_item['images'] = str(self.output_dir / original_image_name)
            augmented_data.append(original_item)
            
            # 为每个图像生成所有增强样本
            for augment_type in medical_augment_types:
                new_image_name = self.augment_image(original_image_path, augment_type)
                if new_image_name:
                    # 创建增强数据的标注
                    augmented_item = item.copy()
                    augmented_item['images'] = str(self.output_dir / new_image_name)
                    augmented_data.append(augmented_item)
            
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(original_data)} 个样本")
        
        # 保存增强后的数据集
        save_jsonl(augmented_data, self.output_jsonl)
        
        print(f"医学图像数据增强完成!")
        print(f"原始样本数: {len(original_data)}")
        print(f"增强后样本数: {len(augmented_data)}")
        print(f"增强倍数: {len(augmented_data) / len(original_data):.2f}")
        print(f"增强后的数据集保存在: {self.output_jsonl}")
        print(f"增强后的图像保存在: {self.output_dir}")

def main():
    # 医学图像数据增强配置
    config = {
        'jsonl_path': '/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN/train.jsonl',
        'image_dir': '/home/qianq/mycodes/lungNodule/datasets/LIDC-IDRI-MLLM-CLF/images',
        'output_dir': '/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN-AUG/images',
        'output_jsonl': '/home/qianq/mycodes/llm/data/image-text-to-text/LIDC-IDRI-MLLM-CLF-EN-AUG/train.jsonl'
    }
    
    # 创建医学图像数据增强器
    augmentor = MedicalDataAugmentation(
        image_dir=config['image_dir'],
        output_dir=config['output_dir'],
        output_jsonl=config['output_jsonl']
    )
    
    # 执行数据增强
    augmentor.process_dataset(config['jsonl_path'])

if __name__ == "__main__":
    main()