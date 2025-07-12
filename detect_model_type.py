# detect_model_type.py
import json
import os
from swift.llm import get_model_info_meta

def detect_model_type(model_path):
    """检测模型类型"""
    print(f"检测模型类型: {model_path}")
    
    # 读取配置文件
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("配置文件信息:")
        print(f"  model_type: {config.get('model_type', 'N/A')}")
        print(f"  architectures: {config.get('architectures', 'N/A')}")
        
        # 根据配置推断Swift模型类型
        model_type = config.get('model_type', '')
        architectures = config.get('architectures', [])
        
        if 'llava_next' in model_type.lower():
            if any('mistral' in arch.lower() for arch in architectures):
                swift_model_type = "llava1_6_mistral_hf"
            else:
                swift_model_type = "llava_next_qwen_hf"
        elif 'llava' in model_type.lower():
            swift_model_type = "llava1_5_hf"
        else:
            swift_model_type = "llava1_6_mistral_hf"  # 默认
        
        print(f"推荐的Swift模型类型: {swift_model_type}")
        
        # 测试模型类型
        possible_types = [
            "llava1_6_mistral_hf",
            "llava1_5_hf", 
            "llava_next_qwen_hf",
            "llava1_6_vicuna_hf",
            "llava1_6_yi_hf"
        ]
        
        print("\n测试Swift模型类型:")
        for test_type in possible_types:
            try:
                model_info, model_meta = get_model_info_meta(
                    model_id_or_path=model_path,
                    model_type=test_type
                )
                print(f"✅ {test_type}: 成功")
                return test_type
            except Exception as e:
                print(f"❌ {test_type}: {str(e)[:100]}...")
        
        return swift_model_type
    else:
        print("❌ 配置文件不存在")
        return None

if __name__ == "__main__":
    model_path = "/home/qianq/model/llava1_6-mistral-7b-instruct"
    correct_type = detect_model_type(model_path)
    if correct_type:
        print(f"\n推荐使用模型类型: {correct_type}")