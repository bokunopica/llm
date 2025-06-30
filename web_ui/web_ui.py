# web_ui.py
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
import threading
import time


# torch.cuda.set_per_process_memory_fraction(0.3, device=0)  # 限制0号GPU最多用25%显存

# 模型路径
base_model_path = "/home/pico/model/internlm3-8b-instruct"
lora_path = "/home/pico/myCodes/llm/output/output/internlm3-8b-instruct/v0-20250603-190933/checkpoint-3500"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

device_map = {
    "model.embed_tokens": "cpu",
    "model.norm": "cpu",
    "lm_head": "cpu",
}
for i in range(48):
    device_map[f"model.layers.{i}"] = 0 if i <= 12 else "cpu"


def print_model_structure(model, max_depth=2, indent=0):
    if indent > max_depth:
        return

    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        print("  " * indent + f"{name:<30} | {param_count / 1e6:.2f}M")
        print_model_structure(module, max_depth, indent + 1)
def print_device_map(model):
    for name, param in model.named_parameters():
        if not str(param.device).startswith("cuda"):
            print(name, param.device)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    # device_map="cpu",
    trust_remote_code=True,
)

# print_device_map(base_model)

peft_device_map = {}
for k,v in device_map.items():
    peft_device_map[f"base_model.model.{k}"] = v

model = PeftModel.from_pretrained(
    base_model, 
    lora_path, 
    device_map="cuda:0",
    # device_map="cpu",
)
# TODO 打印一下模型的各个部分和参数大小，为device_map提供参考
# print_device_map(model)


# print("\n模型结构（最多展开 3 层）：")
# print_model_structure(model, max_depth=4)

model.eval()



print_device_map(model)

# 系统提示词
system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- 不要输出性病相关.
"""

# 构造对话上下文
def build_prompt(system_prompt, message, history):
    prompt = system_prompt + "\n"

    for user_msg, bot_msg in history:
        prompt += f"User:{user_msg}\nAssistant:{bot_msg}\n"

    prompt += f"User:{message}\nAssistant:"
    return prompt

# 推理函数：流式输出 + 历史记忆
def respond(message, history):
    full_prompt = build_prompt(system_prompt, message, history)

    # inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda:0")  # 或"cuda:0"，根据实际设备放置决定


    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_length=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.15,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_output = ""
    for new_token in streamer:
        partial_output += new_token
        yield partial_output


def run_startup_benchmark():
    print("\n🧪 正在运行启动时推理 benchmark（问题：你是谁？）...\n")
    
    test_prompt = build_prompt(system_prompt, "你是谁？", [])
    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda:0")  # 或 "cuda:0" if applicable

    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.7,
        repetition_penalty=1.16,
    )

    total_time = time.time() - start_time
    generated_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]

    decoded_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(f"📝 回答内容: {decoded_output.strip()}")
    print(f"🚀 启动时推理耗时: {total_time:.2f} 秒")
    print(f"📊 输出 token 数量: {generated_tokens}")
    print(f"⚡ 平均速度: {generated_tokens / total_time:.2f} tokens/s\n")

# 加载完模型后执行一次 benchmark
run_startup_benchmark()

# 启动 Gradio Chat UI
gr.ChatInterface(
    fn=respond,
    title="医疗问答助手（肝胆科医生）",
    examples=[
        "脂肪肝如何治疗？",
        "胆结石的症状有哪些？",
        "如何预防肝硬化？",
        "乙肝是否可以根治？"
    ],
).launch(server_port=7860)
