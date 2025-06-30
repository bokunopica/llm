import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel
import threading

base_model_path = "/home/pico/model/internlm3-8b-instruct"
lora_path = "/home/pico/myCodes/llm/output/output/internlm3-8b-instruct/v0-20250603-190933/checkpoint-3500"


tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="cpu",  # 全部加载到CPU
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

def build_prompt(system_prompt, message, history):
    prompt = system_prompt + "\n"
    for user_msg, bot_msg in history:
        prompt += f"用户：{user_msg}\n医生：{bot_msg}\n"
    prompt += f"用户：{message}\n医生："
    return prompt

def respond(message, history):
    full_prompt = build_prompt(system_prompt, message, history)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}  # 明确放CPU
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    def generate():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            print("生成异常:", e)
            raise e
    thread = threading.Thread(target=generate)
    thread.start()
    partial_output = ""
    for new_token in streamer:
        partial_output += new_token
        yield partial_output

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
