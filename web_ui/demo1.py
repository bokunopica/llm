import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import threading
from transformers import TextIteratorStreamer

# === 模型加载 ===
base_model_path = "/home/pico/model/internlm3-8b-instruct"
lora_path = "/home/pico/myCodes/llm/output/output/internlm3-8b-instruct/v0-20250603-190933/checkpoint-3500"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, lora_path, device_map="cuda:0")
model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
- 不要输出性病相关.
"""

def build_prompt(system_prompt, messages):
    prompt = system_prompt + "\n"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "
    return prompt

def render_messages(messages):
    html = ""
    for msg in messages:
        role_class = "user" if msg["role"] == "user" else "bot"
        content = msg["content"].replace(system_prompt, '').replace("\n", "<br>")

        # 如果是 assistant 的回答，只保留 Assistant: 之后的部分
        if role_class == "bot":
            if "Assistant:" in content:
                content = content.split("Assistant:")[-1].strip()

        html += f"""
        <div class="message {role_class}">
            <div class="content">{content}</div>
        </div>
        """
    return html

def model_inference_stream(user_input, history):
    history = history + [{"role": "user", "content": user_input}]
    prompt = build_prompt(system_prompt, history)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=10.0)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.14,
        streamer=streamer,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        # 实时显示，不包含旧内容
        yield gr.update(
            value=render_messages(history + [{"role": "assistant", "content": generated_text.strip()}])
        ), "", history

    # ❗只追加新生成部分
    history.append({"role": "assistant", "content": generated_text.strip()})
    yield gr.update(value=render_messages(history), visible=True), "", history



# 自定义CSS（保持你的样式）
custom_css = """
body {
    background-color: #0f1117;
    color: #ffffff;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}
.gradio-container {
    max-width: 700px;
    margin: auto;
    padding: 2em;
    border-radius: 12px;
    background: linear-gradient(145deg, #1b1e29, #13151c);
    box-shadow: 0 0 15px rgba(0,0,0,0.6);
}
h1 {
    text-align: center;
    color: #7dd3fc;
    margin-bottom: 1em;
}
.message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 15px;
}
/* 用户头像：蓝色圆形+emoji */
.message.user::before {
    content: '👤';
    display: inline-flex;
    justify-content: center;
    align-items: center;
    min-width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #3b82f6;
    color: white;
    font-size: 20px;
    flex-shrink: 0;
}
/* 助手头像：绿色圆形+emoji */
.message.bot::before {
    content: '🤖';
    display: inline-flex;
    justify-content: center;
    align-items: center;
    min-width: 36px;
    height: 36px;
    border-radius: 50%;
    background-color: #10b981;
    color: white;
    font-size: 20px;
    flex-shrink: 0;
}
.message .content {
    background-color: #1f2937;
    padding: 12px 16px;
    border-radius: 12px;
    max-width: 80%;
    white-space: pre-wrap;
    font-size: 16px;
    line-height: 1.4;
}
.message.user .content {
    color: #cbd5e1;
}
.message.bot .content {
    color: #d1fae5;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🤖 医疗问答助手（肝胆科医生）")
    chat_html = gr.HTML(render_messages([]), elem_id="chat-container", visible=True)

    user_input = gr.Textbox(placeholder="请输入您的问题，然后回车", show_label=False)
    clear_btn = gr.Button("清空聊天")

    state = gr.State([])

    user_input.submit(model_inference_stream, [user_input, state], [chat_html, user_input, state], queue=True)
    clear_btn.click(lambda: (gr.update(value=""), [],), [ ], [chat_html, state], queue=False)

demo.launch(server_port=8000)
