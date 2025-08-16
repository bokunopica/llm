from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

# 使用 CPU
device = torch.device("cpu")

# 加载 PubMedBERT
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

app = FastAPI(title="PubMedBERT CPU Embedding Service with Cached Labels")

# 缓存 labels embedding
LABELS = ["malignant", "benign"]

def encode_texts(texts, batch_size=16):
    """分批次生成 CLS 向量"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    return np.vstack(embeddings)

label_embeddings = encode_texts(LABELS)

# 请求数据模型
class EmbeddingRequest(BaseModel):
    texts: list[str]
    ground_truth: list[str]
    threshold: float = 0.7

@app.post("/judge_answer")
def judge_answer(req: EmbeddingRequest):
    text_embeddings = encode_texts(req.texts)
    ground_truth_embeddings = encode_texts(req.ground_truth)
    results = []

    for emb_text in text_embeddings:
        res = {}
        for label, v_label in zip(LABELS, label_embeddings):
            sim = float(cosine_similarity([emb_text], [v_label])[0][0])
            score = 1.0 if sim >= req.threshold else max(0.0, sim/req.threshold)
            res[label] = {"similarity": sim, "score": score}
        results.append(res)
    return {"results": results}

# 直接用 Python 启动
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
