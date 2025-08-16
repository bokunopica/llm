import requests

API_URL = "http://127.0.0.1:19191/judge_answer"


def reward_function(sample, prediction, threshold=0.7):
    """
    sample: 数据样本, 包含 'response' 字段作为 gold label
    prediction: 模型生成的文本
    """
    gold_label = sample["response"].strip().lower()  # 'malignant' 或 'benign'
    other_label = "benign" if gold_label == "malignant" else "malignant"

    # 调用本地评判 API
    resp = requests.post(API_URL, json={"texts": [prediction], "threshold": threshold})
    # print(resp.json())
    result = resp.json()["results"][
        0
    ]  # 假设返回格式: {"malignant": {"score": ...}, "benign": {"score": ...}}
    # print(f"Sample prediction result: {result}")

    # reward = gold_label 的分数减去另一标签的分数
    reward = result[gold_label]["score"] - result[other_label]["score"]
    return reward


if __name__ == "__main__":
    sample = {"response": "benign"}
    prediction = "The tumor is benign and does not require immediate intervention."
    reward = reward_function(sample, prediction)
    print(f"Reward: {reward}")
