import requests

API_URL = "http://127.0.0.1:8000/judge_answer"

def reward_function(sample, threshold=0.7):
    answer = sample['response']
    resp = requests.post(API_URL, json={"texts": [answer], "threshold": threshold})
    result = resp.json()["results"][0]

    # reward = malignant_score - benign_score
    reward = result["malignant"]["score"] - result["benign"]["score"]
    return reward

if __name__ == "__main__":
    sample = {
        "response": "The tumor is benign and does not require immediate intervention."
    }
    reward = reward_function(sample)
    print(f"Reward: {reward}")