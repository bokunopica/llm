import json
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_jsonl(file_path):
    """加载JSONL文件"""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def normalize_label(label):
    """标准化标签，将所有可能的变体转换为统一格式"""
    if isinstance(label, str):
        label = label.lower().strip()
        if label in ["malignant", "malignant.", "[malignant]"]:
            return "malignant"
        elif label in ["benign", "benign.", "[benign]"]:
            return "benign"
    return label


def extract_predictions_and_labels(results):
    """从结果中提取预测值和真实标签"""
    predictions = []
    labels = []

    for result in results:
        # 提取预测结果
        pred = result.get("response", "").strip()
        pred = normalize_label(pred)

        # 提取真实标签
        label = result.get("labels", "").strip()
        label = normalize_label(label)

        # 只有当预测和标签都有效时才添加
        if pred in ["malignant", "benign"] and label in ["malignant", "benign"]:
            predictions.append(pred)
            labels.append(label)
        else:
            print(
                f"跳过无效数据: pred='{result.get('response', '')}', label='{result.get('labels', '')}'"
            )

    return predictions, labels


def calculate_metrics(predictions, labels):
    """计算各种评估指标"""
    # 转换为二进制标签 (malignant=1, benign=0)
    y_true = [1 if label == "malignant" else 0 for label in labels]
    y_pred = [1 if pred == "malignant" else 0 for pred in predictions]

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 详细的分类报告
    report = classification_report(y_true, y_pred, target_names=["benign", "malignant"])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "total_samples": len(y_true),
        "true_malignant": sum(y_true),
        "true_benign": len(y_true) - sum(y_true),
        "pred_malignant": sum(y_pred),
        "pred_benign": len(y_pred) - sum(y_pred),
    }


def calculate_class_specific_metrics(predictions, labels):
    """计算每个类别的详细指标"""
    # 转换为二进制标签
    y_true = [1 if label == "malignant" else 0 for label in labels]
    y_pred = [1 if pred == "malignant" else 0 for pred in predictions]

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 对于Malignant类（阳性类）
    malignant_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    malignant_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    malignant_f1 = (
        2
        * (malignant_precision * malignant_recall)
        / (malignant_precision + malignant_recall)
        if (malignant_precision + malignant_recall) > 0
        else 0
    )

    # 对于Benign类（阴性类）
    benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    benign_f1 = (
        2 * (benign_precision * benign_recall) / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0
        else 0
    )

    return {
        "malignant": {
            "precision": malignant_precision,
            "recall": malignant_recall,
            "f1_score": malignant_f1,
            "support": tp + fn,
        },
        "benign": {
            "precision": benign_precision,
            "recall": benign_recall,
            "f1_score": benign_f1,
            "support": tn + fp,
        },
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }

def print_results(metrics, class_metrics):
    """打印结果"""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n📊 OVERALL METRICS:")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

    print(f"\n🔍 CONFUSION MATRIX:")
    cm = class_metrics["confusion_matrix"]
    print(f"                 Predicted")
    print(f"                 Benign  Malignant")
    print(f"Actual  Benign   {cm['true_negative']:6d}  {cm['false_positive']:9d}")
    print(f"        Malignant{cm['false_negative']:6d}  {cm['true_positive']:9d}")

    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(metrics["classification_report"])


def save_results(metrics, class_metrics, output_file):
    """保存结果到文件"""
    results = {
        "overall_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
        },
        "dataset_statistics": {
            "total_samples": int(metrics["total_samples"]),
            "true_malignant": int(metrics["true_malignant"]),
            "true_benign": int(metrics["true_benign"]),
            "pred_malignant": int(metrics["pred_malignant"]),
            "pred_benign": int(metrics["pred_benign"]),
        },
        "class_specific_metrics": {
            "malignant": {
                "precision": float(class_metrics["malignant"]["precision"]),
                "recall": float(class_metrics["malignant"]["recall"]),
                "f1_score": float(class_metrics["malignant"]["f1_score"]),
                "support": int(class_metrics["malignant"]["support"]),
            },
            "benign": {
                "precision": float(class_metrics["benign"]["precision"]),
                "recall": float(class_metrics["benign"]["recall"]),
                "f1_score": float(class_metrics["benign"]["f1_score"]),
                "support": int(class_metrics["benign"]["support"]),
            },
        },
        "confusion_matrix": {
            "true_negative": int(class_metrics["confusion_matrix"]["true_negative"]),
            "false_positive": int(class_metrics["confusion_matrix"]["false_positive"]),
            "false_negative": int(class_metrics["confusion_matrix"]["false_negative"]),
            "true_positive": int(class_metrics["confusion_matrix"]["true_positive"]),
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")


def eval(ckpt_path):
    # 加载结果
    input_file = os.path.join(ckpt_path, "inference_results.jsonl")
    print(f"Loading results from: {input_file}")
    results = load_jsonl(input_file)
    print(f"Loaded {len(results)} results")

    # 提取预测和标签
    predictions, labels = extract_predictions_and_labels(results)
    print(f"Valid predictions: {len(predictions)}")

    if len(predictions) == 0:
        print("❌ No valid predictions found!")
        return

    # 计算指标
    metrics = calculate_metrics(predictions, labels)
    class_metrics = calculate_class_specific_metrics(predictions, labels)

    # 打印结果
    print_results(metrics, class_metrics)

    # 保存结果
    save_results(
        metrics,
        class_metrics,
        output_file=os.path.join(ckpt_path, "evaluation_results.json"),
    )


def main():
    RESULT_FOLDER = "/home/qianq/mycodes/llm/results/"

    # print(
    #     "############## swift-projector-llava-1.5-7b-hf-epoch=5-lr=1e-5 ##############"
    # )
    # eval(
    #     os.path.join(
    #         RESULT_FOLDER,
    #         "swift-projector-llava-1.5-7b-hf-epoch=5-lr=1e-5",
    #         "v0-20250710-154830",
    #         "checkpoint-250",
    #     ),
    # )
    # eval(
    #     os.path.join(
    #         RESULT_FOLDER,
    #         "swift-projector-llava-1.5-7b-hf-epoch=5-lr=1e-5",
    #         "v0-20250710-154830",
    #         "checkpoint-410",
    #     ),
    # )


    print(
        "############## swift-projector-llava-1.5-7b-hf-EPOCH=5-LR=1e-6 ##############"
    )
    eval(
        os.path.join(
            RESULT_FOLDER,
            "swift-projector-llava-1.5-7b-hf-EPOCH=5-LR=1e-6",
            "v1-20250711-140224",
            "checkpoint-250",
        ),
    )
    eval(
        os.path.join(
            RESULT_FOLDER,
            "swift-projector-llava-1.5-7b-hf-EPOCH=5-LR=1e-6",
            "v1-20250711-140224",
            "checkpoint-410",
        ),
    )



if __name__ == "__main__":
    main()
