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

        # 检查是否包含malignant关键词
        if any(
            keyword in label
            for keyword in ["malignant", "cancer", "malicious", "suspicious"]
        ):
            return "malignant"

        # 检查是否包含benign关键词
        if any(
            keyword in label
            for keyword in ["benign", "normal", "healthy", "non-malignant"]
        ):
            return "benign"

        # 精确匹配
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
        else:
            predictions.append("benign")  # 默认预测为benign
            # print(f"Invalid prediction or label: {pred} | {label}")
        labels.append(label)

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
    report = classification_report(y_true, y_pred, target_names=["benign", "malignant"], digits=5)

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


# def print_results(metrics, class_metrics=None):
def print_results(metrics):
    """打印结果"""
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n📊 OVERALL METRICS:")
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")

    # print(f"\n🔍 CONFUSION MATRIX:")
    # cm = class_metrics["confusion_matrix"]
    # print(f"                 Predicted")
    # print(f"                 Benign  Malignant")
    # print(f"Actual  Benign   {cm['true_negative']:6d}  {cm['false_positive']:9d}")
    # print(f"        Malignant{cm['false_negative']:6d}  {cm['true_positive']:9d}")

    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(metrics["classification_report"])
    print(type(metrics["classification_report"]))


def eval(input_file):
    # 加载结果

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
    calculate_class_specific_metrics(predictions, labels)


    # 打印结果
    # print_results(metrics, class_metrics)
    print_results(metrics)

    # # 保存结果
    # save_results(
    #     metrics,
    #     class_metrics,
    #     output_file=input_file.replace(".jsonl", "_evaluation.json"),
    # )


def main():
    import argparse

    argparse.parser = argparse.ArgumentParser(description="Evaluate inference results")
    argparse.parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing inference results",
    )
    eval(argparse.parser.parse_args().input_file)


if __name__ == "__main__":
    main()
