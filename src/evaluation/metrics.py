import re
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)


# ---------------------------------------
# LLM Response → Numeric Label Mapping
# ---------------------------------------

def map_answer_to_label(answer: str):
    """
    Converts raw LLM text output into numeric label.
    1 = Positive
    0 = Negative
    """

    if pd.isna(answer) or not str(answer).strip():
        return -1

    answer_clean = re.sub(r"\s+", " ", str(answer).lower())

    if "positive" in answer_clean:
        return 1
    elif "negative" in answer_clean:
        return 0

    return -1


# ---------------------------------------
# Main Evaluation Function
# ---------------------------------------

def evaluate_predictions(df: pd.DataFrame, prediction_column="predictions"):
    """
    Generic evaluation function.

    Works for:
    - Encoder baselines (column = "predictions")
    - RAG / Cache models (column = "response")
    """

    if prediction_column == "response":
        df["predictions"] = df["response"].apply(map_answer_to_label)
    else:
        df["predictions"] = df[prediction_column]

    y_true = df["label"]
    y_pred = df["predictions"]

    report = classification_report(y_true, y_pred, digits=3)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    weighted_precision = precision_score(y_true, y_pred, average="weighted")
    weighted_recall = recall_score(y_true, y_pred, average="weighted")

    print(report)
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
    }
