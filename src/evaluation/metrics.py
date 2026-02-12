from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_predictions(df):

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
    print(f"Weighted F1: {weighted_f1:.3f}")

    return {
        "accuracy": acc,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
    }
