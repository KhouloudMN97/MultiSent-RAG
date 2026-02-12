from src.evaluation.metrics import compute_metrics


def evaluate_baseline(model, df):
    """
    Evaluate encoder or LLM baseline models.

    Assumes:
        model.predict() returns numeric labels (0/1)
        df["label"] contains ground truth (0/1)
    """

    predictions = model.predict(df["text"].tolist())
    return compute_metrics(df["label"], predictions)
