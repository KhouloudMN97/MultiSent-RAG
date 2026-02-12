import re
import pandas as pd
from src.evaluation.metrics import compute_metrics


def map_answer_to_label(answer: str) -> int:
    """
    Convert raw LLM response into numeric sentiment label.

    Label convention:
        0 -> Negative
        1 -> Positive
    """

    if pd.isna(answer) or not str(answer).strip():
        return -1

    answer_cleaned = re.sub(
        r'assistant\s*',
        '',
        str(answer),
        flags=re.IGNORECASE
    ).strip().lower()

    if "positive" in answer_cleaned:
        return 1
    elif "negative" in answer_cleaned:
        return 0

    return -1


def evaluate_rag(df):
    """
    Evaluate RAG / RAG-Cache models.

    Assumes:
        df["response"] contains raw LLM outputs
        df["label"] contains ground truth labels (0/1)
    """

    predictions = df["response"].apply(map_answer_to_label)
    return compute_metrics(df["label"], predictions)
