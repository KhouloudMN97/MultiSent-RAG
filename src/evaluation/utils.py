import re
import pandas as pd


def map_answer_to_label(answer):

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
