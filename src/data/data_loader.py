import pandas as pd


SUPPORTED_LANGS = [
    "ar", "en", "fr", "es", "de",
    "hi", "pt", "it", "bg", "fa",
    "ja", "zh"
]


def load_mms_test_data(test_path: str):
    """
    Load MMS test CSV file.

    Expected columns:
        text, label, language
    """

    df = pd.read_csv(test_path)

    df = df[df["language"].isin(SUPPORTED_LANGS)].copy()

    return df


def split_by_language(df):
    """
    Split dataframe into per-language subsets.
    """

    language_dfs = {}

    for lang in SUPPORTED_LANGS:
        lang_df = df[df["language"] == lang].copy()
        if len(lang_df) > 0:
            language_dfs[lang] = lang_df

    return language_dfs
