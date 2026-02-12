"""
Entry point for running encoder baselines (mBERT / XLM-R) on MMS test set.
"""

from src.baselines.encoder import EncoderClassifier
from src.core.data_loader import load_mms_test_data, split_by_language
from src.core.evaluator import evaluate_multilingual


def main():

    test_path = "data/mms_testset.csv" # you dataset preprocessed and saved

    df_test = load_mms_test_data(test_path)

    language_dfs = split_by_language(df_test)

    # Choose model:
    model_name = "bert-base-multilingual-cased"
    # model_name = "xlm-roberta-base"

    model = EncoderClassifier(model_name)

    evaluate_multilingual(model, language_dfs)


if __name__ == "__main__":
    main()
