from src.baselines.encoder import EncoderClassifier
from src.baselines.llm_classifier import LLMClassifier
from src.data.data_loader import load_mms_test_data, split_by_language
from src.evaluation.baseline_evaluator import evaluate_baseline


def main():

    # Path to MMS test set (must be downloaded and saved (fully or part of it) locally as CSV)
    test_path = "data/mms_testset.csv"


    df_test = load_mms_test_data(test_path)
    language_dfs = split_by_language(df_test)

    # Choose model name as in paper (from these models), or any other model
    model_name = "bert-base-multilingual-cased"
    # model_name = "xlm-roberta-base"
    # model_name = "bigscience/bloomz-7b1"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "mistralai/Mistral-7B-v0.1"

    if model_name in [
        "bert-base-multilingual-cased",
        "xlm-roberta-base"
    ]:
        model = EncoderClassifier(model_name)
    else:
        model = LLMClassifier(model_name, num_labels=2)

    for lang, df in language_dfs.items():
        print(f"\nEvaluating on: {lang.upper()}")
        evaluate_baseline(model, df)


if __name__ == "__main__":
    main()
