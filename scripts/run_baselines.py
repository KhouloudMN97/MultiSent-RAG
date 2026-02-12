from src.baselines.encoder import EncoderClassifier
from src.baselines.llm_classifier import LLMClassifier
from src.core.data_loader import load_mms_test_data, split_by_language
from src.core.evaluator import evaluate_multilingual


def main():

    test_path = "data/mms_testset.csv"

    df_test = load_mms_test_data(test_path)
    language_dfs = split_by_language(df_test)

    # Choose model name exactly as in paper
    model_name = "bert-base-multilingual-cased"
    # model_name = "xlm-roberta-base"
    # model_name = "bigscience/bloomz-7b1"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "mistralai/Mistral-7B-v0.1"

    # Encoder models
    if model_name in [
        "bert-base-multilingual-cased",
        "xlm-roberta-base"
    ]:
        model = EncoderClassifier(model_name)

    # LLM models (quantized + LoRA inside LLMClassifier)
    else:
        model = LLMClassifier(model_name, num_labels=2)

    evaluate_multilingual(model, language_dfs)


if __name__ == "__main__":
    main()
