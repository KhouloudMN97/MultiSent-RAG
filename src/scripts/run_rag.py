import pandas as pd
from src.rag.multisent_rag import MultiSentRAG


def main():

    model = MultiSentRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.1"
        # or "meta-llama/Meta-Llama-3-8B-Instruct"
    )

        seen_languages = ["en", "fr", "ar", "es", "de", "hi", "pt", "it"]
    unseen_languages = ["bg", "fa", "ja", "zh"]

    for lang in languages:
        path = f"{base_path}/test_set_{lang}.csv"

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing file for {lang}")
            continue

        print(f"Running {lang}...")

        # Decide mode based on language
        if lang in seen_languages:
            mode = "fewshot"
        else:
            mode = "zeroshot"

        predictions = []

        for text in df["text"]:
            pred = model.predict(text, mode=mode)
            predictions.append(pred)

        df["prediction"] = predictions
        df.to_csv(f"data/results_{lang}.csv", index=False)


if __name__ == "__main__":
    main()
