from src.evaluation.metrics import evaluate_predictions


def evaluate_multilingual(model, language_dfs):

    metrics_by_lang = {}

    for lang, df in language_dfs.items():

        print(f"\nEvaluating on: {lang.upper()}")

        predictions = model.predict(df["text"].tolist())
        df["predictions"] = predictions

        metrics = evaluate_predictions(df)

        metrics_by_lang[lang] = metrics

    return metrics_by_lang
