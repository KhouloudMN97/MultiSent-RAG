from src.data.wikipedia_loader import WikipediaLoader
import pandas as pd
import os


def main():

    languages = ['en', 'fr', 'ar', 'es', 'de', 'hi', 'pt', 'it']

    topics = {
        'en': ["Positive emotions", "Negative emotions", "Sentiment expression"],
        'fr': ["Émotions positives", "Émotions négatives", "Expression des sentiments"],
        'ar': ["مشاعر إيجابية", "مشاعر سلبية", "تعبير عن المشاعر"],
        'es': ["Emociones positivas", "Emociones negativas", "Expresión de sentimientos"],
        'de': ["Positive Emotionen", "Negative Emotionen", "Gefühlsausdruck"],
        'hi': ["सकारात्मक भावनाएँ", "नकारात्मक भावनाएँ", "भावना की अभिव्यक्ति"],
        'pt': ["Emoções positivas", "Emoções negativas", "Expressão de sentimentos"],
        'it': ["Emozioni positive", "Emozioni negative", "Espressione dei sentimenti"]
    }

    all_docs = []

    for lang in languages:
        loader = WikipediaLoader(lang)
        docs = loader.extract_topics(topics[lang], max_results=100)
        all_docs.extend(docs)

    wiki_df = pd.DataFrame([
        {
            "text": doc.text,
            "language": doc.metadata["language"],
            "title": doc.metadata["title"],
            "source": doc.metadata["source"],
            "description": doc.metadata["description"]
        }
        for doc in all_docs
    ])

    os.makedirs("data", exist_ok=True)
    wiki_df.to_csv("data/wikipedia.csv", index=False)

    print(f"Wikipedia documents saved: {len(wiki_df)}")


if __name__ == "__main__":
    main()
