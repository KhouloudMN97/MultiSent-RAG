import pandas as pd
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.pipeline.multisent_rag import MultiSentRAG
from src.evaluation.rag_evaluator import map_answer_to_label
from src.evaluation.metrics import compute_metrics


VECTORSTORE_PATH = "data/chroma_vectorstore"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

SEEN_LANGUAGES = ["en", "fr", "ar", "es", "de", "hi", "pt", "it"]
UNSEEN_LANGUAGES = ["bg", "fa", "ja", "zh"]
ALL_LANGUAGES = SEEN_LANGUAGES + UNSEEN_LANGUAGES

BASE_TEST_PATH = "data/test_sets"


def main():

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"}
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding_model
    )

    model = MultiSentRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.1"
    )

    for lang in ALL_LANGUAGES:

        print(f"\nRunning MultiSent-RAG on {lang.upper()}")

        path = f"{BASE_TEST_PATH}/test_set_{lang}.csv"

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing file for {lang}")
            continue

        mode = "fewshot" if lang in SEEN_LANGUAGES else "zeroshot"

        predictions = []

        for text in tqdm(df["text"], desc=lang):

            retrieved_docs = vectorstore.similarity_search(
                query=text,
                k=7
            )

            retrieved_docs_text = []
            for i, doc in enumerate(retrieved_docs):
                meta = doc.metadata
                doc_text = (
                    f"Source: {meta.get('source', 'Unknown')}\n"
                    f"Title: {meta.get('title', f'Document {i}')}\n"
                    f"Language: {meta.get('language', 'Unknown')}\n"
                    f"Label: {meta.get('label', 'unknown')}\n"
                    f"Content: {doc.page_content}\n"
                )
                retrieved_docs_text.append(doc_text)

            context = "\nExtracted documents:\n" + "".join(
                [f"Document {i}:::\n{doc}" for i, doc in enumerate(retrieved_docs_text)]
            )

            full_input = f"{context}\n\nText: {text}"

            answer = model.predict([full_input], mode=mode)[0]
            predictions.append(answer)

        df["answer"] = predictions
        df["predicted_label"] = df["answer"].apply(map_answer_to_label)

        metrics = compute_metrics(
            y_true=df["label"],
            y_pred=df["predicted_label"]
        )

        print(metrics)

        df.to_csv(
            f"data/results_multisent_rag_{lang}.csv",
            index=False
        )


if __name__ == "__main__":
    main()
