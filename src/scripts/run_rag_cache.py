import pandas as pd
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.rag.multisent_rag import MultiSentRAG
from src.memory.semantic_cache import SemanticCache
from src.evaluation.metrics import evaluate_predictions


VECTOR_PATH = "data/chroma_db"
TEST_PATH = "data/test_sets/test_set_en.csv"


def main():

    # 🔹 Load Vector Store
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    vector_db = Chroma(
        persist_directory=VECTOR_PATH,
        embedding_function=embedding_model,
    )

    # 🔹 Load Reader Model
    reader = MultiSentRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.1"
        # or model_name=""meta-llama/Meta-Llama-3-8B-Instruct""
    )

    # 🔹 Initialize Semantic Cache
    cache = SemanticCache(
        json_file="cache_file.json",
        threshold=0.7,
        knowledge_vector_database=vector_db,
        rag_prompt_template=reader.build_fewshot_prompt("{question}"),
        reader_llm=reader.generator,
    )

    # 🔹 Load Test Data
    df = pd.read_csv(TEST_PATH)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        response = cache.ask(row["text"])

        results.append({
            "text": row["text"],
            "label": row["label"],
            "response": response["answer"],
            "source": response["source"],
        })

    results_df = pd.DataFrame(results)

    metrics = evaluate_predictions(results_df)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    results_df.to_csv("data/results_rag_cache.csv", index=False)


if __name__ == "__main__":
    main()
