from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from typing import Dict


def init_cache():
    index = AnnoyIndex(768, "angular")
    encoder = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return index, encoder


def retrieve_cache(json_file: str) -> Dict:
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"embeddings": [], "responses": []}
    return cache


def store_cache(json_file: str, cache: Dict):
    with open(json_file, "w") as file:
        json.dump(cache, file)


class SemanticCache:
    """
    MultiSent-RAG Semantic Cache (Memory Layer)

    - Uses Annoy for fast approximate nearest neighbor search
    - Falls back to vector database + LLM if no close match found
    """

    def __init__(
        self,
        json_file: str,
        threshold: float,
        knowledge_vector_database,
        rag_prompt_template: str,
        reader_llm,
    ):
        self.index, self.encoder = init_cache()
        self.threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(json_file)
        self.knowledge_vector_database = knowledge_vector_database
        self.rag_prompt_template = rag_prompt_template
        self.reader_llm = reader_llm
        self.index_built = False

        # Load existing embeddings
        for i, emb in enumerate(self.cache["embeddings"]):
            self.index.add_item(i, emb)

        if self.cache["embeddings"]:
            self.index.build(10)
            self.index_built = True

    def ask(self, query: str) -> Dict:

        embedding = self.encoder.encode([query])[0]
        embedding = embedding / np.linalg.norm(embedding)

        # 🔎 Check cache
        if self.index_built and len(self.cache["embeddings"]) > 0:
            idx, dist = self.index.get_nns_by_vector(
                embedding, 1, include_distances=True
            )

            if dist[0] <= self.threshold:
                row_id = int(idx[0])
                return {
                    "answer": self.cache["responses"][row_id],
                    "source": "cache",
                }

        # 🔁 Otherwise: RAG retrieval
        retrieved_docs = self.knowledge_vector_database.similarity_search(
            query=query, k=7
        )

        context = "\n".join([doc.page_content for doc in retrieved_docs])

        final_prompt = self.rag_prompt_template.format(
            question=query,
            context=context,
        )

        answer = self.reader_llm(final_prompt)[0]["generated_text"]

        # Store in cache
        new_id = len(self.cache["responses"])
        self.cache["embeddings"].append(embedding.tolist())
        self.cache["responses"].append(answer)

        self.index.add_item(new_id, embedding)

        if not self.index_built:
            self.index.build(10)
            self.index_built = True

        store_cache(self.json_file, self.cache)

        return {"answer": answer, "source": "vector_db"}
