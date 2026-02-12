import os
import pandas as pd
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


RETRIEVAL_LANGUAGES = ['en', 'fr', 'ar', 'es', 'de', 'pt', 'hi', 'it']
EMBEDDING_TOKENIZER = "xlm-roberta-base"
VECTOR_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def load_wikipedia(path: str) -> List[Document]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text"])
    df = df[df["language"].isin(RETRIEVAL_LANGUAGES)]

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "language": row["language"],
                    "source": "Wikipedia",
                    "title": row.get("title", ""),
                    "description": row.get("description", "")
                }
            )
        )

    return documents


def load_structured_data(path: str, source_name: str) -> List[Document]:
    df = pd.read_csv(path)
    df = df[df["language"].isin(RETRIEVAL_LANGUAGES)].copy()

    # Ensure binary labels (0 = negative, 1 = positive)
    if "label" in df.columns:
        df["label"] = df["label"].map({0: 0, 2: 1}).fillna(df["label"])

    documents = []
    for _, row in df.iterrows():
        documents.append(
            Document(
                page_content=row["text"],
                metadata={
                    "language": row["language"],
                    "label": row.get("label"),
                    "source": source_name
                }
            )
        )

    return documents


def split_documents(knowledge_base: List[Document]) -> List[Document]:

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=768,
        chunk_overlap=int(768 / 10),
        add_start_index=True,
        strip_whitespace=True
    )

    processed_docs = []
    for doc in knowledge_base:
        processed_docs.extend(text_splitter.split_documents([doc]))

    # Remove duplicates
    seen_texts = set()
    unique_docs = []

    for doc in processed_docs:
        if doc.page_content not in seen_texts:
            seen_texts.add(doc.page_content)
            unique_docs.append(doc)

    return unique_docs


def build_vectorstore(
    wikipedia_path: str,
    mms_path: str,
    cardiff_path: str,
    persist_directory: str
):

    print("Loading Wikipedia...")
    wiki_docs = load_wikipedia(wikipedia_path)

    print("Loading MMS...")
    mms_docs = load_structured_data(mms_path, source_name="MMS")

    print("Loading Cardiff...")
    cardiff_docs = load_structured_data(cardiff_path, source_name="Cardiff")

    knowledge_base = wiki_docs + mms_docs + cardiff_docs

    print(f"Total raw documents: {len(knowledge_base)}")

    docs_processed = split_documents(knowledge_base)

    print(f"Total processed chunks: {len(docs_processed)}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=VECTOR_EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"}
    )

    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma.from_documents(
        docs_processed,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    vectorstore.persist()

    print("Vector store successfully built and saved.")
