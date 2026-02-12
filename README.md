# MultiSent-RAG  
A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing

Implementation of the paper:

**MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing**  
Submitted to *Information Processing & Management (IP&M)*.

---

## 🔍 Overview

MultiSent-RAG is a multilingual sentiment analysis framework that integrates:

- Encoder-based multilingual baselines (mBERT, XLM-R)
- Decoder-based LLM baselines used in a non-generative classification setting (BLOOMZ, LLaMA-3, Mistral)
- Retrieval-Augmented Generation (RAG)
- Semantic memory caching
- Evaluation across 12 languages

In this repository, LLMs are used in a **non-generative setting**, i.e., as sequence classification models rather than text generators.

The current implementation includes the multilingual baseline evaluation pipeline.

---

## 📁 Project Structure

```
src/
  baselines/     # Encoder and LLM-based classification models
  core/          # Data loading and evaluation logic
  evaluation/    # Metrics computation
  rag/           # Retrieval module (RAG)
  memory/        # Semantic memory cache

scripts/
  run_baselines.py

tests/
```

---

## 📊 Dataset

### Massive Multilingual Sentiment Corpus (MMS)

We evaluate baselines on the **Massive Multilingual Sentiment (MMS)** dataset:

HuggingFace: https://huggingface.co/datasets/Brand24/mms

The MMS corpus contains over 6 million sentiment-labeled instances across multiple languages.

---

### 🌍 Language Selection

Evaluation is conducted on the following 12 languages:

- ar (Arabic)
- en (English)
- fr (French)
- es (Spanish)
- de (German)
- hi (Hindi)
- pt (Portuguese)
- it (Italian)
- bg (Bulgarian)
- fa (Persian)
- ja (Japanese)
- zh (Chinese)

---

### ⚖ Sampling Strategy

For evaluation:

- 1,000 test instances per language
- Binary labels only (negative and positive)

Total benchmark size: 12,000 samples.

---

## 🧠 Models Evaluated

The following pretrained models are evaluated:

- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `bigscience/bloomz-7b1`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`

LLM-based models are loaded using 4-bit quantization (nf4) for efficient inference, as described in the paper.

---
---

## 🗄️ Vector Database Construction (Chroma)

MultiSent-RAG relies on a persistent Chroma vector database that stores multilingual retrieval knowledge.

The vector database combines:

### 🔹 Structured Data (Retrieval Knowledge)

- **Cardiff NLP Tweet Sentiment Multilingual**
  - 8 languages: `en, fr, ar, es, de, pt, hi, it`
  - Full corpus
  - Used for retrieval only

- **MMS (train subset)**
  - Same 8 languages
  - 10,000 samples per language (80,000 total)
  - Used for retrieval only

Zero-shot languages (`bg`, `fa`, `ja`, `zh`) are **excluded** from the vector database and used only for evaluation.

---

### 🔹 Unstructured Data (Wikipedia)

For the same 8 retrieval languages:

- Queries:
  - Positive emotions
  - Negative emotions
  - Sentiment expression
- Up to 100 documents per language
- Each document truncated to ~1000 characters

---

## 🧱 How to Build the Vector Database

The vector database must be built before running retrieval-based models.


### 📦 Build Retrieval Database (Chroma)

The retrieval database must be created before running MultiSent-RAG.

Run the following commands **in order**:

# 1️⃣ Build Wikipedia knowledge base
python src/scripts/build_wikipedia.py

# 2️⃣ Build Chroma vector database
python src/scripts/build_vectorstore.py


---

## 🚀 Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Run baseline evaluation:

```
python scripts/run_baselines.py
```

Model selection can be modified inside `scripts/run_baselines.py`.
