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

---

## 📁 Project Structure

```
src/
  baselines/         # Encoder and LLM-based classification models
  core/              # Data loading and evaluation logic
  data/              # Data loading utilities
  evaluation/        # Metrics + label mapping
  rag/               # MultiSent-RAG reader model
  scripts/
      build_wikipedia.py
      run_baselines.py
      run_rag.py
  vectorstore/
      build_vectorstore.py

tests/
README.md
requirements.txt
```

---

## 📊 Datasets

### Structured Data

We use:

- **Cardiff NLP Tweet Sentiment Multilingual**
  - 8 languages: `en, fr, ar, es, de, pt, hi, it`
  - Used as retrieval knowledge source

- **Massive Multilingual Sentiment (MMS)**
  - Train subset: 10k samples per language (8 languages → 80k total)
  - Test subset: 1k samples per language
  - Zero-shot languages: `bg, fa, ja, zh` (evaluation only)

---

### Unstructured Data – Wikipedia

For the same 8 retrieval languages:

Queries:
- Positive emotions
- Negative emotions
- Sentiment expression

Up to 100 documents per language are extracted and truncated to ~1000 characters.

These documents are embedded and stored in a Chroma vector database.

---

## 🧠 Models Evaluated

- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `bigscience/bloomz-7b1`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`

LLM-based models are loaded using 4-bit quantization (nf4) for efficient inference.

---

# 🗄️ Vector Database Construction (Chroma)

MultiSent-RAG relies on a persistent Chroma vector database that combines:

- Wikipedia (unstructured)
- MMS (train subset)
- Cardiff NLP dataset

Zero-shot languages are **excluded** from the vectorstore.

---

# 🚀 Execution Order (Reproducibility Guide)

To fully reproduce the pipeline, run the following steps **in order**:

---

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Build Wikipedia Knowledge Base

```bash
python src/scripts/build_wikipedia.py
```

This extracts multilingual Wikipedia documents and saves:

```
data/wikipedia.csv
```

---

### 3️⃣ Build Chroma Vector Database

```bash
python src/vectorstore/build_vectorstore.py
```

This:

- Loads Wikipedia
- Loads MMS (train subset)
- Loads Cardiff dataset
- Splits documents into chunks (768 tokens)
- Embeds using `paraphrase-multilingual-mpnet-base-v2`
- Saves persistent Chroma database

---

### 4️⃣ Run Baseline Models

```bash
python src/scripts/run_baselines.py
```

This evaluates:

- mBERT
- XLM-R
- Quantized LLM classifiers

---

### 5️⃣ Run MultiSent-RAG

```bash
python src/scripts/run_rag.py
```

This performs:

- Top-k retrieval (k=7) from Chroma
- Context injection into prompt
- Few-shot evaluation on 8 seen languages
- Zero-shot evaluation on 4 unseen languages
- Sentiment prediction
- Metric computation (Accuracy, Precision, Recall, F1)

Results are saved per language.

---

## 🔬 MultiSent-RAG Design

For each input text:

1. Retrieve top-k semantically similar documents
2. Inject retrieved context into the prompt
3. Use instruction-tuned LLM (Mistral or LLaMA-3)
4. Generate one-word sentiment prediction
5. Map to numeric label
6. Compute evaluation metrics

---

## 📌 Notes

- LLMs are used in a generative inference setup but constrained to single-word classification.
- Retrieval is performed using multilingual MPNet embeddings.
- All code is modular and organized for reproducibility and clarity.
- Memory-augmented variant is implemented separately.

---

## 📎 Citation

If you use this repository, please cite:

*MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing*  
Information Processing & Management (Under Review)

