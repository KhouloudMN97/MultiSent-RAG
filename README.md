# MultiSent-RAG  
A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing

Implementation of the paper:

**MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing**  
Submitted to *Information Processing & Management (IP&M)*.

---

## 🔍 Overview

MultiSent-RAG is a multilingual sentiment analysis framework that integrates:

- Encoder-based multilingual baselines (mBERT, XLM-R)
- Retrieval-Augmented Generation (RAG)
- Semantic memory caching
- Evaluation across 12 languages

This repository currently includes the encoder baseline implementation and evaluation pipeline.

---

## 📁 Project Structure

```
src/
  baselines/     # Encoder-based multilingual models
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

We evaluate encoder-based baselines on the **Massive Multilingual Sentiment (MMS)** dataset:

HuggingFace: https://huggingface.co/datasets/Brand24/mms

The MMS corpus contains over 6 million sentiment-labeled instances across multiple languages.

---

### 🌍 Language Selection

We evaluate on the following 12 languages:

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

## 🚀 Reproducibility

Install dependencies:

```
pip install -r requirements.txt
```

Run encoder baselines:

```
python scripts/run_baselines.py
```

---

## 🧠 Models

- bert-base-multilingual-cased  
- xlm-roberta-base  

Model selection can be modified inside `scripts/run_baselines.py`.

---

## 📌 Notes

- The RAG and memory modules are structured for integration within the MultiSent-RAG framework.
- For full experimental details, please refer to the paper.
