# MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing

Official implementation of:

**MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing**  
Khouloud Mnassri, Reza Farahbakhsh, Noel Crespi  
*Information Processing & Management* — Accepted, 2026  
📄 DOI: [10.1016/j.ipm.2026.104990](https://doi.org/10.1016/j.ipm.2026.104990)

---

## Overview

Multilingual sentiment analysis remains challenging due to limited labeled data and strong cross-lingual variation. We introduce:

- **MultiSent-RAG**: a training-free retrieval-augmented framework that integrates structured sentiment corpora with unstructured multilingual evidence
- **MultiSent-RAG-Cache**: extends MultiSent-RAG with a semantic memory module that reuses prior label inferences based on embedding similarity

Our study uses 80,000 labeled instances across 12 languages, including low-resource and zero-shot settings. Retrieval augmentation yields F1 gains of up to +0.65 points, with relative gains exceeding 70–110% over strong LLM baselines.

---

## System Architecture

![MultiSent-RAG Architecture](assets/figure_1.jpg)

MultiSent-RAG follows a three-stage pipeline:
Input Text

↓

Embedding (paraphrase-multilingual-mpnet-base-v2)

↓

Chroma Vector Retrieval (top-k = 7)

↓

LLM Inference (4-bit quantized, prompt-based)

↓

Label Mapping → Final Sentiment Prediction


With optional semantic cache memory (MultiSent-RAG-Cache):

![MultiSent-RAG-Cache Workflow](assets/figure_2.jpg)
Semantic Cache (Annoy index, angular distance)

→ Cache hit (distance ≤ threshold) → return cached prediction

→ Cache miss → full RAG pipeline → store new prediction in cache

---

## Languages Covered

| Seen Languages (Few-Shot) | Unseen Languages (Zero-Shot) |
|--------------------------|------------------------------|
| en, fr, ar, es, de, pt, hi, it | bg, fa, ja, zh |

Zero-shot languages are never indexed in the vector store — evaluation tests true cross-lingual transfer.

---

## Models

### Encoder Baselines
- `bert-base-multilingual-cased`
- `xlm-roberta-base`

### LLM Baselines (4-bit Quantized)
- `bigscience/bloomz-7b1`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`

### MultiSent-RAG & MultiSent-RAG-Cache
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.1`

---

## Knowledge Sources

**Structured:**
- Cardiff NLP Tweet Sentiment Multilingual (8 languages)
- Massive Multilingual Sentiment — MMS (10k samples/language, 80k total)

**Unstructured:**
- Wikipedia articles in 8 languages (up to 100 documents/language)

All sources embedded with `paraphrase-multilingual-mpnet-base-v2` and stored in a persistent Chroma vector database.

---

## Results (Weighted F1)

| Model | Avg F1 (12 Languages) |
|-------|----------------------|
| mBERT | 0.459 |
| LLaMA-3 (baseline) | 0.516 |
| Mistral (baseline) | 0.472 |
| MultiSent-RAG (LLaMA-3) | 0.783 |
| MultiSent-RAG (Mistral) | 0.768 |
| MultiSent-RAG-Cache (LLaMA-3) | 0.760 |
| MultiSent-RAG-Cache (Mistral) | 0.748 |

---

## Project Structure
MultiSent-RAG/

├── assets/

│   ├── figure_1.jpg               # MultiSent-RAG architecture

│   └── figure_2.jpg               # MultiSent-RAG-Cache workflow

├── scripts/

│   ├── build_wikipedia.py         # Build Wikipedia knowledge base

│   ├── build_vectorstore.py       # Build Chroma vector database

│   ├── run_baselines.py           # Run encoder and LLM baselines

│   ├── run_multisent_rag.py       # Run MultiSent-RAG (all 12 languages)

│   └── run_multisent_rag_cache.py # Run MultiSent-RAG with semantic cache

├── src/

│   ├── baselines/

│   │   ├── encoder.py             # Encoder-based classifier

│   │   └── llm_classifier.py      # LLM-based classifier

│   ├── data/

│   │   ├── data_loader.py         # Load and split MMS test data

│   │   └── wikipedia_loader.py    # Load Wikipedia knowledge

│   ├── evaluation/

│   │   ├── baseline_evaluator.py  # Evaluate baselines

│   │   ├── rag_evaluator.py       # Evaluate RAG outputs

│   │   └── metrics.py             # Compute accuracy, F1

│   ├── memory/

│   │   └── semantic_cache.py      # Semantic cache (core contribution)

│   └── pipeline/

│       └── multisent_rag.py       # MultiSent-RAG core pipeline

├── .gitignore

├── requirements.txt

└── README.md

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- CUDA-compatible GPU recommended. Change `device` to `"cpu"` in scripts if unavailable.
- For gated models (LLaMA 3): `huggingface-cli login`

---

## How to Run

### 1. Build Wikipedia knowledge base
```bash
python scripts/build_wikipedia.py
```

### 2. Build Chroma vector database
```bash
python scripts/build_vectorstore.py
```

### 3. Run baselines
```bash
python scripts/run_baselines.py
```
To switch models, edit `model_name` in the script (options are commented in the file).

### 4. Run MultiSent-RAG (all 12 languages)
```bash
python scripts/run_multisent_rag.py
```

### 5. Run MultiSent-RAG with Semantic Cache
```bash
python scripts/run_multisent_rag_cache.py
```
> **Note:** Runs on English by default. Change `TEST_PATH` for other languages (e.g., `test_set_fr.csv`, `test_set_ar.csv`).

---

## Citation

```bibtex
@article{mnassri2026multisentrag,
  title     = {MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing},
  author    = {Mnassri, Khouloud and Farahbakhsh, Reza and Crespi, Noel},
  journal   = {Information Processing \& Management},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.ipm.2026.104990}
}
```

---

## Contact

**Khouloud Mnassri**  
Samovar, Télécom SudParis, Institut Polytechnique de Paris, 91120 Palaiseau, France  
📧 khouloud.mnassri@telecom-sudparis.eu


## License

This repository will be released under an open license upon publication confirmation. Please contact the authors for reuse permissions in the meantime.