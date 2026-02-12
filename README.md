# 🚀 MultiSent-RAG  
### Retrieval & Memory-Augmented Multilingual Sentiment Intelligence

> Research implementation of **MultiSent-RAG**, submitted to *Information Processing & Management (IP&M)*.

---

## 🧠 What This Project Demonstrates

This repository showcases a **full-stack Retrieval-Augmented Generation (RAG) system for multilingual NLP**, combining:

- 🔎 Dense multilingual retrieval (Chroma + MPNet embeddings)
- 🧾 LLM inference with quantized models (Mistral, LLaMA-3, BLOOMZ)
- 🗄 Semantic memory caching (Annoy + angular distance)
- 🌍 Cross-lingual generalization (12 languages, including zero-shot)
- 📊 Reproducible evaluation pipeline

This is not a notebook experiment.  
It is a **modular, production-style RAG architecture** designed for multilingual AI systems.

---

## 🏗 System Architecture

MultiSent-RAG follows a layered AI system design:

```
Input Text
    ↓
Embedding (MPNet)
    ↓
Chroma Vector Retrieval (top-k)
    ↓
LLM Inference (4-bit quantized)
    ↓
Label Mapping
```

With optional semantic memory:

```
Semantic Cache (Annoy index)
→ If high similarity → reuse prediction
→ Otherwise → full RAG pipeline
```

---

## 🌍 Languages Covered

| Training Languages | Zero-Shot Evaluation |
|-------------------|---------------------|
| en, fr, ar, es, de, pt, hi, it | bg, fa, ja, zh |

Zero-shot languages are never indexed in the vector store — evaluation tests true cross-lingual transfer.

---

## 📦 Knowledge Sources

### Structured
- Cardiff NLP Multilingual Tweets
- Massive Multilingual Sentiment (MMS)

### Unstructured
- Wikipedia (emotion & sentiment queries per language)

All sources are embedded using:

```
paraphrase-multilingual-mpnet-base-v2
```

and stored in a persistent **Chroma vector database**.

---

## 🧠 Models Used

**Encoders**
- mBERT
- XLM-R

**LLMs (4-bit quantized)**
- BLOOMZ-7B
- LLaMA-3-8B
- Mistral-7B

LLMs operate in a generative setup constrained to single-word sentiment outputs.

---

## 🗄 Semantic Cache (Memory Layer)

The semantic cache:

- Stores embeddings + predictions
- Uses Annoy with angular distance
- Applies similarity threshold (default: 0.9)
- Skips retrieval + LLM inference when similarity is high

This enables:
- Stability analysis
- Latency vs accuracy trade-offs
- Memory-aware LLM orchestration

---

## 📁 Project Structure

```
src/
├── baselines/        # Encoder & LLM classification models
├── rag/              # Core RAG pipeline
├── memory/           # Semantic cache implementation
├── vectorstore/      # Chroma DB construction
├── evaluation/       # Metrics & evaluation logic
├── scripts/          # Experiment entry points
```

---

## 🚀 How to Run

### 1️⃣ Install

```bash
pip install -r requirements.txt
```

### 2️⃣ Build Wikipedia Knowledge

```bash
python src/scripts/build_wikipedia.py
```

### 3️⃣ Build Vector Database

```bash
python src/vectorstore/build_vectorstore.py
```

### 4️⃣ Run Baselines

```bash
python src/scripts/run_baselines.py
```

### 5️⃣ Run MultiSent-RAG

```bash
python src/scripts/run_rag.py
```

### 6️⃣ Run MultiSent-RAG + Memory

```bash
python src/scripts/run_rag_cache.py
```

---

## 📊 Skills Demonstrated

This repository demonstrates practical experience with:

- Vector databases (Chroma)
- Approximate nearest neighbor search (Annoy)
- Multilingual embeddings
- LLM quantization
- Retrieval-Augmented Generation
- Memory-aware inference
- Cross-lingual evaluation
- Modular AI system design

---

## 🔬 Research Context

Implementation of:

**MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing**  
Submitted to *Information Processing & Management (IP&M)*.
