# 🚀 MultiSent-RAG  
### Retrieval & Memory-Augmented Multilingual Sentiment Intelligence


---

## 🧠 What This Project Demonstrates

This repository showcases a **full-stack Retrieval-Augmented Generation (RAG) system for training-free multilingual NLP**, combining:

- 🔎 Vector Store + Semantic Search (Google ChromaDB + multilingual MPNet embeddings)
- 🧠 Retrieval-Augmented Generation (RAG) 
- 🗄 Memory-Augmented Inference via Semantic Cache (Spotify Annoy, ANN search)
- 🌍 Cross-Lingual & Zero-Shot Generalization (over 12 languages)


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
- Wikipedia (sentiment analysis knowledge per language)

All sources are embedded using:

```
paraphrase-multilingual-mpnet-base-v2
```

and stored in a persistent **Chroma vector database**.

---

## 🧠 Models

### Encoder Baselines 
- `bert-base-multilingual-cased`
- `xlm-roberta-base`

Used as sequence classification models without any fine-tuning.

---

### LLM Baselines (4-bit Quantized)
- `bigscience/bloomz-7b1`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B`

LLMs are treated as sequence classifiers.

---

### MultiSent-RAG & MultiSent-RAG-Cache

RAG experiments use instruction-tuned models only:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct`

These models operate in a generative inference setup (prompt-based), without supervised fine-tuning.

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
├── baselines/
├── core/
├── data/
├── evaluation/
├── memory/
├── rag/
├── scripts/

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


## 🔬 Context

Implementation of:

**MultiSent-RAG: A Retrieval and Memory-Augmented System for Multilingual Sentiment Processing**  
Submitted to *Information Processing & Management (IP&M)*.
