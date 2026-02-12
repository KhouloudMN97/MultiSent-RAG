## Dataset

### Massive Multilingual Sentiment Corpus (MMS)

We evaluate encoder-based baselines on the **Massive Multilingual Sentiment (MMS)** dataset:

HuggingFace: https://huggingface.co/datasets/Brand24/mms

The MMS corpus contains over 6 million sentiment-labeled instances across multiple languages :contentReference[oaicite:1]{index=1}.

---

### Language Selection

From the full multilingual corpus, we select the following 12 languages:

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

### Sampling Strategy

For evaluation, we sample:

- **1,000 test instances per language**
- Binary labels only (negative and positive)

This results in a balanced multilingual test benchmark of 12,000 samples.

If sufficient computational resources are available, the full dataset can be used instead of the 1k-per-language subset.

---

### Data Preparation Steps

To reproduce our setup:

1. Download the MMS dataset from HuggingFace:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("Brand24/mms")
