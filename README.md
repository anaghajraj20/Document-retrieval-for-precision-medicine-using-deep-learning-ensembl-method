# Document Retrieval for Precision Medicine

A full implementation of a multi-stage biomedical information retrieval pipeline based on [Liu et al., JMIR Medical Informatics, 2021 (e28272)](https://medinform.jmir.org/2021/11/e28272). The system retrieves and ranks PubMed abstracts against structured patient queries from the **TREC 2019 Precision Medicine Track**, combining classical BM25 retrieval with deep learning re-ranking.

---

## Overview

Given a patient profile (disease + gene mutation + demographics), the pipeline retrieves the most clinically relevant PubMed abstracts through a two-phase process: BM25-based initial retrieval followed by a neural re-ranking ensemble.

```
Patient Query (disease, gene, demographic)
        │
        ▼
┌─────────────────────────────┐
│   Phase 1: Initial Retrieval│
│  • Text Preprocessing       │
│  • Query Expansion (CTD/NCBI)│
│  • Field-Weighted Boosting  │
│  • BM25Okapi Retrieval      │
└────────────┬────────────────┘
             │ Top-N candidates
             ▼
┌─────────────────────────────┐
│   Phase 2: Re-Ranking       │
│  • BiGRU-Att (treatment     │
│    classification)          │
│  • MatchPyramid (semantic   │
│    relevance matching)      │
│  • Logistic Regression      │
│    Ensemble Combiner        │
└────────────┬────────────────┘
             │
             ▼
      Final Ranked List
```

---

## Pipeline Components

| Stage | Method | Description |
|---|---|---|
| Preprocessing | NLP | Lowercase → NUM replacement → stop-word removal → Porter stemming → entity normalisation |
| Query Expansion | CTD / NCBI Gene | Expands disease and gene terms with synonyms and abbreviations |
| Query Boosting | Field weighting | Disease/gene ×3, treatment keywords ×2, demographics ×1 |
| Initial Retrieval | BM25Okapi | Lexical retrieval over preprocessed PubMed corpus |
| Text Classification | BiGRU + Attention | Classifies documents for treatment-focus relevance |
| Relevance Matching | MatchPyramid | Dot-product interaction matrix + 2D CNN for semantic matching |
| Re-ranking | Logistic Regression | Combines BM25, classification, and matching scores |
| Evaluation | TREC metrics | P@5, P@10, NDCG@5, NDCG@10 + ablation study |

---

## Dataset

The notebook uses the **TREC 2019 Precision Medicine Track** format:

- **Documents:** PubMed-style abstracts (title + abstract)
- **Queries:** Structured patient topics — `{ disease, gene, demographic }`
- **Relevance judgements:** TREC qrels format (`0` = not relevant, `1` = relevant, `2` = highly relevant)

The included corpus simulates 20 realistic PubMed abstracts across melanoma, NSCLC, and breast cancer, with 3 annotated patient queries, faithfully mirroring the paper's data format.

---

## Requirements

```bash
pip install rank_bm25 sentence-transformers nltk scikit-learn torch datasets requests tqdm
```

Also requires NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Python:** 3.8+  
**PyTorch:** 1.10+ (CPU or CUDA)

---

## Usage

Open and run `precision_medicine_retrieval.ipynb` cell by cell. The notebook is self-contained — no external data downloads required.

```bash
jupyter notebook precision_medicine_retrieval.ipynb
```

Cells execute in order:

1. Install dependencies
2. Load corpus, queries, and qrels
3. Preprocess text
4. Expand and boost queries
5. BM25 initial retrieval
6. Train BiGRU-Att classifier
7. Train MatchPyramid re-ranker
8. Train Logistic Regression combiner
9. Run full pipeline
10. Evaluate and ablation study

---

## Results

The ablation study measures the contribution of each component:

| Method | P@5 | NDCG@5 |
|---|---|---|
| BM25 only | baseline | baseline |
| + Query Expansion | ↑ | ↑ |
| + Query Boosting | ↑ | ↑ |
| + Full Ensemble (BiGRU-Att + MatchPyramid) | best | best |

The full ensemble consistently outperforms the BM25 baseline, confirming that deep semantic re-ranking improves retrieval precision beyond lexical matching.

---

## Reference

> Liu et al. (2021). *A Multi-Stage Document Retrieval System for Precision Medicine.* JMIR Medical Informatics, 9(11), e28272. https://doi.org/10.2196/28272

---

## License

This implementation is for research and educational purposes. See the original paper for methodology details.

