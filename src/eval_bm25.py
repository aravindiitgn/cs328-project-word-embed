#!/usr/bin/env python
# src/evaluate_bm25.py

import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi  # pip install rank_bm25

# 1) Text cleaning

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# 2) Metrics

def precision_at_k(is_selected, preds, k=10):
    return sum(is_selected[i] for i in preds[:k]) / k

def mrr_at_k(is_selected, preds, k=10):
    for rank, idx in enumerate(preds[:k], start=1):
        if is_selected[idx]:
            return 1.0 / rank
    return 0.0

# 3) Main evaluation

def main(sample_size: int = 100):
    ds = load_dataset('ms_marco', 'v1.1', split='validation')

    p10_bm25, mrr_bm25 = [], []

    n = sample_size or len(ds)
    for i in tqdm(range(n), desc='Evaluating BM25', unit='qry'):
        ex     = ds[i]
        query  = ex['query']
        texts  = ex['passages']['passage_text']
        is_sel = ex['passages']['is_selected']

        # — BM25 retrieval —
        doc_tokens = [clean_text(t).split() for t in texts]
        bm25_local = BM25Okapi(doc_tokens)
        q_tokens   = clean_text(query).split()
        scores     = bm25_local.get_scores(q_tokens)
        topk_b     = np.argsort(-scores)[:10]

        p10_bm25.append(precision_at_k(is_sel, topk_b))
        mrr_bm25.append(mrr_at_k(is_sel, topk_b))

    # Report results
    print(f"\nBM25 Results over {n} queries:")
    print(f"P@10: {np.mean(p10_bm25):.4f}")
    print(f"MRR@10: {np.mean(mrr_bm25):.4f}")

if __name__ == '__main__':
    main(sample_size=10000)  # or set sample_size as you want
