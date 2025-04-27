#!/usr/bin/env python
# src/evaluate_bert_full.py

import re
import numpy as np
import faiss
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ───────────────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────────────
def precision_at_k(gold_texts, retrieved_texts, k=10):
    return sum(1 for t in retrieved_texts[:k] if t in gold_texts) / k

def mrr_at_k(gold_texts, retrieved_texts, k=10):
    for rank, t in enumerate(retrieved_texts[:k], start=1):
        if t in gold_texts:
            return 1.0 / rank
    return 0.0

# ───────────────────────────────────────────────────────────────────────────────
# Query embedding (BERT)
# ───────────────────────────────────────────────────────────────────────────────
def embed_query_bert(q, tokenizer, model, device='cuda'):
    enc = tokenizer(
        q,
        return_tensors='pt',
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        hid = model(**enc).last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')
    faiss.normalize_L2(hid)
    return hid

# ───────────────────────────────────────────────────────────────────────────────
# Main evaluation: full-corpus BERT-only
# ───────────────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load & flatten MS MARCO train passages
    print("Loading and flattening train passages…")
    train_ds = load_dataset('ms_marco', 'v1.1', split='train')
    corpus_texts = []
    for ex in tqdm(train_ds, desc='Flatten train'):
        corpus_texts.extend(ex['passages']['passage_text'])

    # 2) Load validation queries + gold passages
    print("Loading validation queries and gold labels…")
    val_ds = load_dataset('ms_marco', 'v1.1', split='validation')
    queries    = [ex['query'] for ex in val_ds]
    gold_lists = [
        [t for t, sel in zip(ex['passages']['passage_text'], ex['passages']['is_selected']) if sel]
        for ex in val_ds
    ]

    # 3) Load BERT encoder & FAISS index
    print("Loading BERT model and FAISS index…")
    tokenizer  = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').eval().to(device)
    idx_bert   = faiss.read_index('index/faiss/bert_msmarco.index')

    # 4) Evaluate
    print(f"Evaluating {len(queries)} queries with BERT…")
    p10_b, mrr_b = [], []

    for q, gold in tqdm(zip(queries, gold_lists), total=len(queries), desc='Queries'):
        qb = embed_query_bert(q, tokenizer, bert_model, device)
        _, ib = idx_bert.search(qb, 10)
        retrieved = [corpus_texts[i] for i in ib[0]]
        p10_b.append(precision_at_k(gold, retrieved))
        mrr_b.append(mrr_at_k(gold, retrieved))

    # 5) Report results
    print("\nFull-corpus retrieval (train corpus) results [BERT]:")
    print(f"P@10  = {np.mean(p10_b):.4f}")
    print(f"MRR@10 = {np.mean(mrr_b):.4f}")

if __name__ == '__main__':
    main()
