#!/usr/bin/env python
# src/evaluate_full_hf.py

import re
import numpy as np
import faiss
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

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
# Query embedding
# ───────────────────────────────────────────────────────────────────────────────
def embed_query_bert(q, tokenizer, model, device='cuda'):
    enc = tokenizer(q,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256).to(device)
    with torch.no_grad():
        hid = model(**enc).last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')
    faiss.normalize_L2(hid)
    return hid

def embed_query_w2v(q, w2v_model):
    tokens = re.sub(r'[^a-z0-9\s]', ' ', q.lower()).split()
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        out = np.zeros((1, w2v_model.vector_size), dtype='float32')
    else:
        out = np.mean(vecs, axis=0, keepdims=True).astype('float32')
        faiss.normalize_L2(out)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# Main evaluation: full‑corpus = ALL train‑split passages
# ───────────────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Load & flatten MS MARCO train passages
    print("Loading and flattening train passages…")
    train_ds = load_dataset('ms_marco', 'v1.1', split='train')
    corpus_texts = []
    for ex in tqdm(train_ds, desc='Flatten train'):
        texts = ex['passages']['passage_text']
        corpus_texts.extend(texts)
    # corpus_texts[i] corresponds to index i in our FAISS index

    # 2) Load validation queries + gold passages
    print("Loading validation queries and gold labels…")
    val_ds = load_dataset('ms_marco', 'v1.1', split='validation')
    queries = [ex['query'] for ex in val_ds]
    gold_lists = []
    for ex in val_ds:
        texts = ex['passages']['passage_text']
        sels  = ex['passages']['is_selected']
        gold_lists.append([t for t, sel in zip(texts, sels) if sel == 1])

    # 3) Load encoders & FAISS indices
    print("Loading models and FAISS indices…")
    tokenizer  = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').eval().to(device)
    w2v_model  = Word2Vec.load('embeddings/word2vec/msmarco_w2v.model')
    idx_bert   = faiss.read_index('index/faiss/bert_msmarco.index')
    idx_w2v    = faiss.read_index('index/faiss/w2v_msmarco.index')

    # 4) Evaluate
    print(f"Evaluating {len(queries)} queries…")
    p10_b, mrr_b = [], []
    p10_w, mrr_w  = [], []

    for q, gold in tqdm(zip(queries, gold_lists), total=len(queries), desc='Queries'):
        # BERT
        qb = embed_query_bert(q, tokenizer, bert_model, device)
        _, ib = idx_bert.search(qb, 10)
        retrieved_b = [corpus_texts[i] for i in ib[0]]
        p10_b.append(precision_at_k(gold, retrieved_b))
        mrr_b.append(mrr_at_k(gold, retrieved_b))

        # Word2Vec
        qw = embed_query_w2v(q, w2v_model)
        _, iw = idx_w2v.search(qw, 10)
        retrieved_w = [corpus_texts[i] for i in iw[0]]
        p10_w.append(precision_at_k(gold, retrieved_w))
        mrr_w.append(mrr_at_k(gold, retrieved_w))

    # 5) Report results
    print("\nFull‑corpus retrieval (train corpus) results:")
    print(f"BERT →  P@10 = {np.mean(p10_b):.4f},  MRR@10 = {np.mean(mrr_b):.4f}")
    print(f"W2V  →  P@10 = {np.mean(p10_w):.4f},  MRR@10 = {np.mean(mrr_w):.4f}")

if __name__ == '__main__':
    main()
