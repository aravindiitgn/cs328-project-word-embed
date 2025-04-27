#!/usr/bin/env python
# src/evaluate_rerank.py

import re
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi  # pip install rank_bm25

# 1) Text cleaning & embedding helpers

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def embed_bert(texts, tokenizer, model, device='cuda'):
    enc = tokenizer(
        texts, padding=True, truncation=True,
        max_length=256, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        hid = model(**enc).last_hidden_state  # (B, T, D)
    mask   = enc['attention_mask'].unsqueeze(-1)  # (B, T, 1)
    summed = (hid * mask).sum(dim=1)              # (B, D)
    counts = mask.sum(dim=1)                      # (B, 1)
    vecs   = (summed / counts).cpu().numpy()      # (B, D)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-8, None)

def embed_w2v(texts, w2v_model):
    vecs = []
    D    = w2v_model.vector_size
    for t in texts:
        toks = clean_text(t).split()
        wv   = [w2v_model.wv[w] for w in toks if w in w2v_model.wv]
        if wv:
            v = np.stack(wv, 0).mean(axis=0)
        else:
            v = np.zeros(D, dtype='float32')
        norm = np.linalg.norm(v)
        vecs.append(v / (norm if norm > 0 else 1.0))
    return np.stack(vecs, axis=0)

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds     = load_dataset('ms_marco', 'v1.1', split='validation')

    # Load models once
    print("Loading BERT…")
    tokenizer  = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')\
                           .eval().to(device)

    print("Loading Word2Vec…")
    w2v_model = Word2Vec.load('embeddings/word2vec/msmarco_w2v.model')

    # Prepare metric accumulators
    p10_bert, mrr_bert = [], []
    p10_w2v,  mrr_w2v  = [], []
    p10_bm25, mrr_bm25 = [], []

    n = sample_size or len(ds)
    for i in tqdm(range(n), desc='Evaluating', unit='qry'):
        ex     = ds[i]
        query  = ex['query']
        texts  = ex['passages']['passage_text']
        is_sel = ex['passages']['is_selected']

        # — BERT retrieval —
        q_vec   = embed_bert([query], tokenizer, bert_model, device)
        d_vecs  = embed_bert(texts, tokenizer, bert_model, device)
        sims    = (d_vecs @ q_vec.T).squeeze(1)
        topk    = np.argsort(-sims)[:10]
        p10_bert.append(precision_at_k(is_sel, topk))
        mrr_bert.append(mrr_at_k(is_sel, topk))

        # — Word2Vec retrieval —
        q_w2v   = embed_w2v([query], w2v_model)
        d_w2v   = embed_w2v(texts, w2v_model)
        sims_w  = (d_w2v @ q_w2v.T).squeeze(1)
        topk_w  = np.argsort(-sims_w)[:10]
        p10_w2v.append(precision_at_k(is_sel, topk_w))
        mrr_w2v.append(mrr_at_k(is_sel, topk_w))

        # — BM25 baseline retrieval —
        doc_tokens = [clean_text(t).split() for t in texts]
        bm25_local = BM25Okapi(doc_tokens)
        q_tokens   = clean_text(query).split()
        scores     = bm25_local.get_scores(q_tokens)
        topk_b     = np.argsort(-scores)[:10]
        p10_bm25.append(precision_at_k(is_sel, topk_b))
        mrr_bm25.append(mrr_at_k(is_sel, topk_b))

    # Report results
    print(f"\nResults over {n} queries:")
    print(f"BERT   → P@10: {np.mean(p10_bert):.4f}, MRR@10: {np.mean(mrr_bert):.4f}")
    print(f"W2V    → P@10: {np.mean(p10_w2v):.4f}, MRR@10: {np.mean(mrr_w2v):.4f}")
    print(f"BM25   → P@10: {np.mean(p10_bm25):.4f}, MRR@10: {np.mean(mrr_bm25):.4f}")

if __name__ == '__main__':
    # Evaluate on up to 10k queries (max 10047 in MS MARCO v1.1 validation)
    main(sample_size=10000)
