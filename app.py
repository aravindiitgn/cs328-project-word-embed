# app.py

import os
import re
import numpy as np
import faiss
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

# ── 1) Page config must come first ─────────────────────────────────────────────
st.set_page_config(
    page_title="MS MARCO Semantic Search",
    layout="wide"
)

# ── 2) Text cleaning & embedding utilities ────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_w2v_query(query: str, w2v_model: Word2Vec):
    tokens = clean_text(query).split()
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not vecs:
        out = np.zeros((1, w2v_model.vector_size), dtype='float32')
    else:
        out = np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True).astype('float32')
        faiss.normalize_L2(out)
    return out

def compute_bert_query(query: str, tokenizer, model, device='cuda'):
    enc = tokenizer(
        query,
        return_tensors='pt',
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        hid = model(**enc).last_hidden_state  # (1, T, D)
    vec = hid.mean(dim=1).cpu().numpy().astype('float32')  # (1, D)
    faiss.normalize_L2(vec)
    return vec

# ── 3) Caching with st.cache_data / st.cache_resource ──────────────────────────
@st.cache_data
def load_passages():
    path = os.path.join('data', 'processed', 'msmarco_train.txt')
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

@st.cache_resource
def load_w2v_model():
    return Word2Vec.load(os.path.join('embeddings', 'word2vec', 'msmarco_w2v.model'))

@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model     = AutoModel.from_pretrained('bert-base-uncased')\
                         .eval().to('cuda')
    return tokenizer, model

@st.cache_resource
def load_faiss_index(index_name: str):
    path = os.path.join('index', 'faiss', index_name)
    return faiss.read_index(path)

# ── 4) Streamlit UI ────────────────────────────────────────────────────────────
st.title("MS MARCO Semantic Search Demo")
st.markdown("Select BERT or Word2Vec, enter a query, and retrieve top‑k passages.")

method = st.selectbox("Embedding Method", ["BERT", "Word2Vec"])
query  = st.text_input("Your query here")
top_k  = st.slider("Number of results (k)", 1, 20, 10)

if st.button("Search") and query:
    passages = load_passages()

    if method == "BERT":
        tokenizer, bert_model = load_bert_model()
        idx = load_faiss_index('bert_msmarco.index')
        q_vec = compute_bert_query(query, tokenizer, bert_model, device='cuda')
    else:
        w2v_model = load_w2v_model()
        idx = load_faiss_index('w2v_msmarco.index')
        q_vec = compute_w2v_query(query, w2v_model)

    # CPU‐based FAISS search
    distances, indices = idx.search(q_vec, top_k)

    st.write(f"### Top {top_k} Results using **{method}**")
    for rank, (score, pid) in enumerate(zip(distances[0], indices[0]), start=1):
        st.markdown(f"**{rank}. (score: {score:.4f})**")
        st.write(passages[pid])
        st.markdown("---")
