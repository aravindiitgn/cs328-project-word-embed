import sys
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

def compute_query_vec_w2v(query: str, model: Word2Vec):
    toks = query.lower().split()
    vecs = [model.wv[w] for w in toks if w in model.wv]
    return np.mean(vecs, axis=0).astype('float32')

def compute_query_vec_bert(query: str, tokenizer, model, device='cuda'):
    inputs = tokenizer(
        query,
        return_tensors='pt',
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        out = model(**inputs)
    vec = out.last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')
    faiss.normalize_L2(vec)
    return vec

def main():
    if len(sys.argv) != 3:
        print("Usage: python query.py [w2v|bert] \"your query\"")
        sys.exit(1)

    mode, query = sys.argv[1], sys.argv[2]

    if mode == 'w2v':
        w2v = Word2Vec.load('embeddings/word2vec/msmarco_w2v.model')
        vec = compute_query_vec_w2v(query, w2v).reshape(1, -1)
        faiss.normalize_L2(vec)
        index = faiss.read_index('index/faiss/w2v_msmarco.index')
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased').eval().to('cuda')
        vec = compute_query_vec_bert(query, tokenizer, model).reshape(1, -1)
        index = faiss.read_index('index/faiss/bert_msmarco.index')

    distances, ids = index.search(vec, 10)
    print("Top‑10 doc IDs:", ids[0])

if __name__ == '__main__':
    main()
