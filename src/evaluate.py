import re
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

# -------- text cleaning (same for both) --------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# -------- Precision@10 helper --------
def precision_at_10(is_selected, topk_idxs):
    # (# relevant in top‑k) / 10
    return sum(is_selected[i] for i in topk_idxs) / 10.0

# -------- Word2Vec embed helper --------
def get_w2v_vector(text: str, w2v_model: Word2Vec) -> np.ndarray:
    tokens = clean_text(text).split()
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(w2v_model.vector_size, dtype=float)

# -------- main comparison loop --------
def main(sample_size: int = 100):
    device = 'cuda'
    # load validation split
    ds = load_dataset('ms_marco', 'v1.1', split='validation')

    # load BERT
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')\
                          .eval().to(device)

    # load Word2Vec
    w2v_model = Word2Vec.load('embeddings/word2vec/msmarco_w2v.model')

    bert_scores = []
    w2v_scores = []

    for ex in tqdm(ds.select(range(sample_size)), desc='Eval both'):
        texts     = ex['passages']['passage_text']
        is_sel    = ex['passages']['is_selected']  # 0/1 list

        # --- BERT path on GPU ---
        # encode query
        q_enc = tokenizer(
            ex['query'],
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
        with torch.no_grad():
            q_hid = bert_model(**q_enc).last_hidden_state    # (1, T, D)
        q_vec = torch.nn.functional.normalize(q_hid.mean(dim=1), p=2, dim=1)  # (1, D)

        # encode passages
        clean_texts = [clean_text(t) for t in texts]
        p_enc = tokenizer(
            clean_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            p_hid = bert_model(**p_enc).last_hidden_state  # (P, T, D)
        p_vec = torch.nn.functional.normalize(p_hid.mean(dim=1), p=2, dim=1)  # (P, D)

        # cosine sims & top‑k
        sims = (p_vec @ q_vec.T).squeeze(1)         # (P,)
        P = sims.size(0)
        k = min(10, P)
        topk_bert = torch.topk(sims, k=k).indices.cpu().numpy()
        bert_scores.append(precision_at_10(is_sel, topk_bert))

        # --- Word2Vec path on CPU ---
        # embed query & docs
        q_w2v = get_w2v_vector(ex['query'], w2v_model)
        # normalize
        if np.linalg.norm(q_w2v) > 0:
            q_w2v /= np.linalg.norm(q_w2v)
        doc_w2v = []
        for t in texts:
            vec = get_w2v_vector(t, w2v_model)
            if np.linalg.norm(vec) > 0:
                vec /= np.linalg.norm(vec)
            doc_w2v.append(vec)
        doc_w2v = np.stack(doc_w2v, axis=0)        # (P, D)

        # cosine sims & top‑k
        sims_w = doc_w2v @ q_w2v         # (P,)
        P = sims_w.shape[0]
        k = min(10, P)
        topk_w2v = np.argsort(-sims_w)[:k]  # descending
        w2v_scores.append(precision_at_10(is_sel, topk_w2v))

    print(f"\nMean P@10 over {sample_size} queries:")
    print(f"  BERT:  {np.mean(bert_scores):.4f}")
    print(f"  W2V:   {np.mean(w2v_scores):.4f}")

if __name__ == '__main__':
    main(sample_size=100)