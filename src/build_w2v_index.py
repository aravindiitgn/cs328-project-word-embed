import os
import numpy as np
import faiss

def build_w2v_index():
    vecs = np.load('embeddings/word2vec/msmarco_w2v_doc.npy')
    faiss.normalize_L2(vecs)
    d = vecs.shape[1]

    quant = faiss.IndexFlatIP(d)
    ivf   = faiss.IndexIVFFlat(quant, d, 512, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vecs)
    ivf.add(vecs)

    os.makedirs('index/faiss', exist_ok=True)
    faiss.write_index(ivf, 'index/faiss/w2v_msmarco.index')
    print("Built W2V FAISS index at index/faiss/w2v_msmarco.index")

if __name__ == '__main__':
    build_w2v_index()
