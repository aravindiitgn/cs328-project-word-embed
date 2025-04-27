import os
import numpy as np
import faiss

def build(fpath: str, index_path: str, nlist: int = 512):
    vectors = np.load(fpath).astype('float32')
    faiss.normalize_L2(vectors)
    d = vectors.shape[1]

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vectors)
    index.add(vectors)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

if __name__ == '__main__':
    # Build BERT index
    build(
        'embeddings/bert/msmarco_bert.npy',
        'index/faiss/bert_msmarco.index'
    )
