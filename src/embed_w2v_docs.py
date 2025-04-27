import os
import numpy as np
from gensim.models import Word2Vec

def main():
    os.makedirs('embeddings/word2vec', exist_ok=True)
    model = Word2Vec.load('embeddings/word2vec/msmarco_w2v.model')

    vectors = []
    with open('data/processed/msmarco_train.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            vecs = [model.wv[t] for t in tokens if t in model.wv]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                # fallback to zeros if no known token
                vectors.append(np.zeros(model.vector_size))
    arr = np.vstack(vectors).astype('float32')
    np.save('embeddings/word2vec/msmarco_w2v_doc.npy', arr)
    print("Saved W2V doc embeddings:", arr.shape)

if __name__ == '__main__':
    main()
