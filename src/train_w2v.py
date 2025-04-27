from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import os

def main():
    os.makedirs('embeddings/word2vec', exist_ok=True)
    sentences = LineSentence('data/processed/msmarco_train.txt')
    model = Word2Vec(
        sentences,
        vector_size=300,
        window=5,
        min_count=5,
        workers=4,
        epochs=5
    )
    model.save('embeddings/word2vec/msmarco_w2v.model')

if __name__ == '__main__':
    main()
