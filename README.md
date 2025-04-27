# MS MARCO Semantic Search

A demo semantic-search pipeline over the MS MARCO corpus, comparing static Word2Vec vs. contextual BERT embeddings with FAISS for approximate nearest-neighbor retrieval.

Includes scripts for:
- Preprocessing MS MARCO passages
- Training a Word2Vec model
- Generating BERT embeddings
- Building FAISS indices
- Querying from CLI or Streamlit
- Evaluating P@10 and MRR@10

---

## 📂 Repository Structure

```
word-embed-search/
├── data/
│   └── processed/         # Cleaned MS MARCO passages (one per line)
├── embeddings/
│   ├── word2vec/          # Word2Vec .model & doc vectors
│   └── bert/              # BERT .npy embeddings
├── index/
│   └── faiss/             # FAISS index files (.index)
├── src/                   # Source scripts
│   ├── preprocess.py      # Clean & prepare MS MARCO
│   ├── train_w2v.py       # Train Word2Vec model
│   ├── embed_bert.py      # Generate BERT embeddings
│   ├── embed_w2v_docs.py  # Generate Word2Vec doc vectors
│   ├── build_index.py     # Build FAISS index for BERT
│   ├── build_w2v_index.py # Build FAISS index for Word2Vec
│   ├── query.py           # CLI querying tool
│   ├── evaluate_rerank.py # Evaluation (rerank pool)
│   └── evaluate_full.py   # Evaluation (full corpus)
├── app.py                 # Streamlit demo
├── README.md              # This file
└── .gitignore
```

---

## ⚙️ Requirements

- Python 3.8+  
- (Optional) CUDA 10.2+ for GPU acceleration  

```bash
pip install   gensim   transformers   torch torchvision torchaudio   faiss-cpu   scikit-learn   numpy   pandas   datasets   tqdm   streamlit
```

For GPU support, replace `faiss-cpu` with `faiss-gpu` and install the CUDA-enabled PyTorch wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Setup & Pipeline

1. **Preprocess MS MARCO Passages**  
   ```bash
   python src/preprocess.py
   ```  
   - Loads the MS MARCO v1.1 train split  
   - Cleans & lowercases each passage  
   - Writes `data/processed/msmarco_train.txt`

2. **Train Word2Vec**  
   ```bash
   python src/train_w2v.py
   ```  
   - Trains a Word2Vec model on `msmarco_train.txt`  
   - Saves `embeddings/word2vec/msmarco_w2v.model`

3. **Generate BERT Embeddings**  
   ```bash
   python src/embed_bert.py
   ```  
   - Mean-pools a `bert-base-uncased` model over each passage  
   - Saves `embeddings/bert/msmarco_bert.npy`

4. **Generate Word2Vec Document Vectors**  
   ```bash
   python src/embed_w2v_docs.py
   ```  
   - Averages Word2Vec token vectors per passage  
   - Saves `embeddings/word2vec/msmarco_w2v_doc.npy`

5. **Build FAISS Indices**  
   - **BERT index:**  
     ```bash
     python src/build_index.py
     ```  
     Outputs `index/faiss/bert_msmarco.index`  
   - **Word2Vec index:**  
     ```bash
     python src/build_w2v_index.py
     ```  
     Outputs `index/faiss/w2v_msmarco.index`

---

## 🔍 Querying

### CLI

```bash
# Using BERT
python src/query.py bert "what is artificial intelligence"

# Using Word2Vec
python src/query.py w2v "how to bake banana bread"
```

### Streamlit Demo

```bash
streamlit run app.py
```

1. Select **BERT** or **Word2Vec**  
2. Enter a natural-language query & `top-k`  
3. View retrieved passages with similarity scores  

---

## 📊 Evaluation

- **Reranking Built-in Pool**  
  ```bash
  python src/evaluate.py
  ```  
  Measures P@10 & MRR@10 over the MS MARCO validation candidate list

- **Full-corpus Retrieval**  
  ```bash
  python src/evaluate_full.py
  ```  
  Retrieves from all ~8 M passages via FAISS

Typical results:
- **BERT** → Precision@10 ~0.28, MRR@10 ~0.30  
- **W2V** → Precision@10 ~0.10, MRR@10 ~0.12  

---

## 📝 Notes

- Data & indexes are not checked in; they are added to `.gitignore`.  
- For versioned binaries, consider Git LFS.  
- Out-of-the-box BERT is not fine-tuned; for higher accuracy, fine-tune on MS MARCO pairs or use a Sentence-Transformer.

---

## 📚 References

- [MS MARCO dataset (v1.1)](https://microsoft.github.io/msmarco/)  
- [FAISS](https://github.com/facebookresearch/faiss)  
- [HuggingFace Transformers](https://github.com/huggingface/transformers)  
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
