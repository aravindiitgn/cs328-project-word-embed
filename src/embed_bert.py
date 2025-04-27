import os, re
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def embed_batch(texts, tokenizer, model, device):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        out = model(**enc)
    hid = out.last_hidden_state              # (B, T, D)
    mask = enc['attention_mask'].unsqueeze(-1)
    vsum = (hid * mask).sum(dim=1)
    cnt = mask.sum(dim=1)
    return (vsum / cnt).cpu().numpy()         # (B, D)

def main(batch_size=32):
    device = 'cuda'
    os.makedirs('embeddings/bert', exist_ok=True)

    # load the dataset of queries+passages
    ds = load_dataset('ms_marco', 'v1.1', split='train')

    # flatten all passages into a single list
    all_texts = []
    for ex in tqdm(ds, desc='Collect texts'):
        ps = ex['passages']
        if isinstance(ps, dict):
            texts = ps['passage_text']
        else:
            texts = [p['passage_text'] for p in ps]
        all_texts.extend(texts)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model     = AutoModel.from_pretrained('bert-base-uncased').eval().to(device)

    # now embed in batches
    vectors = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc='BERT embedding'):
        batch = all_texts[i:i+batch_size]
        clean = [clean_text(t) for t in batch]
        vecs  = embed_batch(clean, tokenizer, model, device)
        vectors.append(vecs)

    arr = np.vstack(vectors)
    np.save('embeddings/bert/msmarco_bert.npy', arr)
    print("Saved BERT embeddings:", arr.shape)

if __name__ == '__main__':
    main()
