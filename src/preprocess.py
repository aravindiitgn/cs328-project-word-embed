import os
import re
from datasets import load_dataset
from tqdm import tqdm

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    os.makedirs('data/processed', exist_ok=True)
    ds = load_dataset('ms_marco', 'v1.1', split='train')

    out_path = 'data/processed/msmarco_train.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        for example in tqdm(ds, desc='Cleaning passages'):
            passages = example['passages']
            # In v1.1, passages is a dict-of-lists
            texts = passages['passage_text']
            for txt in texts:
                if not txt:
                    continue
                f.write(clean_text(txt) + '\n')

if __name__ == '__main__':
    main()
