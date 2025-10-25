# src/data.py
import os
import requests
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FILE_PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")

def download():
    if not os.path.exists(FILE_PATH):
        print("Downloading tiny shakespeare...")
        r = requests.get(URL)
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(r.text)
        print("Downloaded to", FILE_PATH)
    else:
        print("Data already exists:", FILE_PATH)

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

def load_data():
    download()
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)
    # simple split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return tokenizer, train_data, val_data
