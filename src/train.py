# src/train.py
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data import load_data
from model import DecoderOnlyTransformer

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def train():
    # hyperparams
    d_model = 128
    n_layer = 4
    n_head = 4
    d_ff = 512
    block_size = 128
    batch_size = 32
    epochs = 10
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, train_data, val_data = load_data()
    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DecoderOnlyTransformer(tokenizer.vocab_size, d_model=d_model, n_layer=n_layer, n_head=n_head, d_ff=d_ff, max_seq_len=block_size)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 1e9
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)  # (B, T, V)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        ppl = math.exp(val_loss)
        print(f"\nEpoch {epoch} val_loss={val_loss:.4f} ppl={ppl:.2f}")
        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"checkpoints/best.pt")
            print("Saved best model.")
    print("Training complete.")

if __name__ == "__main__":
    train()
