# src/generate.py
import torch
from data import load_data
from model import DecoderOnlyTransformer

def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())

if __name__ == "__main__":
    tokenizer, _, _ = load_data()
    model = DecoderOnlyTransformer(tokenizer.vocab_size, d_model=128, n_layer=4, n_head=4, d_ff=512, max_seq_len=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    prompt = "To be, or not to be:"
    out = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0)
    print(out)
