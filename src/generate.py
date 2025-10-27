# src/generate.py
import torch
from data import load_data
from model import FullTransformer, DecoderOnlyTransformer

def generate_full_transformer(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    """使用完整Transformer生成文本"""
    model.eval()
    device = next(model.parameters()).device
    
    # 编码提示文本
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        # 确保输入长度不超过最大序列长度
        idx_cond = idx if idx.size(1) <= model.encoder.max_seq_len else idx[:, -model.encoder.max_seq_len:]
        
        # 使用相同的输入作为encoder和decoder的输入
        logits = model(idx_cond, idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    
    return tokenizer.decode(idx[0].tolist())

def generate_decoder_only(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    """使用Decoder-Only Transformer生成文本"""
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试完整Transformer
    print("=== 完整Transformer生成 ===")
    full_model = FullTransformer(tokenizer.vocab_size)
    full_model.to(device)
    try:
        full_model.load_state_dict(torch.load("checkpoints/full_transformer_best.pt", map_location=device))
        prompt = "To be, or not to be:"
        out = generate_full_transformer(full_model, tokenizer, prompt, max_new_tokens=100, temperature=0.8)
        print(f"输入: {prompt}")
        print(f"输出: {out}\n")
    except FileNotFoundError:
        print("完整Transformer模型未找到，请先训练模型")
    
    # 测试Decoder-Only Transformer
    print("=== Decoder-Only Transformer生成 ===")
    decoder_model = DecoderOnlyTransformer(tokenizer.vocab_size)
    decoder_model.to(device)
    try:
        decoder_model.load_state_dict(torch.load("checkpoints/decoder_only_best.pt", map_location=device))
        prompt = "To be, or not to be:"
        out = generate_decoder_only(decoder_model, tokenizer, prompt, max_new_tokens=100, temperature=0.8)
        print(f"输入: {prompt}")
        print(f"输出: {out}")
    except FileNotFoundError:
        print("Decoder-Only模型未找到，请先训练模型")
