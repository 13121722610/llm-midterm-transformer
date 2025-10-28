# src/generate.py
import torch
import argparse
from data import load_data
from model import FullTransformer, DecoderOnlyTransformer

def generate_full_transformer(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0):
    """使用完整Transformer生成文本"""
    model.eval()
    device = next(model.parameters()).device
    
    # 编码提示文本
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        # 修复：使用固定的序列长度限制（与训练时一致）
        idx_cond = idx if idx.size(1) <= 64 else idx[:, -64:]  # 硬编码为训练时的64
        
        # 使用相同的输入作为encoder和decoder的输入
        # 注意：这里我们使用完整的序列作为输入
        logits = model(idx_cond, idx_cond)
        logits = logits[:, -1, :] / temperature  # 只取最后一个时间步
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

def main():
    parser = argparse.ArgumentParser(description='Transformer文本生成')
    parser.add_argument('--prompt', type=str, default='To be, or not to be', help='输入提示')
    parser.add_argument('--model', type=str, choices=['full', 'decoder'], default='decoder', help='模型类型')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成温度')
    
    args = parser.parse_args()
    
    tokenizer, _, _ = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == 'full':
        print("使用完整Transformer生成...")
        # 使用训练时的配置（与train.py中的快速测试配置一致）
        model = FullTransformer(
            tokenizer.vocab_size, 
            d_model=64,           # 与训练时一致
            n_layer=2,            # 与训练时一致
            n_head=2,             # 与训练时一致
            d_ff=256,             # 与训练时一致
            max_seq_len=64        # 与训练时一致
        )
        try:
            model.load_state_dict(torch.load("checkpoints/full_transformer_best.pt", map_location=device))
            model.to(device)
            result = generate_full_transformer(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
        except FileNotFoundError:
            print("完整Transformer模型未找到，请先训练模型")
        except Exception as e:
            print(f"模型加载错误: {e}")
    else:
        print("使用Decoder-Only Transformer生成...")
        # 使用训练时的配置（与train.py中的快速测试配置一致）
        model = DecoderOnlyTransformer(
            tokenizer.vocab_size,
            d_model=64,           # 与训练时一致
            n_layer=2,            # 与训练时一致
            n_head=2,             # 与训练时一致
            d_ff=256,             # 与训练时一致
            max_seq_len=64        # 与训练时一致
        )
        try:
            model.load_state_dict(torch.load("checkpoints/decoder_only_best.pt", map_location=device))
            model.to(device)
            result = generate_decoder_only(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature)
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
        except FileNotFoundError:
            print("Decoder-Only模型未找到，请先训练模型")
        except Exception as e:
            print(f"模型加载错误: {e}")

if __name__ == "__main__":
    main()
