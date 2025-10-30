# src/generate.py
import torch
import argparse
from data import load_data
from model import DecoderOnlyTransformer, FullTransformer

def generate_with_sampling(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=20, top_p=0.9, max_seq_len=256):
    """使用改进采样策略的生成函数"""
    model.eval()
    device = next(model.parameters()).device
    
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        # 序列长度限制 - 使用传入的max_seq_len参数
        idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
        
        # 获取logits - 根据模型类型调整
        if isinstance(model, FullTransformer):
            # FullTransformer需要encoder输入，这里简单重复输入作为encoder输入
            logits = model(idx_cond, idx_cond)
        else:
            # DecoderOnlyTransformer
            logits = model(idx_cond)
            
        logits = logits[:, -1, :] / temperature
        
        # Top-k 采样
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Top-p (核采样)
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        probs = torch.softmax(logits, dim=-1)
        
        # 如果所有概率都是0（被屏蔽了），回退到随机采样
        if torch.all(probs == 0):
            probs = torch.ones_like(probs) / probs.size(-1)
            
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    
    return tokenizer.decode(idx[0].tolist())

def main():
    parser = argparse.ArgumentParser(description='改进的Transformer文本生成')
    parser.add_argument('--prompt', type=str, default='To be, or not to be', help='输入提示')
    parser.add_argument('--model', type=str, choices=['full', 'decoder'], default='decoder', help='模型类型')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成温度 (0.1-1.0)')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样 (0.0-1.0)')
    
    args = parser.parse_args()
    
    tokenizer, _, _ = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 统一的模型配置 - 与训练时完全一致
    model_config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': 128,           # 与训练一致
        'n_layer': 4,             # 与训练一致  
        'n_head': 4,              # 与训练一致
        'd_ff': 512,             # 与训练一致
        'max_seq_len': 128,       # 与训练一致
        'dropout': 0.1
    }
    
    if args.model == 'full':
        print("使用完整Transformer生成...")
        # 使用FullTransformer类
        model = FullTransformer(**model_config)
        model_files = [
            "checkpoints/full_transformer_improved_best.pt",
            "checkpoints/full_transformer_best.pt",
            "checkpoints/best.pt"
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                model.load_state_dict(torch.load(model_file, map_location=device))
                model.to(device)
                print(f"成功加载完整Transformer模型: {model_file}")
                model_loaded = True
                break
            except FileNotFoundError:
                print(f"未找到模型文件: {model_file}")
                continue
            except Exception as e:
                print(f"加载 {model_file} 失败: {e}")
                continue
        
        if model_loaded:
            # 为完整Transformer设置专门的生成参数
            full_temp = 1.2  # 更高的温度增加多样性
            full_top_k = 10  # 更严格的top-k
            full_top_p = 0.8 # 更严格的top-p
            
            result = generate_with_sampling(
                model, tokenizer, args.prompt, 
                args.max_new_tokens, full_temp, full_top_k, full_top_p,
                max_seq_len=model_config['max_seq_len']
            )
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
            print(f"参数: temperature={full_temp}, top_k={full_top_k}, top_p={full_top_p} (为完整Transformer优化)")
        else:
            print("所有完整Transformer模型文件都未找到或加载失败，请先训练模型")
            
    else:
        print("使用Decoder-Only Transformer生成...")
        model = DecoderOnlyTransformer(**model_config)
        model_files = [
            "checkpoints/decoder_only_improved_best.pt",
            "checkpoints/decoder_only_best.pt",
            "checkpoints/best.pt"
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                model.load_state_dict(torch.load(model_file, map_location=device))
                model.to(device)
                print(f"成功加载Decoder-Only模型: {model_file}")
                model_loaded = True
                break
            except FileNotFoundError:
                print(f"未找到模型文件: {model_file}")
                continue
            except Exception as e:
                print(f"加载 {model_file} 失败: {e}")
                continue
        
        if model_loaded:
            # 保持原有的Decoder-Only参数
            result = generate_with_sampling(
                model, tokenizer, args.prompt,
                args.max_new_tokens, args.temperature, args.top_k, args.top_p,
                max_seq_len=model.max_seq_len
            )
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
            print(f"参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        else:
            print("所有Decoder-Only模型文件都未找到或加载失败，请先训练模型")

if __name__ == "__main__":
    main()
