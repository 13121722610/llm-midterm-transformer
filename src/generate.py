# src/generate.py
import torch
import argparse
from data import load_data
from model import DecoderOnlyTransformer

def generate_with_sampling(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=20, top_p=0.9):
    """使用改进采样策略的生成函数"""
    model.eval()
    device = next(model.parameters()).device
    
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        # 序列长度限制
        idx_cond = idx if idx.size(1) <= model.max_seq_len else idx[:, -model.max_seq_len:]
        
        # 获取logits
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
    parser.add_argument('--model', type=str, choices=['full', 'decoder'], default='full', help='模型类型')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成温度 (0.1-1.0)')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k采样')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样 (0.0-1.0)')
    
    args = parser.parse_args()
    
    tokenizer, _, _ = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model == 'full':
        print("使用完整Transformer生成...")
        model = DecoderOnlyTransformer(
            tokenizer.vocab_size, 
            d_model=256,           # 与训练时一致
            n_layer=6,             # 与训练时一致
            n_head=8,              # 与训练时一致
            d_ff=512,              # 与训练时一致
            max_seq_len=128        # 与训练时一致
        )
        try:
            model.load_state_dict(torch.load("checkpoints/full_transformer_improved_best.pt", map_location=device))
            model.to(device)
            result = generate_with_sampling(
                model, tokenizer, args.prompt, 
                args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
            print(f"参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        except FileNotFoundError:
            print("完整Transformer模型未找到，请先训练模型")
            print("尝试使用备用模型文件...")
            try:
                model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
                model.to(device)
                result = generate_with_sampling(
                    model, tokenizer, args.prompt, 
                    args.max_new_tokens, args.temperature, args.top_k, args.top_p
                )
                print(f"输入: {args.prompt}")
                print(f"输出: {result}")
                print(f"参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
            except:
                print("所有模型文件都未找到，请先训练模型")
        except Exception as e:
            print(f"模型加载错误: {e}")
            
    else:
        print("使用Decoder-Only Transformer生成...")
        model = DecoderOnlyTransformer(
            tokenizer.vocab_size,
            d_model=256,           # 与训练时一致
            n_layer=6,             # 与训练时一致
            n_head=8,              # 与训练时一致
            d_ff=512,              # 与训练时一致
            max_seq_len=128        # 与训练时一致
        )
        try:
            model.load_state_dict(torch.load("checkpoints/decoder_only_improved_best.pt", map_location=device))
            model.to(device)
            result = generate_with_sampling(
                model, tokenizer, args.prompt,
                args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
            print(f"输入: {args.prompt}")
            print(f"输出: {result}")
            print(f"参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        except FileNotFoundError:
            print("Decoder-Only模型未找到，请先训练模型")
        except Exception as e:
            print(f"模型加载错误: {e}")

if __name__ == "__main__":
    main()
