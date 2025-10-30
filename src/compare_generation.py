# compare_generation.py
import torch
import argparse
from data import load_data
from model import DecoderOnlyTransformer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=20, top_p=0.9):
    """生成文本函数"""
    model.eval()
    device = next(model.parameters()).device
    
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        
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
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
        
        probs = torch.softmax(logits, dim=-1)
        
        if torch.all(probs == 0):
            probs = torch.ones_like(probs) / probs.size(-1)
            
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    
    return tokenizer.decode(idx[0].tolist())

def compare_models_generation():
    """对比不同层数模型的生成效果"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _, _ = load_data()
    
    # 不同层数的模型配置
    model_configs = [
        {'layers': 2, 'path': 'checkpoints/ablation_layers_2_best.pt', 'name': '2层模型'},
        {'layers': 4, 'path': 'checkpoints/ablation_layers_4_best.pt', 'name': '4层模型'},
        {'layers': 6, 'path': 'checkpoints/ablation_layers_6_best.pt', 'name': '6层模型(3轮)'},
        {'layers': 6, 'path': 'checkpoints/decoder_only_improved_best.pt', 'name': '6层模型(5轮)'}
    ]
    
    # 测试不同的prompt
    test_prompts = [
        "To be, or not to be",
        "Once upon a time",
        "The future of AI",
        "Love is",
        "In the beginning"
    ]
    
    results = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"🧪 测试: {config['name']}")
        print(f"{'='*60}")
        
        try:
            # 创建模型
            model = DecoderOnlyTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=256,
                n_layer=config['layers'],
                n_head=8,
                d_ff=1024,
                max_seq_len=256,
                dropout=0.1
            ).to(device)
            
            # 加载权重
            model.load_state_dict(torch.load(config['path'], map_location=device))
            print(f"✅ 成功加载模型: {config['path']}")
            
            # 对每个prompt生成文本
            for prompt in test_prompts:
                print(f"\n📝 Prompt: \"{prompt}\"")
                generated = generate_text(model, tokenizer, prompt, max_new_tokens=30)
                print(f"📤 生成: {generated}")
                
                results.append({
                    'model': config['name'],
                    'layers': config['layers'],
                    'prompt': prompt,
                    'generated_text': generated
                })
            
            # 释放模型内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 加载模型失败: {config['path']} - {e}")
    
    # 保存生成结果
    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/tables/generation_comparison.csv', index=False)
    print(f"\n✅ 生成对比结果保存至: results/tables/generation_comparison.csv")
    
    return results

if __name__ == "__main__":
    compare_models_generation()
