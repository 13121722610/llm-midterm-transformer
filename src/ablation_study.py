# src/ablation_study.py
import os
import time
import math
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import load_data
from model import DecoderOnlyTransformer

class CharDataset:
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def ensure_directories():
    """确保所有输出目录存在"""
    directories = [
        "checkpoints",
        "results/figures", 
        "results/tables",
        "results/logs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_pretrained_weights(model, pretrained_path="checkpoints/decoder_only_improved_best.pt"):
    """智能加载预训练权重，处理层数不匹配的情况"""
    try:
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        model_state = model.state_dict()
        
        transferred_params = 0
        total_params = 0
        
        for name, param in pretrained_state.items():
            if name in model_state:
                if param.shape == model_state[name].shape:
                    model_state[name] = param
                    transferred_params += param.numel()
                else:
                    # 层数不匹配，尝试智能匹配
                    if 'layers' in name:
                        layer_num = int(name.split('.')[1])
                        if layer_num < len(model.layers):
                            model_state[name] = param
                            transferred_params += param.numel()
            total_params += param.numel()
        
        model.load_state_dict(model_state)
        transfer_rate = transferred_params / total_params * 100
        print(f"✅ 成功加载预训练权重: {transfer_rate:.1f}% 参数")
        return True
    except Exception as e:
        print(f"❌ 加载预训练权重失败: {e}")
        return False

def quick_evaluate_model(model, val_loader, criterion, device):
    """快速评估模型"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_val_loss += loss.item()
    
    return total_val_loss / len(val_loader)

def train_ablation_model(model, tokenizer, config, model_name, device):
    """训练单个消融实验模型"""
    # 准备数据加载器
    _, train_data, val_data = load_data()
    train_dataset = CharDataset(train_data, config['block_size'])
    val_dataset = CharDataset(val_data, config['block_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练记录
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"开始训练 {model_name}...")
    
    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            xb, yb = xb.to(device), yb.to(device)
            
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        avg_val_loss = quick_evaluate_model(model, val_loader, criterion, device)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val PPL: {math.exp(avg_val_loss):.2f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pt")
    
    return {
        'model_name': model_name,
        'layers': config['n_layer'],
        'best_val_loss': best_val_loss,
        'best_val_ppl': math.exp(best_val_loss),
        'final_val_loss': val_losses[-1],
        'final_val_ppl': math.exp(val_losses[-1]),
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_epochs': config['epochs']
    }

def plot_ablation_results(results, original_result=None):
    """绘制消融实验结果图"""
    layers = [r['layers'] for r in results]
    val_ppls = [r['best_val_ppl'] for r in results]
    parameters = [r['parameters'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 困惑度对比
    bars = ax1.bar([str(l) for l in layers], val_ppls, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_xlabel('Transformer层数')
    ax1.set_ylabel('验证困惑度 (PPL)')
    ax1.set_title('不同层数架构的验证困惑度对比\n(消融实验)')
    ax1.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bar, ppl in zip(bars, val_ppls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 如果有原始6层5轮的结果，添加参考线
    if original_result:
        ax1.axhline(y=original_result['best_val_ppl'], color='red', linestyle='--', 
                   label=f"原始6层5轮: {original_result['best_val_ppl']:.2f}")
        ax1.legend()
    
    # 参数量对比
    ax2.bar([str(l) for l in layers], [p/1e6 for p in parameters], 
            color='orange', alpha=0.7)
    ax2.set_xlabel('Transformer层数')
    ax2.set_ylabel('参数量 (百万)')
    ax2.set_title('不同层数架构的参数量对比')
    ax2.grid(True, alpha=0.3)
    
    # 在柱状图上显示参数量
    for i, (layer, param) in enumerate(zip(layers, parameters)):
        ax2.text(i, param/1e6 + 0.1, f'{param/1e6:.1f}M', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/layer_ablation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 消融实验对比图已生成: results/figures/layer_ablation_comparison.png")

def load_original_results():
    """加载原始6层5轮模型的训练结果"""
    try:
        # 从原始结果文件加载
        original_df = pd.read_csv("results/tables/decoder_only_improved_final_results.csv")
        original_result = {
            'model_name': 'decoder_only_improved_6L5E',
            'layers': 6,
            'best_val_loss': original_df['best_val_loss'].iloc[0],
            'best_val_ppl': original_df['best_val_ppl'].iloc[0],
            'final_val_loss': original_df['final_val_loss'].iloc[0],
            'final_val_ppl': original_df['final_val_ppl'].iloc[0],
            'parameters': original_df['parameters'].iloc[0],
            'train_epochs': original_df['total_epochs'].iloc[0],
            'note': '原始6层5轮完整训练'
        }
        print("✅ 成功加载原始6层5轮模型结果")
        return original_result
    except Exception as e:
        print(f"❌ 加载原始结果失败: {e}")
        return None

def layer_ablation_study():
    """层数消融实验主函数"""
    ensure_directories()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("🚀 开始层数消融实验...")
    print("📝 实验目标: 比较不同层数对模型性能的影响")
    print("⏱️  每个配置训练3个epoch进行快速对比\n")
    
    # 加载原始6层5轮模型结果
    original_result = load_original_results()
    if original_result:
        print(f"📊 原始6层5轮模型: 困惑度 {original_result['best_val_ppl']:.2f}, 训练轮次 {original_result['train_epochs']}")
    
    # 加载数据
    tokenizer, _, _ = load_data()
    
    # 测试不同的层数配置
    layer_configs = [2, 4, 6]  # 2层(浅) vs 4层(中) vs 6层(深)
    results = []
    
    for n_layer in layer_configs:
        print(f"\n{'='*60}")
        print(f"🧪 消融实验: {n_layer} 层 Decoder-Only Transformer")
        print(f"{'='*60}")
        
        # 创建模型配置
        config = {
            'd_model': 256,
            'n_layer': n_layer,
            'n_head': 8,
            'd_ff': 1024,
            'block_size': 256,
            'batch_size': 32,
            'epochs': 3,  # 快速训练3个epoch
            'lr': 1e-4,
            'dropout': 0.1,
            'weight_decay': 0.01
        }
        
        # 创建模型
        model = DecoderOnlyTransformer(
            tokenizer.vocab_size,
            d_model=config['d_model'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            d_ff=config['d_ff'],
            max_seq_len=config['block_size'],
            dropout=config['dropout']
        ).to(device)
        
        # 尝试加载预训练权重
        load_success = load_pretrained_weights(model)
        
        # 训练模型
        model_name = f"ablation_layers_{n_layer}"
        result = train_ablation_model(model, tokenizer, config, model_name, device)
        results.append(result)
        
        print(f"✅ {n_layer}层模型完成 - 最佳验证困惑度: {result['best_val_ppl']:.2f}")
    
    # 保存消融实验结果
    ablation_df = pd.DataFrame(results)
    ablation_file = "results/tables/layer_ablation_results.csv"
    ablation_df.to_csv(ablation_file, index=False)
    print(f"\n🎯 层数消融实验完成! 结果保存至: {ablation_file}")
    
    # 生成消融实验对比图
    plot_ablation_results(results, original_result)
    
    # 打印总结
    print(f"\n📋 消融实验总结:")
    print(f"{'层数':<8} {'困惑度':<10} {'参数量':<12} {'训练轮次'}")
    print(f"{'-'*40}")
    for result in results:
        print(f"{result['layers']:<8} {result['best_val_ppl']:<10.2f} {result['parameters']/1e6:<12.1f}M {result['train_epochs']:<10}")
    
    if original_result:
        print(f"\n📊 原始6层5轮模型参考:")
        print(f"层数: 6, 困惑度: {original_result['best_val_ppl']:.2f}, 参数量: {original_result['parameters']/1e6:.1f}M, 训练轮次: {original_result['train_epochs']}")
    
    return results

if __name__ == "__main__":
    layer_ablation_study()
