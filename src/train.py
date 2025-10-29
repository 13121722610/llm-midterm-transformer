# src/train.py
import os
import time
import math
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from data import load_data
from model import FullTransformer, DecoderOnlyTransformer

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

def save_training_curve(train_losses, val_losses, train_ppls, val_ppls, filename="training_curve.png"):
    """保存训练曲线图"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_ppls, label='Training PPL', color='blue', linestyle='--')
    plt.plot(val_ppls, label='Validation PPL', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.title('Training and Validation Perplexity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join("results/figures", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存到: {filepath}")

def save_results_table(results_dict, filename="results.csv"):
    """保存结果表格"""
    filepath = os.path.join("results/tables", filename)
    df = pd.DataFrame([results_dict])
    df.to_csv(filepath, index=False)
    print(f"结果表格已保存到: {filepath}")

def save_training_log(epoch, train_loss, val_loss, train_ppl, val_ppl, filename="training_log.json"):
    """保存训练日志"""
    filepath = os.path.join("results/logs", filename)
    
    log_entry = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_ppl': train_ppl,
        'val_ppl': val_ppl,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 读取现有日志或创建新日志
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(log_entry)
    
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=2)

def get_model_config(model_type='full'):
    """根据模型类型返回合适的配置 - 解决过拟合"""
    if model_type == 'full':
        # 完整Transformer：使用更小的配置防止过拟合
        return {
            'd_model': 128,        # 减小维度
            'n_layer': 4,          # 减少层数
            'n_head': 4,           # 减少头数
            'd_ff': 512,           # 减小前馈网络
            'block_size': 128,     # 减小序列长度
            'batch_size': 32,
            'epochs': 10,          # 更多轮次但使用早停
            'lr': 1e-4,
            'dropout': 0.3,        # 增加dropout
            'weight_decay': 0.1,   # 权重衰减
            'patience': 3          # 早停耐心值
        }
    else:
        # Decoder-Only：保持较大配置
        return {
            'd_model': 256,
            'n_layer': 6,
            'n_head': 8,
            'd_ff': 1024,
            'block_size': 256,
            'batch_size': 32,
            'epochs': 5,
            'lr': 1e-4,
            'dropout': 0.1,
            'weight_decay': 0.01,
            'patience': 5
        }

def train_transformer(model_type='full'):
    """训练Transformer模型 - 修复过拟合版本"""
    ensure_directories()
    
    # 获取针对性的配置
    config = get_model_config(model_type)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"训练模型类型: {model_type}")
    print(f"优化配置: d_model={config['d_model']}, n_layer={config['n_layer']}, "
          f"n_head={config['n_head']}, dropout={config['dropout']}, "
          f"weight_decay={config['weight_decay']}")

    # 加载数据
    tokenizer, train_data, val_data = load_data()
    train_dataset = CharDataset(train_data, config['block_size'])
    val_dataset = CharDataset(val_data, config['block_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # 初始化模型
    if model_type == 'full':
        model = FullTransformer(
            tokenizer.vocab_size, 
            d_model=config['d_model'], 
            n_layer=config['n_layer'], 
            n_head=config['n_head'], 
            d_ff=config['d_ff'], 
            max_seq_len=config['block_size'],
            dropout=config['dropout']
        )
        model_name = "full_transformer_fixed"
    else:
        model = DecoderOnlyTransformer(
            tokenizer.vocab_size,
            d_model=config['d_model'], 
            n_layer=config['n_layer'], 
            n_head=config['n_head'], 
            d_ff=config['d_ff'], 
            max_seq_len=config['block_size'],
            dropout=config['dropout']
        )
        model_name = "decoder_only_improved"
    
    model = model.to(device)
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.98)
    )
    
    # 学习率调度器 - 使用ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True,
        min_lr=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    best_val_loss = float('inf')
    no_improve_count = 0

    print(f"开始训练 {model_name}...")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            
            if model_type == 'full':
                logits = model(xb[:, :-1], xb[:, :-1])  # (B, T-1, V)
                loss = criterion(logits.view(-1, logits.size(-1)), yb[:, :-1].contiguous().view(-1))
            else:
                logits = model(xb)  # (B, T, V)
                loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix(loss=total_train_loss / (pbar.n + 1))

        avg_train_loss = total_train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)
        train_losses.append(avg_train_loss)
        train_ppls.append(train_ppl)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                if model_type == 'full':
                    logits = model(xb[:, :-1], xb[:, :-1])
                    loss = criterion(logits.view(-1, logits.size(-1)), yb[:, :-1].contiguous().view(-1))
                else:
                    logits = model(xb)
                    loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
                
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)
        val_losses.append(avg_val_loss)
        val_ppls.append(val_ppl)

        # 更新学习率
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: "
              f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
              f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存日志
        save_training_log(epoch+1, avg_train_loss, avg_val_loss, train_ppl, val_ppl, f"{model_name}_log.json")

        # 早停机制和保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pt")
            print(f"保存最佳模型: {model_name}_best.pt")
        else:
            no_improve_count += 1
            print(f"验证损失未改善 {no_improve_count}/{config['patience']}")

        # 早停检查
        if no_improve_count >= config['patience']:
            print(f"早停触发！在第 {epoch+1} 轮停止训练")
            break

        # 每2个epoch保存一次训练曲线
        if (epoch + 1) % 2 == 0:
            save_training_curve(
                train_losses, val_losses, train_ppls, val_ppls, 
                f"{model_name}_training_curve_epoch_{epoch+1}.png"
            )

    # 最终保存训练曲线和结果
    save_training_curve(train_losses, val_losses, train_ppls, val_ppls, f"{model_name}_final_training_curve.png")
    
    # 保存最终结果表格
    final_results = {
        'model_type': model_name,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_ppl': train_ppls[-1],
        'final_val_ppl': val_ppls[-1],
        'best_val_loss': best_val_loss,
        'best_val_ppl': math.exp(best_val_loss),
        'total_epochs': epoch + 1,
        'parameters': sum(p.numel() for p in model.parameters()),
        'config': str(config),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    save_results_table(final_results, f"{model_name}_final_results.csv")
    
    print(f"\n{model_name} 训练完成!")
    print(f"实际训练轮次: {epoch + 1}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证困惑度: {math.exp(best_val_loss):.2f}")
    
    return final_results

def ablation_study():
    """进行消融实验 - 修复过拟合版本"""
    ensure_directories()
    
    print("开始修复过拟合的消融实验...")
    results = []
    
    # 训练完整Transformer（使用修复配置）
    print("\n1. 训练完整Transformer（修复过拟合）...")
    full_results = train_transformer('full')
    results.append(full_results)
    
    # 训练Decoder-Only Transformer
    print("\n2. 训练Decoder-Only Transformer...")
    decoder_results = train_transformer('decoder_only')
    results.append(decoder_results)
    
    # 保存消融实验结果
    ablation_df = pd.DataFrame(results)
    ablation_file = "results/tables/ablation_study_fixed.csv"
    ablation_df.to_csv(ablation_file, index=False)
    print(f"消融实验结果保存到: {ablation_file}")
    
    # 创建消融实验对比图
    plt.figure(figsize=(10, 6))
    
    try:
        # 读取训练日志
        full_logs = json.load(open("results/logs/full_transformer_fixed_log.json"))
        decoder_logs = json.load(open("results/logs/decoder_only_improved_log.json"))
        
        full_val_ppl = [log['val_ppl'] for log in full_logs]
        decoder_val_ppl = [log['val_ppl'] for log in decoder_logs]
        
        plt.plot(full_val_ppl, label='Full Transformer (Fixed)', linewidth=2, marker='o')
        plt.plot(decoder_val_ppl, label='Decoder-Only', linewidth=2, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Perplexity')
        plt.title('Fixed Overfitting: Model Architecture Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        ablation_plot = "results/figures/ablation_comparison_fixed.png"
        plt.savefig(ablation_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"消融实验对比图保存到: {ablation_plot}")
        
        # 打印改进效果
        full_final_ppl = full_val_ppl[-1] if full_val_ppl else float('inf')
        decoder_final_ppl = decoder_val_ppl[-1] if decoder_val_ppl else float('inf')
        
        print(f"\n最终验证困惑度对比:")
        print(f"完整Transformer: {full_final_ppl:.2f}")
        print(f"Decoder-Only: {decoder_final_ppl:.2f}")
        
    except Exception as e:
        print(f"生成对比图时出错: {e}")
    
    return results

if __name__ == "__main__":
    # 可以选择训练单个模型或进行消融实验
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ablation':
            ablation_study()
        elif sys.argv[1] == 'decoder_only':
            train_transformer('decoder_only')
        else:
            train_transformer('full')
    else:
        print("使用方法:")
        print("  python train.py ablation     # 消融实验（修复过拟合）")
        print("  python train.py full         # 完整Transformer（修复过拟合）")
        print("  python train.py decoder_only # Decoder-Only")
        print("过拟合修复配置已启用!")
