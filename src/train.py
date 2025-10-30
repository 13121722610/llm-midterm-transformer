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
from model import DecoderOnlyTransformer  # 只导入DecoderOnlyTransformer

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

def get_model_config():
    """返回Decoder-Only模型的配置"""
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

def train_decoder_only():
    """训练Decoder-Only Transformer模型"""
    ensure_directories()
    
    # 获取配置
    config = get_model_config()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"使用设备: {device}")
    print(f"训练模型: Decoder-Only Transformer")
    print(f"配置: d_model={config['d_model']}, n_layer={config['n_layer']}, "
          f"n_head={config['n_head']}, dropout={config['dropout']}")

    # 加载数据
    tokenizer, train_data, val_data = load_data()
    train_dataset = CharDataset(train_data, config['block_size'])
    val_dataset = CharDataset(val_data, config['block_size'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # 初始化模型 - 只使用DecoderOnlyTransformer
    model = DecoderOnlyTransformer(
        tokenizer.vocab_size,
        d_model=config['d_model'], 
        n_layer=config['n_layer'], 
        n_head=config['n_head'], 
        d_ff=config['d_ff'], 
        max_seq_len=config['block_size'],
        dropout=config['dropout']
    )
    model_name = "decoder_only"
    
    model = model.to(device)
    
    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.98)
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
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
            
            # 简化：直接使用Decoder-Only模型
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

if __name__ == "__main__":
    print("开始训练Decoder-Only Transformer...")
    train_decoder_only()
