#!/bin/bash

# =============================================================================
# LLM Midterm Transformer - 完整实验流程
# 作者：季月含
# 学号：25120410
# 描述：一键复现训练、消融实验和生成对比
# =============================================================================

set -e  # 遇到任何错误立即退出

echo "================================================================"
echo "           LLM Midterm Transformer (完整实验流程)"
echo "================================================================"
echo "开始时间: $(date)"
echo ""

# ============================ 参数配置 ============================
SEED=42
PROJECT_DIR="/data0/yhji/llm-midterm-transformer"
PYTHON_PATH="$PROJECT_DIR/src"
GPU_ID="3"  # 指定使用GPU 3

# ============================ 环境检查 ============================
echo "=== 步骤1: 环境检查 ==="

# 检查conda环境
if ! conda info --envs | grep -q "llm_toy"; then
    echo "创建conda环境: llm_toy"
    conda create -n llm_toy python=3.10 -y
else
    echo "找到conda环境: llm_toy"
fi

# 激活环境
echo "激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate llm_toy

# 检查Python
echo "Python版本: $(python --version)"

# ============================ 依赖安装 ============================
echo ""
echo "=== 步骤2: 安装依赖 ==="

# 检查是否有requirements.txt
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "使用requirements.txt安装依赖..."
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "未找到requirements.txt，安装核心依赖..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy tqdm matplotlib requests tokenizers pandas
fi

# ============================ 环境设置 ============================
echo ""
echo "=== 步骤3: 环境设置 ==="

# 设置Python路径
export PYTHONPATH="$PROJECT_DIR"
export PYTHONHASHSEED=$SEED

# 检查并显示现有目录
echo "项目目录结构:"
for dir in "checkpoints" "results" "data"; do
    if [ -d "$PROJECT_DIR/$dir" ]; then
        echo "✅ $PROJECT_DIR/$dir/ (已存在)"
    else
        echo "📁 创建目录: $PROJECT_DIR/$dir/"
        mkdir -p "$PROJECT_DIR/$dir"
    fi
done

echo "项目目录: $PROJECT_DIR"
echo "Python路径: $PYTHONPATH"
echo "随机种子: $SEED"
echo "使用GPU: $GPU_ID"

# ============================ 数据准备 ============================
echo ""
echo "=== 步骤4: 数据准备 ==="

cd "$PYTHON_PATH"

echo "下载训练数据..."
python -c "from data import load_data; tokenizer, train_data, val_data = load_data(); print(f'数据加载完成!\\n词汇表大小: {tokenizer.vocab_size}\\n训练数据长度: {len(train_data)}\\n验证数据长度: {len(val_data)}')"

# ============================ 模型训练 ============================
echo ""
echo "=== 步骤5: 模型训练 ==="

# 检查是否已有训练好的模型
if [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_best.pt" ] || [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_improved_best.pt" ]; then
    echo "✅ 发现已训练的Decoder-Only模型"
    echo "跳过训练步骤..."
else
    echo "开始训练Decoder-Only Transformer模型..."
    echo "这将需要一些时间，请耐心等待..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python train.py
fi

# ============================ 消融实验检查 ============================
echo ""
echo "=== 步骤6: 消融实验检查 ==="

# 检查消融实验模型文件是否完整
ablation_models_exist=true
for layers in 2 4 6; do
    if [ ! -f "$PROJECT_DIR/src/checkpoints/ablation_layers_${layers}_best.pt" ]; then
        ablation_models_exist=false
        echo "❌ 缺少消融实验模型: ablation_layers_${layers}_best.pt"
        break
    fi
done

# 检查消融实验结果文件
if [ -f "$PROJECT_DIR/src/results/tables/layer_ablation_results.csv" ] && $ablation_models_exist; then
    echo "✅ 消融实验已完成，跳过此步骤"
    echo "📊 消融实验结果:"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('results/tables/layer_ablation_results.csv')
    print(f'层数对比: 2层({df[df[\"layers\"]==2][\"best_val_ppl\"].iloc[0]:.2f}) | 4层({df[df[\"layers\"]==4][\"best_val_ppl\"].iloc[0]:.2f}) | 6层({df[df[\"layers\"]==6][\"best_val_ppl\"].iloc[0]:.2f})')
except:
    print('无法读取消融实验结果')
"
else
    echo "开始层数消融实验..."
    echo "这将训练2层、4层、6层模型进行对比..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python ablation_study.py
fi

# ============================ 图表生成 ============================
echo ""
echo "=== 步骤7: 图表生成 ==="

# 检查是否已有英文图表
if [ -f "$PROJECT_DIR/src/results/figures/layer_ablation_comparison_english.png" ]; then
    echo "✅ 英文图表已存在，跳过生成"
else
    echo "生成消融实验英文图表..."
    python -c "
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
try:
    results_df = pd.read_csv('results/tables/layer_ablation_results.csv')
    results = results_df.to_dict('records')
    layers = [r['layers'] for r in results]
    val_ppls = [r['best_val_ppl'] for r in results]
    parameters = [r['parameters'] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax1.bar([str(l) for l in layers], val_ppls, color=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_xlabel('Number of Layers')
    ax1.set_ylabel('Validation Perplexity (PPL)')
    ax1.set_title('Layer Ablation: Perplexity Comparison')
    ax1.grid(True, alpha=0.3)
    for bar, ppl in zip(bars, val_ppls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')
    ax2.bar([str(l) for l in layers], [p/1e6 for p in parameters], color='orange', alpha=0.7)
    ax2.set_xlabel('Number of Layers')
    ax2.set_ylabel('Parameters (Millions)')
    ax2.set_title('Layer Ablation: Parameter Count')
    ax2.grid(True, alpha=0.3)
    for i, (layer, param) in enumerate(zip(layers, parameters)):
        ax2.text(i, param/1e6 + 0.1, f'{param/1e6:.1f}M', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/layer_ablation_comparison_english.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ 英文图表已生成')
except Exception as e:
    print(f'❌ 图表生成失败: {e}')
"
fi

# ============================ 生成对比 ============================
echo ""
echo "=== 步骤8: 生成对比测试 ==="

# 检查是否已有生成对比结果，但强制重新运行
if [ -f "$PROJECT_DIR/src/results/tables/generation_comparison.csv" ] && [ "$1" != "--force" ]; then
    echo "✅ 生成对比已完成，跳过此步骤"
    echo "📝 生成对比样本:"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('results/tables/generation_comparison.csv')
    # 显示所有模型的样本
    for model in df['model'].unique():
        sample = df[df['model']==model].head(1)
        if not sample.empty:
            prompt = sample['prompt'].iloc[0]
            text = sample['generated_text'].iloc[0]
            print(f'{model}: {text[:50]}...')
except:
    print('无法读取生成对比结果')
"
else
    echo "开始模型生成对比测试..."
    
    # 创建生成对比脚本
    cat > compare_generation.py << 'EOF'
import torch
import pandas as pd
from data import load_data
from model import DecoderOnlyTransformer

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=20, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= 256 else idx[:, -256:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _, _ = load_data()
    
    model_configs = [
        {'layers': 2, 'path': 'checkpoints/ablation_layers_2_best.pt', 'name': '2层模型'},
        {'layers': 4, 'path': 'checkpoints/ablation_layers_4_best.pt', 'name': '4层模型'},
        {'layers': 6, 'path': 'checkpoints/ablation_layers_6_best.pt', 'name': '6层模型(3轮)'},
        {'layers': 6, 'path': 'checkpoints/decoder_only_improved_best.pt', 'name': '6层模型(5轮)'}
    ]
    
    test_prompts = [
        "To be, or not to be",
        "Once upon a time", 
        "The future of AI",
        "Love is",
        "In the beginning"
    ]
    
    results = []
    
    print("🎯 消融实验模型生成对比分析")
    print("=" * 70)
    
    for config in model_configs:
        print(f"\n🧪 {config['name']} ({config['layers']}层)")
        print("-" * 50)
        
        try:
            model = DecoderOnlyTransformer(
                vocab_size=tokenizer.vocab_size,
                d_model=256,
                n_layer=config['layers'],
                n_head=8,
                d_ff=1024,
                max_seq_len=256,
                dropout=0.1
            ).to(device)
            
            model.load_state_dict(torch.load(config['path'], map_location=device, weights_only=False))
            
            for prompt in test_prompts:
                generated = generate_text(model, tokenizer, prompt, max_new_tokens=30)
                print(f"📝 '{prompt}'")
                print(f"   → {generated}")
                
                results.append({
                    'model': config['name'],
                    'layers': config['layers'], 
                    'prompt': prompt,
                    'generated_text': generated
                })
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ 加载失败: {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/tables/generation_comparison.csv', index=False)
    print(f"\n✅ 生成对比完成! 结果保存至: results/tables/generation_comparison.csv")

if __name__ == "__main__":
    compare_models_generation()
EOF

    CUDA_VISIBLE_DEVICES=$GPU_ID python compare_generation.py
    rm compare_generation.py  # 清理临时文件
fi

# ============================ 基础生成测试 ============================
echo ""
echo "=== 步骤9: 基础生成测试 ==="

# 检查模型文件
if [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_best.pt" ] || [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_improved_best.pt" ]; then
    echo "基础生成测试..."
    
    prompts=("To be, or not to be" "Once upon a time" "The future of AI" "Love is" "In the beginning")
    
    for prompt in "${prompts[@]}"; do
        echo ""
        echo "Prompt: \"$prompt\""
        echo "生成结果:"
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate.py --prompt "$prompt" --temperature 0.8 --max_new_tokens 50
        echo "----------------------------------------"
    done
else
    echo "❌ 错误: 未找到训练好的模型文件"
    exit 1
fi

# ============================ 结果汇总 ============================
echo ""
echo "=== 步骤10: 实验结果汇总 ==="
echo "完成时间: $(date)"
echo ""
echo "🎉 实验完成汇总:"
echo "✅ 环境配置完成"
echo "✅ 依赖安装完成" 
echo "✅ 数据准备完成"
echo "✅ 模型训练完成"
echo "✅ 消融实验完成"
echo "✅ 图表生成完成"
echo "✅ 生成对比完成"
echo "✅ 基础生成测试完成"
echo ""
echo "📁 生成的文件:"
echo "模型文件:"
ls -la "$PROJECT_DIR/src/checkpoints/" | grep -E "(decoder_only|ablation_layers)" | head -10
echo ""
echo "结果文件:"
ls -la "$PROJECT_DIR/src/results/tables/"
ls -la "$PROJECT_DIR/src/results/figures/" | grep -E "(ablation|comparison)" | head -10
echo ""
echo "================================================================"
echo "                    完整实验流程执行完毕！"
echo "================================================================"
