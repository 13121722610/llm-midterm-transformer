#!/bin/bash

# =============================================================================
# LLM Midterm Transformer - 完整复现脚本
# 作者：季月含
# 学号：25120410
# 描述：一键复现Transformer训练和生成全过程
# =============================================================================

set -e  # 遇到任何错误立即退出

echo "================================================================"
echo "           LLM Midterm Transformer 复现脚本"
echo "================================================================"
echo "开始时间: $(date)"
echo ""

# ============================ 参数配置 ============================
SEED=42
PROJECT_DIR="/data0/yhji/llm-midterm-transformer"
PYTHON_PATH="$PROJECT_DIR/src"
EXPERIMENT_MODE=${1:-"ablation"}  # 默认进行消融实验

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
    pip install numpy pandas matplotlib requests tqdm tokenizers
fi

# ============================ 环境设置 ============================
echo ""
echo "=== 步骤3: 环境设置 ==="

# 设置Python路径
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"
export PYTHONHASHSEED=$SEED

# 检查并显示现有目录
echo "项目目录结构:"
for dir in "checkpoints" "results/figures" "results/tables" "results/logs" "data"; do
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
echo "实验模式: $EXPERIMENT_MODE"

# ============================ 数据准备 ============================
echo ""
echo "=== 步骤4: 数据准备 ==="

cd "$PROJECT_DIR"

# 下载数据
echo "下载训练数据..."
python -c "
from src.data import load_data
print('正在下载和预处理数据...')
tokenizer, train_data, val_data = load_data()
print(f'数据加载完成!')
print(f'词汇表大小: {tokenizer.vocab_size}')
print(f'训练数据长度: {len(train_data)}')
print(f'验证数据长度: {len(val_data)}')
"

# ============================ 模型训练 ============================
echo ""
echo "=== 步骤5: 模型训练 ==="

cd "$PROJECT_DIR/src"

# 正确的GPU检查 - 修复转义问题
echo "GPU状态:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA可用: True')
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
else:
    print(f'CUDA可用: False')
    print(f'GPU设备: CPU')
"

# 根据模式选择训练方式
case $EXPERIMENT_MODE in
    "ablation")
        echo "开始消融实验（完整Transformer vs Decoder-Only）..."
        echo "这将需要一些时间，请耐心等待..."
        python train.py ablation
        ;;
    "full")
        echo "开始训练完整Transformer模型..."
        python train.py full
        ;;
    "decoder_only")
        echo "开始训练Decoder-Only模型..."
        python train.py decoder_only
        ;;
    *)
        echo "未知模式: $EXPERIMENT_MODE，使用默认消融实验"
        python train.py ablation
        ;;
esac

# ============================ 结果验证 ============================
echo ""
echo "=== 步骤6: 训练结果验证 ==="

# 检查生成的图表和表格
echo "检查训练结果输出..."
if [ -d "$PROJECT_DIR/results" ]; then
    echo "📊 训练结果汇总:"
    
    # 检查图表文件
    if ls "$PROJECT_DIR/results/figures"/*.png 1> /dev/null 2>&1; then
        echo "✅ 训练曲线图:"
        ls "$PROJECT_DIR/results/figures"/*.png
    else
        echo "❌ 未找到训练曲线图"
    fi
    
    # 检查数据表格
    if ls "$PROJECT_DIR/results/tables"/*.csv 1> /dev/null 2>&1; then
        echo "✅ 结果表格:"
        ls "$PROJECT_DIR/results/tables"/*.csv
        echo ""
        echo "📈 最终结果:"
        python -c "
import pandas as pd
import glob
import os

csv_files = glob.glob('$PROJECT_DIR/results/tables/*final_results.csv')
for file in csv_files:
    df = pd.read_csv(file)
    filename = os.path.basename(file)
    print(f'文件: {filename}')
    for col in df.columns:
        if 'ppl' in col.lower() or 'loss' in col.lower():
            print(f'  {col}: {df[col].values[0]:.4f}')
    print()
        "
    else
        echo "❌ 未找到结果表格"
    fi
else
    echo "❌ 未找到results目录"
fi

# ============================ 文本生成 ============================
echo ""
echo "=== 步骤7: 文本生成测试 ==="

# 检查模型文件
models_to_test=()

if [ -f "$PROJECT_DIR/checkpoints/full_transformer_best.pt" ]; then
    models_to_test+=("full_transformer")
fi

if [ -f "$PROJECT_DIR/checkpoints/decoder_only_best.pt" ]; then
    models_to_test+=("decoder_only")
fi

if [ ${#models_to_test[@]} -eq 0 ]; then
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "请先运行训练步骤"
    exit 1
fi

echo "找到训练好的模型: ${models_to_test[*]}"
echo "开始生成测试..."

# 测试不同的prompt
prompts=("To be, or not to be" "Once upon a time" "The future of AI" "Love is" "In the beginning")

for model_type in "${models_to_test[@]}"; do
    echo ""
    echo "🔍 测试模型: $model_type"
    echo "========================================"
    
    for prompt in "${prompts[@]}"; do
        echo ""
        echo "Prompt: \"$prompt\""
        echo "生成结果:"
        
        # 调用生成脚本 - 修复转义问题
        cd "$PROJECT_DIR/src"
        python -c "
import torch
import sys
import os
sys.path.append('.')

try:
    from generate import generate_full_transformer, generate_decoder_only
    from data import load_data
    from model import FullTransformer, DecoderOnlyTransformer
    
    # 加载tokenizer和模型
    tokenizer, _, _ = load_data()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_type = '$model_type'
    prompt = '$prompt'
    
    if model_type == 'full_transformer':
        model = FullTransformer(tokenizer.vocab_size)
        model.load_state_dict(torch.load('../checkpoints/full_transformer_best.pt', map_location=device))
        model.to(device)
        result = generate_full_transformer(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8)
    else:
        model = DecoderOnlyTransformer(tokenizer.vocab_size)
        model.load_state_dict(torch.load('../checkpoints/decoder_only_best.pt', map_location=device))
        model.to(device)
        result = generate_decoder_only(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8)
    
    print(result)
except Exception as e:
    print(f'生成错误: {e}')
    import traceback
    traceback.print_exc()
        "
        echo "----------------------------------------"
    done
done

# ============================ 消融实验分析 ============================
echo ""
echo "=== 步骤8: 消融实验分析 ==="

if [ -f "$PROJECT_DIR/results/tables/ablation_study.csv" ]; then
    echo "📋 消融实验结果对比:"
    python -c "
import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取消融实验结果
ablation_df = pd.read_csv('$PROJECT_DIR/results/tables/ablation_study.csv')
print(ablation_df[['model_type', 'final_val_loss', 'final_val_ppl', 'best_val_ppl']].round(4))

# 生成简单的对比图
plt.figure(figsize=(10, 6))
models = ablation_df['model_type'].tolist()
ppls = ablation_df['best_val_ppl'].tolist()

bars = plt.bar(models, ppls, color=['skyblue', 'lightcoral'])
plt.ylabel('Best Validation Perplexity')
plt.title('Ablation Study: Model Performance Comparison')
plt.grid(True, alpha=0.3)

# 在柱状图上添加数值
for bar, ppl in zip(bars, ppls):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{ppl:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('$PROJECT_DIR/results/figures/ablation_summary.png', dpi=300, bbox_inches='tight')
print('✅ 消融实验总结图已保存: $PROJECT_DIR/results/figures/ablation_summary.png')
    "
else
    echo "ℹ️ 未找到消融实验数据表格"
fi

# ============================ 完成总结 ============================
echo ""
echo "=== 步骤9: 实验完成 ==="
echo "完成时间: $(date)"
echo ""
echo "🎉 实验结果汇总:"
echo "✅ 环境配置完成"
echo "✅ 依赖安装完成" 
echo "✅ 数据准备完成"
echo "✅ 模型训练完成 ($EXPERIMENT_MODE 模式)"
echo "✅ 训练结果验证完成"
echo "✅ 文本生成测试完成"
echo "✅ 消融实验分析完成"
echo ""
echo "📁 生成的文件:"
echo "模型文件: $PROJECT_DIR/checkpoints/"
echo "训练曲线: $PROJECT_DIR/results/figures/"
echo "数据表格: $PROJECT_DIR/results/tables/"
echo "训练日志: $PROJECT_DIR/results/logs/"
echo ""
echo "================================================================"
echo "                  复现脚本执行完毕！"
echo "================================================================"
