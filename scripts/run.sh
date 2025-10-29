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

# ============================ 训练前检查 ============================
echo ""
echo "=== 步骤4: 训练状态检查 ==="

TRAINING_NEEDED=true

# 检查是否已经训练完成 - 修改为检查模型文件
check_training_complete() {
    # 检查各种可能的模型文件
    model_files=(
        "$PROJECT_DIR/src/checkpoints/decoder_only_improved_best.pt"
        "$PROJECT_DIR/src/checkpoints/full_transformer_improved_best.pt"
        "$PROJECT_DIR/src/checkpoints/decoder_only_best.pt"
        "$PROJECT_DIR/src/checkpoints/full_transformer_best.pt"
        "$PROJECT_DIR/src/checkpoints/best.pt"
        "$PROJECT_DIR/checkpoints/decoder_only_improved_best.pt"
        "$PROJECT_DIR/checkpoints/full_transformer_improved_best.pt"
    )
    
    found_models=()
    for model_file in "${model_files[@]}"; do
        if [ -f "$model_file" ]; then
            found_models+=("$model_file")
        fi
    done
    
    if [ ${#found_models[@]} -gt 0 ]; then
        echo "✅ 发现已训练的模型文件:"
        for model in "${found_models[@]}"; do
            echo "   - $(basename $model)"
        done
        echo ""
        echo "💡 提示: 模型已训练完成，跳过训练步骤"
        return 0  # 训练已完成
    else
        echo "❌ 未找到训练好的模型文件"
        return 1  # 需要训练
    fi
}

# 执行训练检查
if check_training_complete; then
    TRAINING_NEEDED=false
    echo "跳过训练步骤，直接进行文本生成..."
else
    echo "开始新的训练..."
fi

# ============================ 数据准备 ============================
echo ""
echo "=== 步骤5: 数据准备 ==="

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
echo "=== 步骤6: 模型训练 ==="

if [ "$TRAINING_NEEDED" = true ]; then
    cd "$PROJECT_DIR/src"

    # 正确的GPU检查
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
else
    echo "✅ 跳过训练步骤 - 模型已训练完成"
    
    # 即使跳过训练，也显示现有的训练结果（如果有的话）
    if [ -f "$PROJECT_DIR/src/results/tables/ablation_study.csv" ] || [ -f "$PROJECT_DIR/results/tables/ablation_study.csv" ]; then
        echo ""
        echo "📊 现有训练结果:"
        RESULTS_DIR="$PROJECT_DIR/src/results"
        if [ -f "$PROJECT_DIR/results/tables/ablation_study.csv" ]; then
            RESULTS_DIR="$PROJECT_DIR/results"
        fi
        
        python -c "
import pandas as pd
try:
    df = pd.read_csv('$RESULTS_DIR/tables/ablation_study.csv')
    if 'model_type' in df.columns and 'best_val_ppl' in df.columns:
        full_ppl = df[df['model_type']=='full']['best_val_ppl'].values[0]
        decoder_ppl = df[df['model_type']=='decoder_only']['best_val_ppl'].values[0]
        improvement = (decoder_ppl - full_ppl) / decoder_ppl * 100
        print(f'   完整Transformer - 困惑度: {full_ppl:.3f}')
        print(f'   Decoder-Only - 困惑度: {decoder_ppl:.3f}')
        print(f'   🎯 性能提升: {improvement:.1f}%')
    else:
        print('   训练结果格式不匹配')
except Exception as e:
    print('   读取训练结果时出错，但模型文件存在')
"
    fi
fi

# ============================ 结果验证 ============================
echo ""
echo "=== 步骤7: 训练结果验证 ==="

# 确定结果目录
if [ -d "$PROJECT_DIR/src/results" ]; then
    RESULTS_DIR="$PROJECT_DIR/src/results"
elif [ -d "$PROJECT_DIR/results" ]; then
    RESULTS_DIR="$PROJECT_DIR/results"
else
    RESULTS_DIR=""
fi

if [ -n "$RESULTS_DIR" ]; then
    echo "📊 训练结果汇总:"
    
    # 检查图表文件
    if ls "$RESULTS_DIR/figures"/*.png 1> /dev/null 2>&1; then
        echo "✅ 训练曲线图:"
        for file in "$RESULTS_DIR/figures"/*.png; do
            echo "   - $(basename $file)"
        done
    else
        echo "❌ 未找到训练曲线图"
    fi
    
    # 检查数据表格
    if ls "$RESULTS_DIR/tables"/*.csv 1> /dev/null 2>&1; then
        echo "✅ 结果表格:"
        for file in "$RESULTS_DIR/tables"/*.csv; do
            echo "   - $(basename $file)"
        done
        echo ""
        echo "📈 最终结果:"
        python -c "
import pandas as pd
import glob
import os

csv_files = glob.glob('$RESULTS_DIR/tables/*final_results.csv')
for file in csv_files:
    try:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        print(f'文件: {filename}')
        for col in df.columns:
            if 'ppl' in col.lower() or 'loss' in col.lower():
                print(f'  {col}: {df[col].values[0]:.4f}')
        print()
    except Exception as e:
        print(f'读取文件 {file} 时出错: {e}')
"
    else
        echo "❌ 未找到结果表格"
    fi
else
    echo "ℹ️ 未找到results目录 - 可能使用了现有模型"
fi

# ============================ 文本生成 ============================
echo ""
echo "=== 步骤8: 文本生成测试 ==="

# 检查模型文件
models_to_test=()

# 检查多个可能的模型文件位置
model_files_to_check=(
    "decoder_only_improved_best.pt"
    "full_transformer_improved_best.pt" 
    "decoder_only_best.pt"
    "full_transformer_best.pt"
    "best.pt"
)

# 修复：移除break语句，检查所有模型文件
for model_file in "${model_files_to_check[@]}"; do
    if [ -f "$PROJECT_DIR/src/checkpoints/$model_file" ]; then
        if [[ "$model_file" == *"decoder"* ]] && [[ ! " ${models_to_test[@]} " =~ " decoder_only " ]]; then
            models_to_test+=("decoder_only")
            echo "找到decoder模型: $model_file"
        elif [[ "$model_file" == *"full"* ]] && [[ ! " ${models_to_test[@]} " =~ " full_transformer " ]]; then
            models_to_test+=("full_transformer")
            echo "找到full transformer模型: $model_file"
        elif [[ "$model_file" == "best.pt" ]]; then
            # 检查best.pt实际是什么模型
            if [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_improved_best.pt" ] && [[ ! " ${models_to_test[@]} " =~ " decoder_only " ]]; then
                models_to_test+=("decoder_only")
                echo "best.pt指向decoder模型"
            elif [ -f "$PROJECT_DIR/src/checkpoints/full_transformer_improved_best.pt" ] && [[ ! " ${models_to_test[@]} " =~ " full_transformer " ]]; then
                models_to_test+=("full_transformer")
                echo "best.pt指向full transformer模型"
            elif [[ ! " ${models_to_test[@]} " =~ " decoder_only " ]]; then
                models_to_test+=("decoder_only")  # 默认
                echo "best.pt使用默认decoder模型"
            fi
        fi
        MODEL_DIR="$PROJECT_DIR/src/checkpoints"
    fi
done

# 去重
models_to_test=($(echo "${models_to_test[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

if [ ${#models_to_test[@]} -eq 0 ]; then
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "请先运行训练步骤"
    exit 1
fi

echo "找到训练好的模型: ${models_to_test[*]}"
echo "开始生成测试..."

# 测试不同的prompt
prompts=("To be, or not to be" "Once upon a time" "The future of AI" "Love is" "In the beginning")

# 使用修改后的generate.py进行文本生成
for model_type in "${models_to_test[@]}"; do
    echo ""
    echo "🔍 测试模型: $model_type"
    echo "========================================"
    
    for prompt in "${prompts[@]}"; do
        echo ""
        echo "Prompt: \"$prompt\""
        echo "生成结果:"
        
        # 使用修改后的generate.py进行生成（使用正确的配置）
        cd "$PROJECT_DIR/src"
        if [ "$model_type" == "full_transformer" ]; then
            python generate.py --model full --prompt "$prompt" --max_new_tokens 50 --temperature 0.8
        else
            python generate.py --model decoder --prompt "$prompt" --max_new_tokens 50 --temperature 0.8
        fi
        echo "----------------------------------------"
    done
done

# ============================ 完成总结 ============================
echo ""
echo "=== 步骤9: 实验完成 ==="
echo "完成时间: $(date)"
echo ""
echo "🎉 实验结果汇总:"
echo "✅ 环境配置完成"
echo "✅ 依赖安装完成" 
if [ "$TRAINING_NEEDED" = true ]; then
    echo "✅ 数据准备完成"
    echo "✅ 模型训练完成 ($EXPERIMENT_MODE 模式)"
else
    echo "✅ 使用现有训练结果"
fi
echo "✅ 训练结果验证完成"
echo "✅ 文本生成测试完成"
echo ""
echo "📁 生成的文件:"
if [ -n "$MODEL_DIR" ]; then
    echo "模型文件: $MODEL_DIR/"
    ls -la "$MODEL_DIR"/*.pt 2>/dev/null || echo "   无模型文件"
fi
if [ -n "$RESULTS_DIR" ]; then
    echo "训练曲线: $RESULTS_DIR/figures/"
    echo "数据表格: $RESULTS_DIR/tables/"
    echo "训练日志: $RESULTS_DIR/logs/"
fi
echo ""
echo "================================================================"
echo "                  复现脚本执行完毕！"
echo "================================================================"
