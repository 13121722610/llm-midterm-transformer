#!/bin/bash

# =============================================================================
# LLM Midterm Transformer - Decoder-Only 版本
# 作者：季月含
# 学号：25120410
# 描述：一键复现Decoder-Only Transformer训练和生成
# =============================================================================

set -e  # 遇到任何错误立即退出

echo "================================================================"
echo "           LLM Midterm Transformer (Decoder-Only)"
echo "================================================================"
echo "开始时间: $(date)"
echo ""

# ============================ 参数配置 ============================
SEED=42
PROJECT_DIR="/data0/yhji/llm-midterm-transformer"
PYTHON_PATH="$PROJECT_DIR/src"

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
    pip install numpy tqdm matplotlib requests tokenizers
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
    echo "跳过训练步骤，直接进行文本生成..."
else
    echo "开始训练Decoder-Only Transformer模型..."
    echo "这将需要一些时间，请耐心等待..."
    python train.py
fi

# ============================ 文本生成 ============================
echo ""
echo "=== 步骤6: 文本生成测试 ==="

# 检查模型文件
if [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_best.pt" ] || [ -f "$PROJECT_DIR/src/checkpoints/decoder_only_improved_best.pt" ]; then
    echo "找到训练好的Decoder-Only模型，开始生成测试..."
    
    # 测试不同的prompt
    prompts=("To be, or not to be" "Once upon a time" "The future of AI" "Love is" "In the beginning")
    
    for prompt in "${prompts[@]}"; do
        echo ""
        echo "Prompt: \"$prompt\""
        echo "生成结果:"
        python generate.py --prompt "$prompt" --temperature 0.8 --max_new_tokens 50
        echo "----------------------------------------"
    done
else
    echo "❌ 错误: 未找到训练好的模型文件"
    echo "请先运行训练步骤"
    exit 1
fi

# ============================ 完成总结 ============================
echo ""
echo "=== 步骤7: 实验完成 ==="
echo "完成时间: $(date)"
echo ""
echo "实验结果汇总:"
echo "✅ 环境配置完成"
echo "✅ 依赖安装完成" 
echo "✅ 数据准备完成"
echo "✅ 模型检查完成"
echo "✅ 文本生成测试完成"
echo ""
echo "项目文件:"
ls -la "$PROJECT_DIR/checkpoints/" | grep decoder_only
echo ""
echo "================================================================"
echo "                  Decoder-Only 版本执行完毕！"
echo "================================================================"
