# 【LLM Midterm Transformer - 文本生成与消融实验】
基于Transformer的莎士比亚文本生成与模型架构分析
Course assignment: decoder-only Transformer (Tiny Shakespeare)

# 📋 项目概述
本项目实现了一个完整的Transformer文本生成系统，包含Decoder-Only Transformer架构。通过层数消融实验（2/4/6层对比）和生成质量分析，深入探究Transformer架构设计对文本生成性能的影响。项目使用Tiny Shakespeare数据集进行训练和评估。

# 🎯 核心特性
Decoder-Only架构: 优化的自回归文本生成模型。
智能采样策略: Top-k + Top-p 混合采样，提升生成多样性  
层数消融实验: 系统分析不同层数架构的性能差异  
完整实验流程: 一键复现训练、评估、生成全流程  

# 🚀 环境配置
创建conda环境
conda create -n llm_toy python=3.10 -y
conda activate llm_toy
安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm matplotlib requests pandas

一键运行完整实验：
赋予执行权限
chmod +x run.sh
运行完整实验流程
./run.sh

分步执行：
1、数据准备
cd src
python -c "from data import load_data; tokenizer, train_data, val_data = load_data()"
2、模型训练
python train.py
3、消融实验
python ablation_study.py
4、生成对比
python compare_generation.py
5、单次生成
python generate.py

# 🧠 模型架构
Decoder-Only Transformer
本项目主要采用Decoder-Only Transformer架构，专为自回归文本生成设计：
层数配置: 2层/4层/6层 (消融实验对比)
隐藏维度: 256
注意力头数: 8头多头注意力
前馈网络: 1024维
位置编码: 可学习位置嵌入
词汇表大小: ~65个字符 (基于Shakespeare数据集)

# 🔬 实验设计
消融实验目标：
架构深度影响: 分析层数对模型表达能力和泛化能力的影响
训练效率: 比较不同复杂度模型的收敛速度和计算需求
生成质量: 评估不同架构的文本连贯性、创造性和语义一致性

评估指标：
主要指标: 验证集困惑度(Perplexity)
生成评估: 文本连贯性、多样性、语义合理性
效率指标: 训练时间、推理速度、内存占用

实验设置：
数据集: Tiny Shakespeare (1MB文本)
训练/验证分割: 90%/10%
批量大小: 32
优化器: AdamW (lr=1e-4, weight_decay=0.01)
早停策略: 5轮无改善停止

# 📈 可视化输出
项目自动生成以下分析图表：
训练曲线图: 损失和困惑度随训练轮次的变化
消融对比图: 不同层数架构的性能柱状图对比
参数量分析: 模型复杂度与性能的关系
生成样本对比: 各模型生成文本的定性分析

# 🛠️ 开发说明
代码结构：
data.py: 数据处理管道，包含字符级tokenizer
model.py: Transformer模型定义，支持灵活的层数配置
train.py: 训练循环、验证和模型保存
generate.py: 文本生成接口，支持多种采样策略
compare_generation.py: 生成质量对比分析模块
ablation_study.py: 消融实验的自动化执行和分析

# 👥 贡献者
季月含 - 25120410 - 项目开发与实验设计
