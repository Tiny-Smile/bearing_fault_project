# 轴承故障诊断项目

## 项目说明（环境配置/运行步骤/目录说明）

### 目录结构

```
bearing_fault_project/
├── p01_env/                # 虚拟环境（隔离依赖，避免冲突）
├── p02_data/               # 数据集（原始+预处理+增强后）
│   ├── raw/               # 原始数据集（CWRU+工业数据集，不修改）
│   │   ├── cwru/          # CWRU公开数据集（按故障类别分类）
│   │   └── industrial/    # 企业工业数据集（按传感器通道分类）
│   ├── preprocessed/      # 预处理后的数据（去趋势+滤波+裁剪）
│   │   ├── cwru/          # 预处理后的CWRU数据
│   │   └── industrial/    # 预处理后的工业数据
│   ├── cwt/               # CWT时频图数据（单通道/多通道张量）
│   └── augmented/         # cDWGAN-GP增强后的数据（后期扩展）
├── p03_code/               # 核心代码（按功能模块化，便于Windsurf生成/整合）
│   ├── p01_utils/          # 通用工具函数（所有模块复用）
│   │   ├── data_utils.py  # 数据加载/预处理/划分
│   │   ├── cwt_utils.py   # CWT变换（Morlet小波）
│   │   ├── metric_utils.py # 评价指标（准确率/F1/混淆矩阵）
│   │   ├── vis_utils.py   # 可视化（波形/时频图/混淆矩阵）
│   │   └── __init__.py    # 包初始化（方便导入）
│   ├── p02_models/         # 模型定义（按模型类型拆分）
│   │   ├── cnn_baseline.py # Baseline CNN模型
│   │   ├── cdwgan_gp.py   # 数据增强模型（后期扩展）
│   │   ├── swin_ca.py     # Swin-T+CA诊断模型（后期扩展）
│   │   └── __init__.py
│   ├── p03_train/          # 训练脚本（按模型拆分）
│   │   ├── train_cnn.py   # CNN Baseline训练
│   │   ├── train_cdwgan.py # cDWGAN-GP训练（后期扩展）
│   │   ├── train_swin.py  # Swin-T+CA训练（后期扩展）
│   │   └── __init__.py
│   ├── p04_test/           # 测试/评估脚本
│   │   ├── test_cnn.py    # CNN Baseline测试
│   │   ├── test_swin.py   # Swin-T+CA测试（后期扩展）
│   │   └── __init__.py
│   └── 05_run_all.py      # 端到端一键运行脚本（整合所有流程）
├── p04_models_ckpt/        # 训练好的模型权重（按模型/时间命名）
│   ├── cnn_baseline/      # CNN Baseline权重
│   ├── cdwgan_gp/         # cDWGAN-GP权重（后期扩展）
│   └── swin_ca/           # Swin-T+CA权重（后期扩展）
├── p05_logs/               # 训练日志（损失/准确率/报错信息）
│   ├── train_cnn.log      # CNN训练日志
│   ├── train_cdwgan.log   # cDWGAN训练日志（后期扩展）
│   └── error.log          # 全局错误日志
├── p06_results/            # 实验结果（可视化图+报告）
│   ├── figures/           # 论文用图（波形/时频图/混淆矩阵）
│   ├── metrics/           # 指标表格（csv格式）
│   └── report/            # 实验报告（md/pdf）
└── README.md              # 项目说明（环境配置/运行步骤/目录说明）
```

### 环境配置

#### 1. 虚拟环境创建
```bash
# 创建虚拟环境
python -m venv 01_env

# 激活虚拟环境 (Windows)
01_env\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source 01_env/bin/activate
```

#### 2. 依赖安装
```bash
# 安装所有依赖包
pip install -r requirements.txt
```

#### 3. 环境验证
```bash
# 运行环境验证脚本
python test_env.py
```

### 主要依赖包

- **深度学习框架**: torch==2.6.0+cu124, torchvision==0.21.0+cu124, torchaudio==2.6.0+cu124
- **数据处理**: numpy==2.0.2, pandas==2.3.3, scipy==1.13.1
- **机器学习**: scikit-learn==1.6.1, imbalanced-learn==0.14.1
- **可视化**: matplotlib==3.10.1, seaborn==0.13.2
- **信号处理**: PyWavelets==1.8.0
- **图像处理**: pillow==12.1.1
- **图论算法**: networkx==3.4.2
- **符号计算**: sympy==1.13.1

### 运行步骤

1. **数据准备**: 将原始数据放入 `p02_data/raw/` 目录
2. **数据预处理**: 运行数据预处理脚本
3. **模型训练**: 选择相应的训练脚本
4. **模型测试**: 运行测试脚本评估性能
5. **结果分析**: 查看 `p06_results/` 目录中的结果

### 项目特性

- **模块化设计**: 代码按功能模块组织，便于维护和扩展
- **多模型支持**: 支持CNN Baseline、Swin-T+CA等多种模型
- **数据增强**: 集成cDWGAN-GP数据增强技术
- **时频分析**: 使用CWT进行时频域特征提取
- **GPU加速**: 支持CUDA加速训练
