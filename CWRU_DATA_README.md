# CWRU 轴承数据集下载和使用指南

## 数据集介绍

CWRU (Case Western Reserve University) 轴承数据集是轴承故障诊断领域最常用的公开数据集。

**官方网站**: https://engineering.case.edu/bearingdatacenter

## 数据下载方法

### 方法一：使用下载脚本（推荐）

```bash
# 下载完整数据集
python p03_code/p01_utils/download_cwru_data.py --data_dir p02_data/raw/cwru

# 只下载样本数据（用于快速测试）
python p03_code/p01_utils/download_cwru_data.py --data_dir p02_data/raw/cwru --sample
```

### 方法二：手动下载

1. 访问 CWRU 官网: https://engineering.case.edu/bearingdatacenter
2. 下载以下数据文件（推荐下载 12kHz Drive End 数据）：
   - Normal Baseline Data
   - 12k Drive End Bearing Fault Data (0.007" fault diameter)
     - Inner Race Fault
     - Outer Race Fault  
     - Ball Fault

3. 将下载的 .zip 文件解压到 `p02_data/raw/cwru/12k_DE/` 目录

## 数据目录结构要求

程序支持以下两种目录结构：

### 标准结构（推荐）
```
p02_data/raw/cwru/
└── 12k_DE/
    ├── normal_0.mat
    ├── inner_race_0.mat
    ├── outer_race_0.mat
    └── ball_0.mat
```

### CWRU 官方结构
```
p02_data/raw/cwru/
└── 12k_DE/
    ├── Normal Baseline Data/
    │   ├── Normal1.mat
    │   ├── Normal2.mat
    │   └── ...
    └── 12k Drive End Bearing Fault Data/
        ├── 0.007/
        │   ├── 0/
        │   │   ├── Ball/
        │   │   ├── Inner Race/
        │   │   └── Outer Race/
        │   └── ...
```

## 数据集说明

### 故障类型
- **Normal (0)**: 正常状态
- **Inner_Race (1)**: 内圈故障
- **Outer_Race (2)**: 外圈故障
- **Ball (3)**: 滚动体故障

### 采样频率
- **12kHz**: 驱动端采样频率（推荐）
- **48kHz**: 风扇端采样频率

### 故障直径
- 0.007" (推荐)
- 0.014"
- 0.021"

## 使用真实数据运行

### 训练模型
```bash
python run_cnn1d.py --data_dir p02_data/raw/cwru/12k_DE --epochs 50
```

### 评估模型
```bash
python p03_code/p04_test/eval_cnn1d.py --data_dir p02_data/raw/cwru/12k_DE --model_path p04_models_ckpt/cnn1d_baseline.pth
```

## 注意事项

1. **文件格式**: 数据必须是 .mat 格式（MATLAB 格式）
2. **依赖库**: 需要安装 scipy 库来读取 .mat 文件
   ```bash
   pip install scipy
   ```
3. **数据大小**: 完整数据集约 500MB
4. **网络问题**: 如果下载脚本失败，请尝试手动下载

## 故障排除

### 问题：提示"数据目录不存在"
**解决**: 确保数据文件已下载并放置到正确的目录

### 问题：无法读取 .mat 文件
**解决**: 
- 检查 scipy 是否已安装
- 确认文件没有损坏

### 问题：数据加载失败
**解决**:
- 检查文件路径是否正确
- 确认文件格式为 .mat
- 查看控制台错误信息

## 数据集引用

如果在论文中使用此数据集，请引用：

```
@article{bearing_data_center,
  title={Bearing Data Center},
  author={Case Western Reserve University},
  journal={https://engineering.case.edu/bearingdatacenter},
  year={2024}
}
```
