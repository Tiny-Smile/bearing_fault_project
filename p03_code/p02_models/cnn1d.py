"""
1D-CNN 模型定义 - CWRU 轴承故障诊断 Baseline

论文参考：
- InceptionTime (Ismail Fawaz et al., 2020) - 1D-CNN 时间序列分类经典论文
- 一维卷积网络用于轴承故障诊断的标准 Baseline 架构

模型结构（三层卷积 + 全局平均池化）：
    输入 (batch, 1, 1024)  ← 原始振动信号（1024 点，~85ms @12kHz）
    → Conv1d(k=16, c=32) + BatchNorm + ReLU + MaxPool(2)
    → Conv1d(k=8, c=64) + BatchNorm + ReLU + MaxPool(2)
    → Conv1d(k=8, c=128) + BatchNorm + ReLU
    → GlobalAvgPool
    → Dropout(0.5)
    → FC(128→64) → ReLU
    → FC(64→4) → Softmax
    输出 (batch, 4)  ← 4 类故障分类概率

特点：结构简单、标准、论文常用、易于复现，参数量适中（91K）
"""

# PyTorch 核心库，提供张量运算、自动微分、神经网络模块等基础设施
import torch
# PyTorch 神经网络模块，包含卷积、池化、全连接层等常用网络组件
import torch.nn as nn
# 类型提示模块，用于为函数参数、返回值和变量提供静态类型注解
from typing import List, Tuple


class CNN1D(nn.Module):
    """
    标准 1D-CNN 模型，用于轴承振动信号故障分类

    4 分类任务：
        - 0: 正常 (Normal)
        - 1: 内圈故障 (Inner Race)
        - 2: 外圈故障 (Outer Race)
        - 3: 滚动体故障 (Ball/Roller)
    """

    def __init__(
        self,
        input_length: int = 1024,      # 输入信号序列长度，默认 1024 点（1.28秒 @12kHz）
        in_channels: int = 1,          # 输入通道数，默认 1（单通道振动信号）
        num_classes: int = 4,          # 分类类别数，默认 4（正常+3种故障）
        kernel_size: int = 16,          # 第一层卷积核大小，默认 16 点（覆盖 ~1.3ms 时域窗口）
        first_channels: int = 32,      # 第一层卷积输出通道数，决定特征图深度，默认 32
    ) -> None:
        """
        初始化 1D-CNN 模型

        Args:
            input_length: 输入信号序列长度，默认 1024（1.28秒 @12kHz）
            in_channels: 输入通道数，默认 1（单通道振动信号）
            num_classes: 分类类别数，默认 4
            kernel_size: 第一层卷积核大小，默认 16
            first_channels: 第一层卷积输出通道数，默认 32
        """
        # 调用父类 nn.Module 的初始化方法，使类具备 PyTorch 模型的所有功能
        # 包括参数管理、GPU 迁移、模型保存与加载等
        super(CNN1D, self).__init__()

        # 模型名称标识，用于日志和保存文件时区分不同模型
        self.model_name = "1D-CNN-Benchmark"

        # ===== 第一层：浅层特征提取 =====
        # Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        # 输出形状: (batch, 32, input_length - 16 + 1)
        # 第一层使用较大的卷积核(16点)，在时域上覆盖约 1.3ms 窗口
        # 足够捕捉轴承振动信号中的周期冲击特征
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,      # 输入通道数，振动信号为 1
            out_channels=first_channels,  # 输出通道数，32 个卷积核产生 32 个特征图
            kernel_size=kernel_size,       # 卷积核长度 16，滑动窗口覆盖 16 个采样点
            stride=1,                      # 步长 1，逐点卷积，最大化时域信息保留
            padding=0,                     # 不填充，保持原始分辨率，不引入额外零值
        )
        # 批归一化层，对卷积输出在通道维度上进行标准化
        # 使每层输入分布稳定，加速收敛，防止梯度消失/爆炸
        # first_channels 个通道，每个通道独立计算均值和方差
        self.bn1 = nn.BatchNorm1d(first_channels)
        # ReLU 激活函数，f(x) = max(0, x)，引入非线性，增加模型表达能力
        # 同时将负值置零，产生稀疏表征
        self.relu1 = nn.ReLU()
        # 最大池化层，窗口大小 2，步长 2，将序列长度减半
        # 实现时间尺度的下采样，增大感受野，同时减少计算量和参数
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== 第二层：中级特征提取 =====
        # 输入来自第一层的 32 通道特征图，输出 64 通道
        # 卷积核缩小到 8 点，更精细地捕捉局部时域模式
        self.conv2 = nn.Conv1d(
            in_channels=first_channels,        # 32，来自第一层
            out_channels=first_channels * 2,    # 64，通道数翻倍，提取更丰富的特征
            kernel_size=8,                       # 卷积核 8 点，精细局部模式
            stride=1,                           # 步长 1，逐点卷积
            padding=0,                          # 不填充
        )
        # 批归一化，对 64 通道输出进行标准化
        self.bn2 = nn.BatchNorm1d(first_channels * 2)
        # ReLU 激活函数，引入非线性
        self.relu2 = nn.ReLU()
        # 最大池化，再次将序列长度减半（已经是 1/2 了，再减半）
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== 第三层：高级特征提取 =====
        # 输入 64 通道，输出 128 通道
        # 最深的卷积层，提取最高级的抽象特征
        self.conv3 = nn.Conv1d(
            in_channels=first_channels * 2,    # 64，来自第二层
            out_channels=first_channels * 4,   # 128，通道数再次翻倍
            kernel_size=8,                     # 卷积核 8 点
            stride=1,                           # 步长 1
            padding=0,                          # 不填充
        )
        # 批归一化，对 128 通道输出进行标准化
        self.bn3 = nn.BatchNorm1d(first_channels * 4)
        # ReLU 激活函数，引入非线性
        self.relu3 = nn.ReLU()

        # ===== 全局池化 =====
        # 自适应平均池化，将任意长度序列压缩为单点输出
        # 输入: (batch, 128, seq_len/4)，输出: (batch, 128, 1)
        # 取代全连接层，大幅减少参数量，同时对输入长度不敏感
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ===== 全连接分类器 =====
        # 第一个全连接层，将 128 维全局特征映射到 64 维隐藏空间
        # 起到"瓶颈层"作用，进一步压缩和抽象特征
        self.fc1 = nn.Linear(first_channels * 4, 64)
        # Dropout 层，训练时随机丢弃 50% 的神经元，防止过拟合
        # 增强模型泛化能力，使其不过分依赖特定神经元的激活
        self.dropout = nn.Dropout(p=0.5)
        # 第二个全连接层，输出分类结果（4 类）
        # 每个输出节点对应一个类别的未归一化得分（logit）
        self.fc2 = nn.Linear(64, num_classes)

        # 初始化网络权重，使用自定义初始化方法
        # 默认 PyTorch 初始化可能不适用于此架构，自定义可加速收敛
        self._init_weights()

    def _init_weights(self) -> None:
        """
        初始化网络权重

        - Conv1d 层：使用 Kaiming 正态初始化（适合 ReLU 激活）
        - BatchNorm1d 层：权重初始化为 1，偏置初始化为 0（恒等变换）
        - Linear 层：使用 Xavier 正态初始化（适合对称激活如 ReLU）
        """
        # 遍历网络中所有模块（包括层本身和子模块）
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming 正态初始化，专为 ReLU 设计
                # mode='fan_out' 按输出通道数分配权重，nonlinearity='relu' 适配 ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果存在偏置项，初始化为 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm 权重初始化为 1，偏置初始化为 0
                # 使归一化最初不起作用，模型逐渐学习合适的缩放和平移
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier 正态初始化，使前向和反向传播中每层方差保持一致
                nn.init.xavier_normal_(m.weight)
                # 偏置初始化为 0
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, 1, 序列长度)

        Returns:
            输出张量，形状 (batch_size, 类别数)
        """
        # 第一层：浅层特征提取
        # 卷积：检测局部时域模式（如短周期的冲击）
        x = self.conv1(x)
        # 批归一化：稳定特征分布
        x = self.bn1(x)
        # ReLU：引入非线性，负值置零
        x = self.relu1(x)
        # 最大池化：降采样，扩大感受野，保留最显著特征
        x = self.pool1(x)

        # 第二层：中级特征提取
        # 卷积：在第一层输出的基础上组合更复杂的局部模式
        x = self.conv2(x)
        # 批归一化：稳定特征分布
        x = self.bn2(x)
        # ReLU：引入非线性
        x = self.relu2(x)
        # 最大池化：进一步降采样
        x = self.pool2(x)

        # 第三层：高级特征提取
        # 卷积：最深层，提取全局和抽象的故障特征
        x = self.conv3(x)
        # 批归一化：稳定特征分布
        x = self.bn3(x)
        # ReLU：引入非线性
        x = self.relu3(x)

        # 全局池化：将特征图压缩为单一数值向量
        # 每个通道取平均，将任意长度的序列转化为固定长度 128
        x = self.global_pool(x)
        # 展平操作，将三维张量 (batch, 128, 1) 变为二维 (batch, 128)
        # 为全连接层做准备
        x = x.view(x.size(0), -1)

        # 全连接层：分类决策
        # 将 128 维特征映射到 64 维隐藏空间，进一步抽象
        x = self.fc1(x)
        # ReLU：引入非线性，产生最终的分类特征
        x = self.relu1(x)
        # Dropout：训练时随机丢弃 50% 神经元，防止过拟合
        # 推理时不生效（自动关闭）
        x = self.dropout(x)
        # 输出层：将 64 维特征映射到 4 个类别得分（logit）
        # 配合 CrossEntropyLoss 自动完成 Softmax 归一化
        x = self.fc2(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量（用于 t-SNE/PCA 可视化）

        与 forward() 的区别在于：只返回 fc1 的输出（64 维），
        而不是 fc2 的分类 logits（4 维），更适合可视化高维特征空间。

        Args:
            x: 输入张量，形状 (batch_size, 1, 序列长度)

        Returns:
            特征向量，形状 (batch_size, 128)（fc1 输出前）
        """
        # 第一层：浅层特征提取（与 forward 相同）
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层：中级特征提取（与 forward 相同）
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 第三层：高级特征提取（与 forward 相同）
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 全局池化：将特征图压缩为单一数值向量
        x = self.global_pool(x)
        # 展平为 (batch, 128)
        x = x.view(x.size(0), -1)

        # 第一个全连接层：输出 64 维特征向量
        x = self.fc1(x)

        return x

    def get_total_params(self) -> int:
        """
        获取模型参数总量

        Returns:
            模型中所有可学习参数的总数量（标量）
        """
        # 遍历所有参数并求和，p.numel() 返回张量中的元素个数
        return sum(p.numel() for p in self.parameters())


def create_model(input_length: int = 1024, device: str = "cuda") -> Tuple[nn.Module, str]:
    """
    便捷函数：创建并返回 1D-CNN 模型

    Args:
        input_length: 输入信号序列长度，默认 1024 点
        device: 训练设备，"cuda" 表示 GPU，"cpu" 表示 CPU

    Returns:
        (模型实例, 设备字符串) — 模型已移动到指定设备上
    """
    # 实例化 CNN1D 模型
    model = CNN1D(input_length=input_length)
    # 检查 CUDA 是否可用，自动降级到 CPU（无 GPU 或未安装 CUDA 版 PyTorch）
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # 将模型参数和缓冲区移动到指定设备（GPU/CPU）
    model = model.to(device)

    return model, str(device)


def print_model_structure(model: nn.Module, input_length: int = 1024) -> None:
    """
    打印模型结构详情

    打印内容包括：模型名称、各层参数量、总体参数量、输入输出形状摘要。

    Args:
        model: 1D-CNN 模型实例
        input_length: 输入序列长度（用于摘要说明）
    """
    # 打印模型名称（居中分隔线）
    print(f"\n{'='*60}")
    print(f"模型: {model.model_name}")
    print(f"{'='*60}")

    # 初始化总参数量计数器
    total_params = 0
    # 打印各层参数量表头
    print("\n各层参数量:")
    print("-" * 50)
    # 遍历所有命名参数
    for name, param in model.named_parameters():
        # 该参数的元素总数（如权重矩阵的元素个数）
        param_count = param.numel()
        # 累加到总参数量
        total_params += param_count
        # 左对齐名称（40 字符宽），右对齐参数量（千分位格式化）
        print(f"  {name:40s}: {param_count:>10,}")

    # 打印分隔线
    print("-" * 50)
    # 打印总参数量
    print(f"  {'Total Params':40s}: {total_params:>10,}")
    # 打印模型结构摘要（输入输出形状）
    print(f"\n模型摘要: 1D-CNN, 3层Conv + GlobalAvgPool + 2层FC")
    print(f"输入形状: (batch, 1, {input_length})")
    print(f"输出形状: (batch, 4)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试模型结构和前向传播
    # 这段代码仅在直接运行本文件时执行，import 时不执行

    # 创建模型实例，input_length=1024
    model, device = create_model(input_length=1024)
    # 打印模型各层参数量和结构摘要
    print_model_structure(model)

    # 测试前向传播
    batch_size = 8                                              # 测试批次大小：8 个样本
    input_tensor = torch.randn(batch_size, 1, 1024).to(device)   # 随机生成 (8, 1, 1024) 的测试张量
    output = model(input_tensor)                                 # 前向传播，得到 (8, 4) 的分类 logits

    # 打印结果验证
    print(f"输入形状: {input_tensor.shape}")  # 验证输入形状应为 torch.Size([8, 1, 1024])
    print(f"输出形状: {output.shape}")        # 验证输出形状应为 torch.Size([8, 4])
    print(f"设备: {device}")                  # 显示模型运行的设备（cuda/cpu）
    print(f"参数量: {model.get_total_params():,} 参数")  # 打印总参数量（格式化千分位）
