"""
1D-CNN 模型定义 - CWRU 轴承故障诊断 Baseline

论文参考：
- InceptionTime (Ismail Fawaz et al., 2020) - 1D-CNN 时间序列分类经典论文
- 一维卷积网络用于轴承故障诊断的标准 Baseline 架构

模型结构：
    Conv1d → BatchNorm → ReLU → MaxPool
    → Conv1d → BatchNorm → ReLU
    → Conv1d → BatchNorm → ReLU
    → GlobalAvgPool → FC → Softmax

适合作为对比基线：结构简单、标准、论文常用、易于复现
"""

import torch
import torch.nn as nn
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
        输入长度: int = 1024,
        通道数: int = 1,
        类别数: int = 4,
        卷积核大小: int = 16,
        第一层通道: int = 32,
    ) -> None:
        """
        初始化 1D-CNN 模型

        Args:
            输入长度: 输入信号序列长度，默认 1024（1.28秒 @12kHz）
            通道数: 输入通道数，默认 1（单通道振动信号）
            类别数: 分类类别数，默认 4
            卷积核大小: 第一层卷积核大小，默认 16
            第一层通道: 第一层卷积输出通道数，默认 32
        """
        super(CNN1D, self).__init__()

        # 模型名称
        self.模型名称 = "1D-CNN-Benchmark"

        # ===== 第一层：浅层特征提取 =====
        # Conv1d(in_channels=1, out_channels=32, kernel_size=16)
        # 输出形状: (batch, 32, 输入长度 - 16 + 1)
        self.conv1 = nn.Conv1d(
            in_channels=通道数,
            out_channels=第一层通道,
            kernel_size=卷积核大小,
            stride=1,
            padding=0,  # 不padding，保留时域信息
        )
        self.bn1 = nn.BatchNorm1d(第一层通道)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 下采样，减半长度

        # ===== 第二层：中级特征提取 =====
        self.conv2 = nn.Conv1d(
            in_channels=第一层通道,
            out_channels=第一层通道 * 2,  # 64
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.bn2 = nn.BatchNorm1d(第一层通道 * 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ===== 第三层：高级特征提取 =====
        self.conv3 = nn.Conv1d(
            in_channels=第一层通道 * 2,
            out_channels=第一层通道 * 4,  # 128
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm1d(第一层通道 * 4)
        self.relu3 = nn.ReLU()

        # ===== 全局池化 =====
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 输出形状: (batch, 128, 1)

        # ===== 全连接分类器 =====
        self.fc1 = nn.Linear(第一层通道 * 4, 64)
        self.dropout = nn.Dropout(p=0.5)  # Dropout 防止过拟合
        self.fc2 = nn.Linear(64, 类别数)

        # 权重初始化（Xavier）
        self._初始化权重()

    def _初始化权重(self) -> None:
        """
        使用 Xavier 均匀初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def 前向传播(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, 1, 序列长度)

        Returns:
            输出张量，形状 (batch_size, 类别数)
        """
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平: (batch, 128)

        # 全连接层
        x = self.fc1(x)
        x = self.relu1(x)  # 复用 ReLU
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def 提取特征(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征向量（用于 t-SNE 可视化）

        Args:
            x: 输入张量，形状 (batch_size, 1, 序列长度)

        Returns:
            特征向量，形状 (batch_size, 128)
        """
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)

        return x

    def 获取参数总量(self) -> int:
        """
        获取模型参数总量

        Returns:
            参数量
        """
        return sum(p.numel() for p in self.parameters())


def 创建模型(输入长度: int = 1024, 设备: str = "cuda") -> Tuple[nn.Module, str]:
    """
    便捷函数：创建并返回 1D-CNN 模型

    Args:
        输入长度: 输入信号序列长度
        设备: 训练设备，"cuda" 或 "cpu"

    Returns:
        (模型实例, 设备字符串)
    """
    模型 = CNN1D(输入长度=输入长度)
    设备 = torch.device(设备 if torch.cuda.is_available() else "cpu")
    模型 = 模型.to(设备)

    return 模型, str(设备)


def 打印模型结构(模型: nn.Module, 输入长度: int = 1024) -> None:
    """
    打印模型结构详情

    Args:
        模型: 1D-CNN 模型实例
        输入长度: 输入序列长度
    """
    print(f"\n{'='*60}")
    print(f"模型: {模型.模型名称}")
    print(f"{'='*60}")

    # 计算参数量
    总参数量 = 0
    print("\n各层参数量:")
    print("-" * 50)
    for name, param in 模型.named_parameters():
        param_count = param.numel()
        总参数量 += param_count
        print(f"  {name:40s}: {param_count:>10,}")

    print("-" * 50)
    print(f"  {'总参数量':40s}: {总参数量:>10,}")
    print(f"\n模型摘要: 1D-CNN, 3层Conv + GlobalAvgPool + 2层FC")
    print(f"输入形状: (batch, 1, {输入长度})")
    print(f"输出形状: (batch, 4)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 测试模型
    模型, 设备 = 创建模型(输入长度=1024)
    打印模型结构(模型)

    # 测试前向传播
    batch_size = 8
    输入 = torch.randn(batch_size, 1, 1024).to(设备)
    输出 = 模型.前向传播(输入)

    print(f"输入形状: {输入.shape}")
    print(f"输出形状: {输出.shape}")
    print(f"设备: {设备}")
    print(f"参数量: {模型.获取参数总量():,} 参数")
