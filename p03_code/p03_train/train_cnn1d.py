"""
1D-CNN 轴承故障诊断模型训练脚本

完整训练流程：
1. 数据加载与预处理
2. 模型构建
3. 训练循环（含早停、学习率调度）
4. 模型评估
5. 结果保存与可视化

使用方法：
    python train_cnn1d.py
    python train_cnn1d.py --config config_cnn1d.yaml
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目路径
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_code_dir = os.path.dirname(_current_dir)  # p03_code目录
sys.path.insert(0, _current_dir)
sys.path.insert(0, _code_dir)

from p01_utils.cwru_dataset import (
    CWRUDataset,
    创建数据加载器,
    划分训练测试集,
    打印数据统计,
    CLASS_NAMES,
)
from p02_models.cnn1d import CNN1D, 打印模型结构


# =============================================================================
# 路径配置
# =============================================================================
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_search_dir = _current_dir
PROJECT_ROOT = None

while _search_dir != os.path.dirname(_search_dir):
    _parent = os.path.dirname(_search_dir)
    if (os.path.exists(os.path.join(_parent, "p02_data")) and
            os.path.exists(os.path.join(_parent, "p03_code"))):
        PROJECT_ROOT = _parent
        break
    _search_dir = _parent

if PROJECT_ROOT is None:
    PROJECT_ROOT = _current_dir

# 输出目录
LOG_DIR = os.path.join(PROJECT_ROOT, "p05_logs")
MODEL_CKPT_DIR = os.path.join(PROJECT_ROOT, "p04_models_ckpt")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "p06_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for _dir in [LOG_DIR, MODEL_CKPT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(_dir, exist_ok=True)


# =============================================================================
# 训练器类
# =============================================================================
class 训练器:
    """
    1D-CNN 模型训练器

    负责：
    - 模型训练循环
    - 验证与测试
    - 学习率调度
    - 早停机制
    - 训练过程记录
    """

    def __init__(
        self,
        模型,
        设备: torch.device,
        学习率: float = 0.001,
        权重衰减: float = 1e-4,
        优化器类型: str = "adam",
        调度器类型: str = "step",
        调度器步长: int = 20,
        调度器gamma: float = 0.5,
        早停耐心值: int = 15,
        早停最小变化: float = 0.001,
        梯度裁剪阈值: Optional[float] = None,
    ) -> None:
        """
        初始化训练器

        Args:
            模型: PyTorch 模型
            设备: 训练设备
            学习率: 初始学习率
            权重衰减: L2 正则化系数
            优化器类型: "adam", "sgd", "adamw"
            调度器类型: "step", "cosine", "none"
            调度器步长: 学习率下降间隔（step调度器）
            调度器gamma: 学习率下降倍数
            早停耐心值: 早停容忍轮数
            早停最小变化: 被认为有提升的最小变化量
            梯度裁剪阈值: 梯度裁剪阈值，None表示不裁剪
        """
        self.模型 = 模型
        self.设备 = 设备
        self.梯度裁剪阈值 = 梯度裁剪阈值

        # 损失函数
        self.损失函数 = nn.CrossEntropyLoss()

        # 优化器
        if 优化器类型.lower() == "adam":
            self.优化器 = optim.Adam(
                模型.parameters(),
                lr=学习率,
                weight_decay=权重衰减,
            )
        elif 优化器类型.lower() == "sgd":
            self.优化器 = optim.SGD(
                模型.parameters(),
                lr=学习率,
                momentum=0.9,
                weight_decay=权重衰减,
            )
        elif 优化器类型.lower() == "adamw":
            self.优化器 = optim.AdamW(
                模型.parameters(),
                lr=学习率,
                weight_decay=权重衰减,
            )
        else:
            raise ValueError(f"不支持的优化器类型: {优化器类型}")

        # 学习率调度器
        if 调度器类型.lower() == "step":
            self.调度器 = StepLR(
                self.优化器,
                step_size=调度器步长,
                gamma=调度器gamma,
            )
        elif 调度器类型.lower() == "cosine":
            self.调度器 = CosineAnnealingLR(
                self.优化器,
                T_max=50,
                eta_min=1e-6,
            )
        else:
            self.调度器 = None

        # 早停配置
        self.早停耐心值 = 早停耐心值
        self.早停最小变化 = 早停最小变化
        self.早停计数器 = 0
        self.最佳验证损失 = float("inf")
        self.最佳模型状态 = None

        # 训练历史
        self.训练历史 = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def 训练轮次(
        self,
        训练数据加载器: DataLoader,
        验证数据加载器: Optional[DataLoader] = None,
        轮次: int = 1,
        日志间隔: int = 10,
    ) -> Dict[str, float]:
        """
        训练一个轮次

        Args:
            训练数据加载器: 训练数据迭代器
            验证数据加载器: 验证数据迭代器（可选）
            轮次: 当前轮次编号
            日志间隔: 日志打印间隔

        Returns:
            本轮训练指标字典
        """
        self.模型.train()
        总损失 = 0.0
        正确数 = 0
        总样本数 = 0

        for 批次索引, (数据, 标签) in enumerate(训练数据加载器):
            数据 = 数据.to(self.设备)
            标签 = 标签.to(self.设备)

            # 前向传播
            self.优化器.zero_grad()
            输出 = self.模型.前向传播(数据)
            损失 = self.损失函数(输出, 标签)

            # 反向传播
            损失.backward()

            # 梯度裁剪
            if self.梯度裁剪阈值 is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.模型.parameters(),
                    self.梯度裁剪阈值,
                )

            self.优化器.step()

            # 统计
            总损失 += 损失.item()
            预测 = 输出.argmax(dim=1)
            正确数 += (预测 == 标签).sum().item()
            总样本数 += 标签.size(0)

            # 打印日志
            if (批次索引 + 1) % 日志间隔 == 0:
                当前准确率 = 100.0 * 正确数 / 总样本数
                当前损失 = 总损失 / (批次索引 + 1)
                print(
                    f"  Epoch {轮次:3d} | Batch {批次索引+1:4d}/{len(训练数据加载器):4d} | "
                    f"Loss: {当前损失:.4f} | Acc: {当前准确率:.2f}%"
                )

        # 计算本轮平均指标
        平均损失 = 总损失 / len(训练数据加载器)
        训练准确率 = 100.0 * 正确数 / 总样本数

        # 更新学习率
        if self.调度器 is not None:
            self.调度器.step()
            当前学习率 = self.调度器.get_last_lr()[0]
        else:
            当前学习率 = self.优化器.param_groups[0]["lr"]

        # 记录历史
        self.训练历史["train_loss"].append(平均损失)
        self.训练历史["train_acc"].append(训练准确率)
        self.训练历史["learning_rate"].append(当前学习率)

        # 验证
        验证指标 = {}
        if 验证数据加载器 is not None:
            验证指标 = self._验证(验证数据加载器)
            self.训练历史["val_loss"].append(验证指标["loss"])
            self.训练历史["val_acc"].append(验证指标["acc"])

            # 早停检查
            if 验证指标["loss"] < self.最佳验证损失 - self.早停最小变化:
                self.最佳验证损失 = 验证指标["loss"]
                self.最佳模型状态 = {k: v.cpu().clone() for k, v in self.模型.state_dict().items()}
                self.早停计数器 = 0
                print(f"  ★ 验证损失改善: {self.最佳验证损失:.4f}")
            else:
                self.早停计数器 += 1

        return {
            "train_loss": 平均损失,
            "train_acc": 训练准确率,
            "val_loss": 验证指标.get("loss", 0),
            "val_acc": 验证指标.get("acc", 0),
            "learning_rate": 当前学习率,
        }

    def _验证(self, 验证数据加载器: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估模型

        Args:
            验证数据加载器: 验证数据迭代器

        Returns:
            验证指标字典
        """
        self.模型.eval()
        总损失 = 0.0
        正确数 = 0
        总样本数 = 0

        with torch.no_grad():
            for 数据, 标签 in 验证数据加载器:
                数据 = 数据.to(self.设备)
                标签 = 标签.to(self.设备)

                输出 = self.模型.前向传播(数据)
                损失 = self.损失函数(输出, 标签)

                总损失 += 损失.item()
                预测 = 输出.argmax(dim=1)
                正确数 += (预测 == 标签).sum().item()
                总样本数 += 标签.size(0)

        return {
            "loss": 总损失 / len(验证数据加载器),
            "acc": 100.0 * 正确数 / 总样本数,
        }

    def 是否早停(self) -> bool:
        """检查是否应该早停"""
        return self.早停计数器 >= self.早停耐心值

    def 恢复最佳模型(self) -> None:
        """恢复到验证集上表现最好的模型"""
        if self.最佳模型状态 is not None:
            self.模型.load_state_dict(self.最佳模型状态)
            print(f"  已恢复最佳模型 (val_loss={self.最佳验证损失:.4f})")

    def 保存训练历史(self, 保存路径: str) -> None:
        """保存训练历史到文件"""
        with open(保存路径, "w") as f:
            json.dump(self.训练历史, f, indent=2)


# =============================================================================
# 评估函数
# =============================================================================
def 评估模型(
    模型: nn.Module,
    测试数据加载器: DataLoader,
    设备: torch.device,
    类别名称: List[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    在测试集上评估模型

    Args:
        模型: 训练好的模型
        测试数据加载器: 测试数据迭代器
        设备: 训练设备
        类别名称: 类别名称列表

    Returns:
        (指标字典, 混淆矩阵, 预测标签数组, 真实标签数组)
    """
    if 类别名称 is None:
        类别名称 = ["Normal", "Inner_Race", "Outer_Race", "Ball"]

    模型.eval()
    所有真实标签 = []
    所有预测标签 = []
    所有预测概率 = []

    with torch.no_grad():
        for 数据, 标签 in 测试数据加载器:
            数据 = 数据.to(设备)
            标签 = 标签.to(设备)

            输出 = 模型.前向传播(数据)
            预测 = 输出.argmax(dim=1)
            概率 = torch.softmax(输出, dim=1)

            所有真实标签.extend(标签.cpu().numpy())
            所有预测标签.extend(预测.cpu().numpy())
            所有预测概率.extend(概率.cpu().numpy())

    所有真实标签 = np.array(所有真实标签)
    所有预测标签 = np.array(所有预测标签)

    # 计算指标
    准确率 = 100.0 * accuracy_score(所有真实标签, 所有预测标签)
    精确率 = 100.0 * precision_score(
        所有真实标签, 所有预测标签, average="weighted", zero_division=0
    )
    召回率 = 100.0 * recall_score(
        所有真实标签, 所有预测标签, average="weighted", zero_division=0
    )
    f1分数 = 100.0 * f1_score(
        所有真实标签, 所有预测标签, average="weighted", zero_division=0
    )

    指标 = {
        "accuracy": 准确率,
        "precision": 精确率,
        "recall": 召回率,
        "f1_score": f1分数,
    }

    # 混淆矩阵
    混淆矩阵 = confusion_matrix(所有真实标签, 所有预测标签)

    return 指标, 混淆矩阵, 所有预测标签, 所有真实标签


def 打印分类报告(
    真实标签: np.ndarray,
    预测标签: np.ndarray,
    类别名称: List[str],
) -> str:
    """
    打印并返回分类报告

    Args:
        真实标签: 真实标签数组
        预测标签: 预测标签数组
        类别名称: 类别名称列表

    Returns:
        分类报告字符串
    """
    报告 = classification_report(
        真实标签,
        预测标签,
        target_names=类别名称,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("分类报告 (Classification Report)")
    print("=" * 60)
    print(报告)
    return 报告


# =============================================================================
# 可视化函数
# =============================================================================
def 绘制训练曲线(
    训练历史: Dict[str, List[float]],
    保存路径: str,
    显示验证: bool = True,
) -> None:
    """
    绘制训练曲线

    Args:
        训练历史: 训练历史字典
        保存路径: 图片保存路径
        显示验证: 是否显示验证曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(训练历史["train_loss"]) + 1)

    # 损失曲线
    axes[0].plot(epochs, 训练历史["train_loss"], "b-", label="Train Loss", linewidth=2)
    if 显示验证 and "val_loss" in 训练历史 and 训练历史["val_loss"]:
        axes[0].plot(epochs, 训练历史["val_loss"], "r--", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, 训练历史["train_acc"], "b-", label="Train Acc", linewidth=2)
    if 显示验证 and "val_acc" in 训练历史 and 训练历史["val_acc"]:
        axes[1].plot(epochs, 训练历史["val_acc"], "r--", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(保存路径, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {保存路径}")


def 绘制混淆矩阵(
    混淆矩阵: np.ndarray,
    类别名称: List[str],
    保存路径: str,
    归一化: bool = False,
) -> None:
    """
    绘制混淆矩阵

    Args:
        混淆矩阵: 混淆矩阵数组
        类别名称: 类别名称列表
        保存路径: 图片保存路径
        归一化: 是否归一化
    """
    if 归一化:
        混淆矩阵 = 混淆矩阵.astype("float") / 混淆矩阵.sum(axis=1)[:, np.newaxis]
        格式字符串 = ".2%"
    else:
        格式字符串 = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        混淆矩阵,
        annot=True,
        fmt=格式字符串,
        cmap="Blues",
        xticklabels=类别名称,
        yticklabels=类别名称,
        cbar_kws={"label": "Normalized Accuracy" if 归一化 else "Count"},
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix - 1D-CNN Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(保存路径, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {保存路径}")


def 绘制学习率曲线(训练历史: Dict[str, List[float]], 保存路径: str) -> None:
    """
    绘制学习率变化曲线

    Args:
        训练历史: 训练历史字典
        保存路径: 图片保存路径
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(训练历史["learning_rate"]) + 1)
    plt.plot(epochs, 训练历史["learning_rate"], "g-", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(保存路径, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"学习率曲线已保存: {保存路径}")


# =============================================================================
# 主训练函数
# =============================================================================
def 主训练函数(
    数据路径: str = None,
    样本长度: int = 1024,
    步长: int = 512,
    归一化方法: str = "minmax",
    训练集比例: float = 0.7,
    批次大小: int = 32,
    训练轮数: int = 50,
    学习率: float = 0.001,
    权重衰减: float = 1e-4,
    优化器类型: str = "adam",
    调度器类型: str = "step",
    调度器步长: int = 20,
    调度器gamma: float = 0.5,
    早停耐心值: int = 15,
    早停最小变化: float = 0.001,
    梯度裁剪阈值: float = None,
    随机种子: int = 42,
    日志间隔: int = 10,
    模型保存路径: str = None,
    使用GPU: bool = True,
) -> Tuple[nn.Module, Dict[str, float], Dict]:
    """
    主训练函数

    Args:
        数据路径: 数据目录路径
        样本长度: 每个样本的序列长度
        步长: 滑动窗口步长
        归一化方法: "minmax", "zscore", "none"
        训练集比例: 训练集比例
        批次大小: 批次大小
        训练轮数: 训练轮数
        学习率: 学习率
        权重衰减: L2 正则化系数
        优化器类型: "adam", "sgd", "adamw"
        调度器类型: "step", "cosine", "none"
        调度器步长: 学习率下降间隔
        调度器gamma: 学习率下降倍数
        早停耐心值: 早停容忍轮数
        早停最小变化: 被认为有提升的最小变化量
        梯度裁剪阈值: 梯度裁剪阈值，None表示不裁剪
        随机种子: 随机种子
        日志间隔: 日志打印间隔
        模型保存路径: 模型保存路径
        使用GPU: 是否使用 GPU

    Returns:
        (训练好的模型, 测试指标字典, 训练历史字典)
    """
    # ===== 固定随机种子 =====
    torch.manual_seed(随机种子)
    np.random.seed(随机种子)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(随机种子)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ===== 设备配置 =====
    if 使用GPU and torch.cuda.is_available():
        设备 = torch.device("cuda:0")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        设备 = torch.device("cpu")
        print("使用 CPU")

    # ===== 加载数据 =====
    print("\n" + "=" * 60)
    print("加载数据集")
    print("=" * 60)

    数据集 = CWRUDataset(
        data_dir=数据路径,
        sample_length=样本长度,
        stride=步长,
        normalize=归一化方法,
    )

    打印数据统计(数据集, "CWRU 轴承数据集")

    # 划分数据集
    训练集, 测试集 = 划分训练测试集(
        数据集,
        test_ratio=1 - 训练集比例,
        random_seed=随机种子,
        stratify=True,
    )

    print(f"训练集: {len(训练集)} 样本")
    print(f"测试集: {len(测试集)} 样本")

    # 创建数据加载器
    训练数据加载器 = DataLoader(
        训练集,
        batch_size=批次大小,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    测试数据加载器 = DataLoader(
        测试集,
        batch_size=批次大小,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ===== 构建模型 =====
    print("\n" + "=" * 60)
    print("构建模型")
    print("=" * 60)

    模型 = CNN1D(
        输入长度=样本长度,
        通道数=1,
        类别数=4,
        卷积核大小=16,
        第一层通道=32,
    ).to(设备)

    打印模型结构(模型, 输入长度=样本长度)

    # ===== 初始化训练器 =====
    训练器实例 = 训练器(
        模型=模型,
        设备=设备,
        学习率=学习率,
        权重衰减=权重衰减,
        优化器类型=优化器类型,
        调度器类型=调度器类型,
        调度器步长=调度器步长,
        调度器gamma=调度器gamma,
        早停耐心值=早停耐心值,
        早停最小变化=早停最小变化,
        梯度裁剪阈值=梯度裁剪阈值,
    )

    # ===== 训练循环 =====
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    开始时间 = datetime.now()
    最好验证准确率 = 0.0

    for 轮次 in range(1, 训练轮数 + 1):
        print(f"\nEpoch {轮次}/{训练轮数}")

        指标 = 训练器实例.训练轮次(
            训练数据加载器=训练数据加载器,
            验证数据加载器=None,  # 本 baseline 简化版不使用验证集
            轮次=轮次,
            日志间隔=日志间隔,
        )

        # 打印本轮指标
        print(
            f"  Train Loss: {指标['train_loss']:.4f} | "
            f"Train Acc: {指标['train_acc']:.2f}% | "
            f"LR: {指标['learning_rate']:.6f}"
        )

        # 保存最佳模型（基于训练准确率，因为没有验证集）
        if 指标['train_acc'] > 最好验证准确率:
            最好验证准确率 = 指标['train_acc']
            if 模型保存路径 is not None:
                torch.save(模型.state_dict(), 模型保存路径)
                print(f"  ★ 模型已保存到: {模型保存路径}")

        # 早停检查
        if 训练器实例.是否早停():
            print(f"\n早停触发: 连续 {训练器实例.早停计数器} 轮验证损失无改善")
            break

    训练时间 = (datetime.now() - 开始时间).total_seconds()

    # ===== 测试评估 =====
    print("\n" + "=" * 60)
    print("测试评估")
    print("=" * 60)

    # 加载最佳模型
    if 模型保存路径 is not None and os.path.exists(模型保存路径):
        模型.load_state_dict(torch.load(模型保存路径, map_location=设备))
        print(f"已加载最佳模型: {模型保存路径}")

    测试指标, 混淆矩阵, 预测标签, 真实标签 = 评估模型(
        模型=模型,
        测试数据加载器=测试数据加载器,
        设备=设备,
        类别名称=list(CLASS_NAMES.values()),
    )

    print(f"\n测试集准确率: {测试指标['accuracy']:.2f}%")
    print(f"精确率 (Precision): {测试指标['precision']:.2f}%")
    print(f"召回率 (Recall): {测试指标['recall']:.2f}%")
    print(f"F1 分数: {测试指标['f1_score']:.2f}%")

    # 打印分类报告（使用测试集的真实标签和预测标签）
    报告 = 打印分类报告(
        真实标签=真实标签,
        预测标签=预测标签,
        类别名称=list(CLASS_NAMES.values()),
    )

    # ===== 保存结果 =====
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
    实验名称 = f"cnn1d_baseline_{时间戳}"

    # 训练曲线
    训练曲线路径 = os.path.join(FIGURES_DIR, f"{实验名称}_training_curve.png")
    绘制训练曲线(训练器实例.训练历史, 训练曲线路径, 显示验证=False)

    # 混淆矩阵
    混淆矩阵路径 = os.path.join(FIGURES_DIR, f"{实验名称}_confusion_matrix.png")
    绘制混淆矩阵(混淆矩阵, list(CLASS_NAMES.values()), 混淆矩阵路径)

    # 学习率曲线
    学习率曲线路径 = os.path.join(FIGURES_DIR, f"{实验名称}_lr_schedule.png")
    绘制学习率曲线(训练器实例.训练历史, 学习率曲线路径)

    # 保存训练历史
    历史路径 = os.path.join(RESULTS_DIR, f"{实验名称}_history.json")
    训练器实例.保存训练历史(历史路径)

    # 保存指标
    指标.update({
        "model_name": "1D-CNN-Benchmark",
        "test_confusion_matrix": 混淆矩阵.tolist(),
        "training_time_seconds": 训练时间,
        "best_train_acc": 最好验证准确率,
        "epochs_trained": len(训练器实例.训练历史["train_loss"]),
    })

    指标路径 = os.path.join(RESULTS_DIR, f"{实验名称}_metrics.json")
    with open(指标路径, "w") as f:
        json.dump(指标, f, indent=2)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"总训练时间: {训练时间:.2f} 秒")
    print(f"训练轮数: {len(训练器实例.训练历史['train_loss'])}")
    print(f"模型保存: {模型保存路径}")
    print(f"结果保存目录: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    return 模型, 测试指标, 训练器实例.训练历史


# English alias
main = 主训练函数


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CWRU 轴承故障诊断 1D-CNN Baseline 训练脚本"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--sample_length", type=int, default=1024, help="样本长度")
    parser.add_argument("--stride", type=int, default=512, help="滑动窗口步长")
    parser.add_argument("--normalize", type=str, default="minmax", help="归一化方法")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--optimizer", type=str, default="adam", help="优化器")
    parser.add_argument("--scheduler", type=str, default="step", help="学习率调度器")
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--log_interval", type=int, default=10, help="日志间隔")
    parser.add_argument("--no_gpu", action="store_true", help="不使用 GPU")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=os.path.join(MODEL_CKPT_DIR, "cnn1d_baseline.pth"),
        help="模型保存路径",
    )

    args = parser.parse_args()

    # 运行训练
    主训练函数(
        数据路径=args.data_dir,
        样本长度=args.sample_length,
        步长=args.stride,
        归一化方法=args.normalize,
        训练集比例=args.train_ratio,
        批次大小=args.batch_size,
        训练轮数=args.epochs,
        学习率=args.lr,
        权重衰减=args.weight_decay,
        优化器类型=args.optimizer,
        调度器类型=args.scheduler,
        早停耐心值=args.patience,
        随机种子=args.seed,
        日志间隔=args.log_interval,
        模型保存路径=args.model_save_path,
        使用GPU=not args.no_gpu,
    )
