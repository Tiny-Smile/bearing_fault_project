"""
1D-CNN 模型评估与可视化脚本

功能：
1. 加载训练好的模型
2. 在测试集上进行评估
3. 生成可视化结果：
   - 训练曲线
   - 混淆矩阵
   - 分类报告
   - t-SNE 特征可视化（可选）

使用方法：
    python eval_cnn1d.py --model_path p04_models_ckpt/cnn1d_baseline.pth
    python eval_cnn1d.py --model_path p04_models_ckpt/cnn1d_baseline.pth --output results/
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
from torch.utils.data import DataLoader
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
    划分训练测试集,
    CLASS_NAMES,
)
from p02_models.cnn1d import CNN1D


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

RESULTS_DIR = os.path.join(PROJECT_ROOT, "p06_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# =============================================================================
# 评估器类
# =============================================================================
class 评估器:
    """
    模型评估器

    负责：
    - 模型加载
    - 测试集评估
    - 指标计算
    - 结果可视化
    """

    def __init__(
        self,
        模型路径: str,
        数据路径: str = None,
        样本长度: int = 1024,
        步长: int = 512,
        归一化方法: str = "minmax",
        训练集比例: float = 0.7,
        批次大小: int = 32,
        随机种子: int = 42,
        设备: str = "auto",
    ) -> None:
        """
        初始化评估器

        Args:
            模型路径: 模型权重文件路径
            数据路径: 数据目录路径
            样本长度: 样本序列长度
            步长: 滑动窗口步长
            归一化方法: 归一化方法
            训练集比例: 训练集比例
            批次大小: 批次大小
            随机种子: 随机种子
            设备: 设备类型，"auto", "cuda", "cpu"
        """
        self.模型路径 = 模型路径
        self.样本长度 = 样本长度
        self.批次大小 = 批次大小

        # 设备配置
        if 设备 == "auto":
            self.设备 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.设备 = torch.device(设备)

        # 加载数据集
        self.数据集 = CWRUDataset(
            data_dir=数据路径,
            sample_length=样本长度,
            stride=步长,
            normalize=归一化方法,
        )

        # 划分数据集
        _, self.测试集 = 划分训练测试集(
            self.数据集,
            test_ratio=1 - 训练集比例,
            random_seed=随机种子,
            stratify=True,
        )

        # 创建测试数据加载器
        self.测试数据加载器 = DataLoader(
            self.测试集,
            batch_size=批次大小,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 加载模型
        self.模型 = self._加载模型()

        # 预测结果
        self.所有真实标签: Optional[np.ndarray] = None
        self.所有预测标签: Optional[np.ndarray] = None
        self.所有预测概率: Optional[np.ndarray] = None
        self.混淆矩阵: Optional[np.ndarray] = None
        self.指标: Optional[Dict] = None

    def _加载模型(self) -> nn.Module:
        """
        加载模型

        Returns:
            加载好的模型
        """
        模型 = CNN1D(
            输入长度=self.样本长度,
            通道数=1,
            类别数=4,
        ).to(self.设备)

        if os.path.exists(self.模型路径):
            模型.load_state_dict(torch.load(self.模型路径, map_location=self.设备))
            print(f"模型已加载: {self.模型路径}")
        else:
            print(f"[警告] 模型文件不存在: {self.模型路径}")
            print("将使用随机初始化的模型进行评估")

        return 模型

    def 评估(self) -> Dict[str, float]:
        """
        在测试集上评估模型

        Returns:
            评估指标字典
        """
        self.模型.eval()
        所有真实标签 = []
        所有预测标签 = []
        所有预测概率 = []

        print("\n正在评估模型...")
        with torch.no_grad():
            for 批次索引, (数据, 标签) in enumerate(self.测试数据加载器):
                数据 = 数据.to(self.设备)
                标签 = 标签.to(self.设备)

                输出 = self.模型.前向传播(数据)
                预测 = 输出.argmax(dim=1)
                概率 = torch.softmax(输出, dim=1)

                所有真实标签.extend(标签.cpu().numpy())
                所有预测标签.extend(预测.cpu().numpy())
                所有预测概率.extend(概率.cpu().numpy())

                if (批次索引 + 1) % 20 == 0:
                    print(f"  批次 {批次索引 + 1}/{len(self.测试数据加载器)}")

        self.所有真实标签 = np.array(所有真实标签)
        self.所有预测标签 = np.array(所有预测标签)
        self.所有预测概率 = np.array(所有预测概率)

        # 计算指标
        self.指标 = {
            "accuracy": 100.0 * accuracy_score(所有真实标签, 所有预测标签),
            "precision": 100.0 * precision_score(
                所有真实标签, 所有预测标签, average="weighted", zero_division=0
            ),
            "recall": 100.0 * recall_score(
                所有真实标签, 所有预测标签, average="weighted", zero_division=0
            ),
            "f1_score": 100.0 * f1_score(
                所有真实标签, 所有预测标签, average="weighted", zero_division=0
            ),
        }

        # 混淆矩阵
        self.混淆矩阵 = confusion_matrix(所有真实标签, 所有预测标签)

        return self.指标

    def 打印结果(self) -> str:
        """
        打印评估结果

        Returns:
            分类报告字符串
        """
        if self.指标 is None:
            raise ValueError("请先调用评估()方法")

        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"准确率 (Accuracy): {self.指标['accuracy']:.2f}%")
        print(f"精确率 (Precision): {self.指标['precision']:.2f}%")
        print(f"召回率 (Recall): {self.指标['recall']:.2f}%")
        print(f"F1 分数 (F1-Score): {self.指标['f1_score']:.2f}%")

        print("\n" + "-" * 60)
        print("混淆矩阵:")
        print("-" * 60)
        print(f"{'':15s}", end="")
        for name in CLASS_NAMES.values():
            print(f"{name:>12s}", end="")
        print()
        for i, name in enumerate(CLASS_NAMES.values()):
            print(f"{name:15s}", end="")
            for j in range(len(CLASS_NAMES)):
                print(f"{self.混淆矩阵[i, j]:>12d}", end="")
            print()

        print("\n" + "-" * 60)
        报告 = classification_report(
            self.所有真实标签,
            self.所有预测标签,
            target_names=list(CLASS_NAMES.values()),
            digits=4,
        )
        print("详细分类报告:")
        print("-" * 60)
        print(报告)

        return 报告

    def 保存结果(self, 输出目录: str, 实验名称: str = None) -> None:
        """
        保存评估结果

        Args:
            输出目录: 输出目录路径
            实验名称: 实验名称（用于文件名前缀）
        """
        os.makedirs(输出目录, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        if 实验名称 is None:
            实验名称 = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存指标
        指标文件 = os.path.join(输出目录, f"{实验名称}_metrics.json")
        with open(指标文件, "w", encoding="utf-8") as f:
            json.dump(self.指标, f, indent=2, ensure_ascii=False)
        print(f"指标已保存: {指标文件}")

        # 保存混淆矩阵
        混淆矩阵文件 = os.path.join(输出目录, f"{实验名称}_confusion_matrix.json")
        with open(混淆矩阵文件, "w", encoding="utf-8") as f:
            json.dump({
                "matrix": self.混淆矩阵.tolist(),
                "class_names": list(CLASS_NAMES.values()),
            }, f, indent=2, ensure_ascii=False)
        print(f"混淆矩阵已保存: {混淆矩阵文件}")

        # 生成并保存可视化
        self._绘制混淆矩阵(
            os.path.join(FIGURES_DIR, f"{实验名称}_confusion_matrix.png")
        )
        self._绘制每类别准确率(
            os.path.join(FIGURES_DIR, f"{实验名称}_per_class_accuracy.png")
        )

    def _绘制混淆矩阵(self, 保存路径: str, 归一化: bool = True) -> None:
        """
        绘制混淆矩阵

        Args:
            保存路径: 图片保存路径
            归一化: 是否归一化
        """
        if self.混淆矩阵 is None:
            raise ValueError("请先调用评估()方法")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 原始混淆矩阵
        sns.heatmap(
            self.混淆矩阵,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()),
            ax=axes[0],
        )
        axes[0].set_xlabel("Predicted Label", fontsize=11)
        axes[0].set_ylabel("True Label", fontsize=11)
        axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=12)

        # 归一化混淆矩阵
        混淆矩阵_归一化 = self.混淆矩阵.astype("float") / self.混淆矩阵.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            混淆矩阵_归一化,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()),
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted Label", fontsize=11)
        axes[1].set_ylabel("True Label", fontsize=11)
        axes[1].set_title("Confusion Matrix (Normalized)", fontsize=12)

        plt.suptitle("1D-CNN Baseline - CWRU Bearing Fault Diagnosis", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(保存路径, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存: {保存路径}")

    def _绘制每类别准确率(self, 保存路径: str) -> None:
        """
        绘制每类别准确率条形图

        Args:
            保存路径: 图片保存路径
        """
        if self.混淆矩阵 is None:
            raise ValueError("请先调用评估()方法")

        # 计算每类准确率
        每类准确率 = self.混淆矩阵.diagonal() / self.混淆矩阵.sum(axis=1) * 100

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            list(CLASS_NAMES.values()),
            每类准确率,
            color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"],
            edgecolor="black",
            linewidth=1.5,
        )

        # 添加数值标签
        for bar, acc in zip(bars, 每类准确率):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        plt.ylim(0, 110)
        plt.xlabel("Fault Type", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Per-Class Accuracy - 1D-CNN Baseline", fontsize=14)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(保存路径, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"每类准确率图已保存: {保存路径}")


# =============================================================================
# 便捷函数
# =============================================================================
def 快速评估(
    模型路径: str,
    数据路径: str = None,
    输出目录: str = None,
) -> Dict[str, float]:
    """
    快速评估模型

    Args:
        模型路径: 模型权重文件路径
        数据路径: 数据目录路径
        输出目录: 输出目录路径

    Returns:
        评估指标字典
    """
    评估器实例 = 评估器(
        模型路径=模型路径,
        数据路径=数据路径,
    )

    评估器实例.评估()
    评估器实例.打印结果()

    if 输出目录 is not None:
        评估器实例.保存结果(输出目录)

    return 评估器实例.指标


# English alias
quick_eval = 快速评估


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1D-CNN 模型评估脚本"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "04_models_ckpt", "cnn1d_baseline.pth"),
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="数据目录路径",
    )
    parser.add_argument(
        "--sample_length",
        type=int,
        default=1024,
        help="样本长度",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=RESULTS_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="实验名称",
    )

    args = parser.parse_args()

    # 创建评估器
    评估器实例 = 评估器(
        模型路径=args.model_path,
        数据路径=args.data_dir,
        样本长度=args.sample_length,
    )

    # 评估
    评估器实例.评估()
    评估器实例.打印结果()

    # 保存结果
    评估器实例.保存结果(
        输出目录=args.output,
        实验名称=args.experiment_name,
    )
