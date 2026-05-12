"""
1D-CNN 模型评估与可视化脚本

功能：
    1. 加载训练好的模型权重
    2. 在测试集上评估模型
    3. 生成可视化结果：
       - 混淆矩阵（原始计数 + 归一化）
       - 每类别准确率条形图
       - t-SNE 特征可视化（可选）
    4. 保存评估结果到 JSON

使用方法：
    python eval_cnn1d.py --model_path p04_models_ckpt/cnn1d_baseline.pth
    python eval_cnn1d.py --model_path p04_models_ckpt/cnn1d_baseline.pth --data_dir p02_data/raw/cwru
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
_code_dir = os.path.dirname(_current_dir)  # p03_code 目录
sys.path.insert(0, _current_dir)
sys.path.insert(0, _code_dir)

from p01_utils.cwru_dataset import (
    CWRUDataset,
    split_train_test,
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
# 评估器类（Evaluator）
# =============================================================================
class Evaluator:
    """
    模型评估器

    负责：
    - 加载模型权重
    - 在测试集上推理
    - 计算评估指标（Acc / P / R / F1）
    - 生成可视化图表
    - 保存评估结果
    """

    def __init__(
        self,
        model_path: str,                      # 模型权重文件路径
        data_path: Optional[str] = None,       # 数据目录路径
        sample_length: int = 1024,            # 样本序列长度
        stride: int = 512,                   # 滑动窗口步长
        normalize: str = "minmax",            # 归一化方法
        train_ratio: float = 0.7,            # 训练集比例
        batch_size: int = 32,                # 批次大小
        random_seed: int = 42,               # 随机种子
        device: str = "auto",                # 设备类型，"auto" | "cuda" | "cpu"
    ) -> None:
        """
        初始化评估器。

        Args:
            model_path: 模型权重文件路径（.pth 文件）
            data_path: 数据目录路径
            sample_length: 样本序列长度
            stride: 滑动窗口步长
            normalize: 归一化方法
            train_ratio: 训练集比例
            batch_size: 批次大小
            random_seed: 随机种子
            device: 设备类型
        """
        self.model_path = model_path
        self.sample_length = sample_length
        self.batch_size = batch_size

        # ---- 设备配置 ----
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- 加载数据集 ----
        self.dataset = CWRUDataset(
            data_dir=data_path,
            sample_length=sample_length,
            stride=stride,
            normalize=normalize,
        )

        # 划分数据集（使用与训练时相同的 random_seed 和 train_ratio）
        _, self.test_set = split_train_test(
            self.dataset,
            test_ratio=1 - train_ratio,
            random_seed=random_seed,
            stratify=True,
        )

        # 创建测试数据加载器
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # ---- 加载模型 ----
        self.model = self._load_model()

        # ---- 预测结果（延迟初始化，在 evaluate() 后填充）----
        self.all_true_labels: Optional[np.ndarray] = None
        self.all_pred_labels: Optional[np.ndarray] = None
        self.all_pred_probs: Optional[np.ndarray] = None
        self.confusion_matrix: Optional[np.ndarray] = None
        self.metrics: Optional[Dict] = None

    def _load_model(self) -> nn.Module:
        """
        加载模型权重。

        Returns:
            加载了权重的 CNN1D 模型实例
        """
        model = CNN1D(
            input_length=self.sample_length,
            in_channels=1,
            num_classes=4,
        ).to(self.device)

        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"模型已加载: {self.model_path}")
        else:
            print(f"[警告] 模型文件不存在: {self.model_path}")
            print("将使用随机初始化的模型进行评估（结果无参考意义）")

        return model

    def evaluate(self) -> Dict[str, float]:
        """
        在测试集上对模型进行完整评估。

        评估流程：
        1. 前向传播（不反向传播，不更新梯度）
        2. 收集所有预测结果
        3. 计算 Accuracy / Precision / Recall / F1
        4. 生成混淆矩阵

        Returns:
            评估指标字典 {"accuracy", "precision", "recall", "f1_score"}
        """
        self.model.eval()
        all_true_labels: List[int] = []
        all_pred_labels: List[int] = []
        all_pred_probs: List[np.ndarray] = []

        print("\n正在评估模型...")
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(self.test_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)

                all_true_labels.extend(labels.cpu().numpy().tolist())
                all_pred_labels.extend(preds.cpu().numpy().tolist())
                all_pred_probs.extend(probs.cpu().numpy())

                if (batch_idx + 1) % 20 == 0:
                    print(f"  批次 {batch_idx + 1}/{len(self.test_loader)}")

        self.all_true_labels = np.array(all_true_labels)
        self.all_pred_labels = np.array(all_pred_labels)
        self.all_pred_probs = np.array(all_pred_probs)

        # ---- 计算评估指标 ----
        accuracy = 100.0 * accuracy_score(self.all_true_labels, self.all_pred_labels)
        precision = 100.0 * precision_score(
            self.all_true_labels, self.all_pred_labels, average="weighted", zero_division=0
        )
        recall = 100.0 * recall_score(
            self.all_true_labels, self.all_pred_labels, average="weighted", zero_division=0
        )
        f1 = 100.0 * f1_score(
            self.all_true_labels, self.all_pred_labels, average="weighted", zero_division=0
        )

        self.metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        # ---- 混淆矩阵 ----
        self.confusion_matrix = confusion_matrix(all_true_labels, all_pred_labels)

        return self.metrics

    def print_results(self) -> str:
        """
        打印评估结果和详细分类报告。

        Returns:
            分类报告字符串
        """
        if self.metrics is None:
            raise ValueError("请先调用 evaluate() 方法")

        print("\n" + "=" * 60)
        print("评估结果 (Evaluation Results)")
        print("=" * 60)

        print(f"准确率 (Accuracy): {self.metrics['accuracy']:.2f}%")
        print(f"精确率 (Precision): {self.metrics['precision']:.2f}%")
        print(f"召回率 (Recall): {self.metrics['recall']:.2f}%")
        print(f"F1 分数: {self.metrics['f1_score']:.2f}%")

        # 详细分类报告
        report = classification_report(
            self.all_true_labels,
            self.all_pred_labels,
            target_names=list(CLASS_NAMES.values()),
            digits=4,
        )
        print("\n详细分类报告:")
        print("-" * 60)
        print(report)

        return report

    def save_results(self, output_dir: str, experiment_name: str = None) -> None:
        """
        保存评估结果到文件。

        Args:
            output_dir: 输出目录路径
            experiment_name: 实验名称（用于文件名前缀）
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ---- 保存指标 JSON ----
        metrics_file = os.path.join(output_dir, f"{experiment_name}_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"指标已保存: {metrics_file}")

        # ---- 保存混淆矩阵 JSON ----
        cm_file = os.path.join(output_dir, f"{experiment_name}_confusion_matrix.json")
        with open(cm_file, "w", encoding="utf-8") as f:
            json.dump(self.confusion_matrix.tolist(), f, indent=2)
        print(f"混淆矩阵已保存: {cm_file}")

        # ---- 生成可视化图表 ----
        self._plot_confusion_matrix(
            os.path.join(FIGURES_DIR, f"{experiment_name}_confusion_matrix.png")
        )
        self._plot_per_class_accuracy(
            os.path.join(FIGURES_DIR, f"{experiment_name}_per_class_accuracy.png")
        )

    def _plot_confusion_matrix(self, save_path: str, normalize: bool = True) -> None:
        """
        绘制混淆矩阵（原始计数 + 归一化双图）。

        Args:
            save_path: 图片保存路径
            normalize: 是否显示归一化版本
        """
        if self.confusion_matrix is None:
            raise ValueError("请先调用 evaluate() 方法")

        cm = self.confusion_matrix
        if normalize:
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt_str = ".2%"
        else:
            cm_normalized = cm
            fmt_str = "d"

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：原始计数
        sns.heatmap(
            self.confusion_matrix,
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

        # 右图：归一化
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=fmt_str,
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")

    def _plot_per_class_accuracy(self, save_path: str) -> None:
        """
        绘制每类别准确率条形图。

        Args:
            save_path: 图片保存路径
        """
        if self.confusion_matrix is None:
            raise ValueError("请先调用 evaluate() 方法")

        # 计算每类准确率 = 对角线元素 / 该类总样本数
        per_class_acc = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)
        per_class_acc = 100.0 * per_class_acc

        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(range(4), per_class_acc, color=colors)
        plt.xlabel('故障类型 (Fault Type)', fontsize=12)
        plt.ylabel('准确率 (Accuracy %)', fontsize=12)
        plt.title('每类别准确率 (Per-Class Accuracy)', fontsize=14)
        plt.xticks(range(4), list(CLASS_NAMES.values()), rotation=15)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)

        # 在柱状图上标注数值
        for bar, acc in zip(bars, per_class_acc):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"每类准确率图已保存: {save_path}")


# =============================================================================
# 便捷函数
# =============================================================================

def quick_evaluate(
    model_path: str,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    快速评估模型（一行代码调用）。

    Args:
        model_path: 模型权重文件路径
        data_path: 数据目录路径
        output_dir: 输出目录路径

    Returns:
        评估指标字典
    """
    evaluator = Evaluator(
        model_path=model_path,
        data_path=data_path,
    )

    evaluator.evaluate()
    evaluator.print_results()
    evaluator.save_results(
        output_dir=output_dir if output_dir else RESULTS_DIR,
    )

    return evaluator.metrics


# 别名（兼容旧代码）
quick_eval = quick_evaluate


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
        default=os.path.join(PROJECT_ROOT, "p04_models_ckpt", "cnn1d_baseline.pth"),
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
        help="样本长度（点数）",
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
        help="实验名称（用于文件名前缀）",
    )

    args = parser.parse_args()

    # 创建评估器
    evaluator = Evaluator(
        model_path=args.model_path,
        data_path=args.data_dir,
        sample_length=args.sample_length,
    )

    # 评估
    evaluator.evaluate()
    evaluator.print_results()

    # 保存结果
    evaluator.save_results(
        output_dir=args.output,
        experiment_name=args.experiment_name,
    )
