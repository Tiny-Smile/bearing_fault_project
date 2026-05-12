"""
1D-CNN 轴承故障诊断模型训练脚本

功能：
    1. 加载真实的 CWRU .mat 数据文件
    2. 构建 1D-CNN 模型
    3. 训练循环（含早停、学习率调度）
    4. 在测试集上评估模型
    5. 保存模型权重、训练曲线、混淆矩阵

使用方法：
    python train_cnn1d.py
    python train_cnn1d.py --data_dir p02_data/raw/cwru --epochs 30
    python train_cnn1d.py --epochs 50 --batch_size 64 --lr 0.0005
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
sys.path.insert(0, _current_dir)

from p01_utils.cwru_dataset import (
    CWRUDataset,
    split_train_test,
    create_dataloaders,
    print_dataset_statistics,
    CLASS_NAMES,
)
from p02_models.cnn1d import CNN1D, print_model_structure


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
# 训练器类（Trainer）
# =============================================================================
class Trainer:
    """
    1D-CNN 模型训练器

    负责：
    - 单轮训练（train_epoch）
    - 验证评估（_validate）
    - 学习率调度
    - 早停机制（Early Stopping）
    - 训练历史记录
    """

    def __init__(
        self,
        model: nn.Module,                       # PyTorch 模型
        device: torch.device,                   # 训练设备（cuda 或 cpu）
        learning_rate: float = 0.001,           # 初始学习率
        weight_decay: float = 1e-4,             # L2 正则化系数，防止过拟合
        optimizer_type: str = "adam",            # 优化器类型："adam" | "sgd" | "adamw"
        scheduler_type: str = "step",            # 学习率调度器："step" | "cosine" | "none"
        scheduler_step: int = 20,               # StepLR 的 step_size（每隔多少 epoch 下降一次）
        scheduler_gamma: float = 0.5,           # 学习率下降倍数（每 step 乘以此值）
        early_stopping_patience: int = 15,      # 早停容忍轮数（验证损失多少轮无改善则停止）
        early_stopping_min_delta: float = 0.001,# 被认为有提升的最小改善量
        grad_clip_threshold: Optional[float] = None,  # 梯度裁剪阈值，None 表示不裁剪
    ) -> None:
        """
        初始化训练器。

        Args:
            model: PyTorch 模型实例
            device: 训练设备
            learning_rate: 初始学习率
            weight_decay: L2 正则化系数
            optimizer_type: 优化器类型
            scheduler_type: 学习率调度器类型
            scheduler_step: 学习率下降间隔
            scheduler_gamma: 学习率下降倍数
            early_stopping_patience: 早停容忍轮数
            early_stopping_min_delta: 被认为有提升的最小变化量
            grad_clip_threshold: 梯度裁剪阈值
        """
        self.model = model
        self.device = device
        self.grad_clip_threshold = grad_clip_threshold

        # 损失函数：交叉熵（适用于多分类任务）
        self.criterion = nn.CrossEntropyLoss()

        # ---- 优化器配置 ----
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "adamw": 
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}，请使用 adam / sgd / adamw")

        # ---- 学习率调度器配置 ----
        if scheduler_type.lower() == "step":
            # 每 scheduler_step 个 epoch 将学习率乘以 scheduler_gamma
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_step,
                gamma=scheduler_gamma,
            )
        elif scheduler_type.lower() == "cosine":
            # 余弦退火调度
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=1e-6,
            )
        else:
            self.scheduler = None

        # ---- 早停配置 ----
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0   # 连续无改善的轮数计数器
        self.best_val_loss = float("inf")  # 最佳验证损失
        self.best_model_state: Optional[Dict] = None  # 最佳模型权重

        # 训练历史（用于绘制训练曲线）
        self.training_history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 1,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        训练一个 epoch。

        Args:
            train_loader: 训练数据迭代器
            val_loader: 验证数据迭代器（可选）
            epoch: 当前 epoch 编号
            log_interval: 打印日志的批次间隔

        Returns:
            包含 train_loss、train_acc、val_loss、val_acc、learning_rate 的字典
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            # ---- 前向传播 ----
            self.optimizer.zero_grad()
            outputs = self.model.forward(data)
            loss = self.criterion(outputs, labels)

            # ---- 反向传播 ----
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            if self.grad_clip_threshold is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_threshold,
                )

            self.optimizer.step()

            # ---- 统计 ----
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 打印日志
            if (batch_idx + 1) % log_interval == 0:
                current_acc = 100.0 * correct / total
                current_loss = total_loss / (batch_idx + 1)
                print(
                    f"  Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(train_loader):4d} | "
                    f"Loss: {current_loss:.4f} | Acc: {current_acc:.2f}%"
                )

        # ---- 计算本轮平均指标 ----
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # ---- 更新学习率（调度器 step） ----
        if self.scheduler is not None:
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
        else:
            current_lr = self.optimizer.param_groups[0]["lr"]

        # ---- 记录历史 ----
        self.training_history["train_loss"].append(avg_loss)
        self.training_history["train_acc"].append(train_acc)
        self.training_history["learning_rate"].append(current_lr)

        # ---- 验证 ----
        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = self._validate(val_loader)
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_acc"].append(val_metrics["acc"])

            # 早停检查：验证损失是否有改善
            if val_metrics["loss"] < self.best_val_loss - self.early_stopping_min_delta:
                self.best_val_loss = val_metrics["loss"]
                # 保存最佳模型权重（深拷贝，避免引用问题）
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.early_stopping_counter = 0
                print(f"  ★ Validation loss improved: {self.best_val_loss:.4f}")
            else:
                self.early_stopping_counter += 1

        return {
            "train_loss": avg_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics.get("loss", 0),
            "val_acc": val_metrics.get("acc", 0),
            "learning_rate": current_lr,
        }

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估模型（仅前向传播，不反向传播）。

        Args:
            val_loader: 验证数据迭代器

        Returns:
            包含 loss 和 acc 的字典
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model.forward(data)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(val_loader),
            "acc": 100.0 * correct / total,
        }

    def should_stop_early(self) -> bool:
        """检查是否应该早停（连续多轮验证损失无改善）。"""
        return self.early_stopping_counter >= self.early_stopping_patience

    def restore_best_model(self) -> None:
        """从 best_model_state 恢复模型权重到最优状态。"""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"  已恢复最佳模型 (val_loss={self.best_val_loss:.4f})")

    def save_training_history(self, save_path: str) -> None:
        """将训练历史字典保存为 JSON 文件。"""
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    在测试集上评估模型，计算 Accuracy、Precision、Recall、F1-Score。

    Args:
        model: 训练好的 PyTorch 模型
        test_loader: 测试数据迭代器
        device: 训练设备
        class_names: 类别名称列表，默认使用 ["Normal", "Inner_Race", "Outer_Race", "Ball"]

    Returns:
        (指标字典, 混淆矩阵, 预测标签数组, 真实标签数组)
    """
    if class_names is None:
        class_names = list(CLASS_NAMES.values())

    model.eval()
    all_true_labels: List[int] = []
    all_pred_labels: List[int] = []
    all_pred_probs: List[np.ndarray] = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model.forward(data)
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_true_labels.extend(labels.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_pred_probs.extend(probs.cpu().numpy())

    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)

    # ---- 计算评估指标 ----
    accuracy = 100.0 * accuracy_score(all_true_labels, all_pred_labels)
    precision = 100.0 * precision_score(
        all_true_labels, all_pred_labels, average="weighted", zero_division=0
    )
    recall = 100.0 * recall_score(
        all_true_labels, all_pred_labels, average="weighted", zero_division=0
    )
    f1 = 100.0 * f1_score(
        all_true_labels, all_pred_labels, average="weighted", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(all_true_labels, all_pred_labels)

    return metrics, cm, all_pred_labels, all_true_labels


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    打印 sklearn 的分类报告（Precision/Recall/F1 per class）。

    Args:
        y_true: 真实标签数组
        y_pred: 预测标签数组
        class_names: 类别名称列表

    Returns:
        分类报告字符串
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    print("\n" + "=" * 60)
    print("分类报告 (Classification Report)")
    print("=" * 60)
    print(report)
    return report


# =============================================================================
# 可视化函数
# =============================================================================

def plot_training_curve(
    history: Dict[str, List[float]],
    save_path: str,
    show_val: bool = True,
) -> None:
    """
    绘制训练曲线（Loss 和 Accuracy 随 Epoch 变化）。

    Args:
        history: 训练历史字典
        save_path: 图片保存路径
        show_val: 是否显示验证曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # 左图：损失曲线
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if show_val and "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 右图：准确率曲线
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    if show_val and "val_acc" in history and history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], "r--", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Training and Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存: {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = False,
) -> None:
    """
    绘制混淆矩阵热力图。

    Args:
        cm: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 图片保存路径
        normalize: 是否归一化（显示百分比而非数量）
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt_str = ".2%"
    else:
        fmt_str = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt_str,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Accuracy" if normalize else "Count"},
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix - 1D-CNN Baseline", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")


def plot_learning_rate_curve(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """
    绘制学习率变化曲线。

    Args:
        history: 训练历史字典
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history["learning_rate"]) + 1)
    plt.plot(epochs, history["learning_rate"], "g-", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"学习率曲线已保存: {save_path}")


# =============================================================================
# 主训练函数
# =============================================================================

def train(
    data_path: Optional[str] = None,
    sample_length: int = 1024,
    stride: int = 512,
    normalize: str = "minmax",
    train_ratio: float = 0.7,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adam",
    scheduler_type: str = "step",
    scheduler_step: int = 20,
    scheduler_gamma: float = 0.5,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 0.001,
    grad_clip_threshold: Optional[float] = None,
    random_seed: int = 42,
    log_interval: int = 10,
    model_save_path: Optional[str] = None,
    use_gpu: bool = True,
) -> Tuple[nn.Module, Dict[str, float], Dict]:
    """
    主训练函数：加载数据 → 构建模型 → 训练 → 评估 → 保存结果。

    Args:
        data_path: 数据目录路径，默认使用 p02_data/raw/cwru/
        sample_length: 每个样本的序列长度（点数），默认 1024
        stride: 滑动窗口步长，默认 512（50% 重叠）
        normalize: 归一化方法，"minmax" | "zscore" | "none"
        train_ratio: 训练集比例，默认 0.7（70% 训练，30% 测试）
        batch_size: 每批次样本数，默认 32
        epochs: 训练轮数，默认 50
        learning_rate: 初始学习率，默认 0.001
        weight_decay: L2 正则化系数，默认 1e-4
        optimizer_type: 优化器类型，默认 "adam"
        scheduler_type: 学习率调度器类型，默认 "step"
        scheduler_step: 学习率下降间隔，默认 20
        scheduler_gamma: 学习率下降倍数，默认 0.5
        early_stopping_patience: 早停容忍轮数，默认 15
        early_stopping_min_delta: 被认为有提升的最小变化量
        grad_clip_threshold: 梯度裁剪阈值
        random_seed: 随机种子，默认 42
        log_interval: 日志打印间隔（每多少批次打印一次）
        model_save_path: 模型权重保存路径
        use_gpu: 是否优先使用 GPU

    Returns:
        (训练好的模型, 测试指标字典, 训练历史字典)
    """
    # ===== 固定随机种子（确保结果可复现）=====
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ===== 设备配置 =====
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # ===== 加载数据集（真实 CWRU .mat 文件）=====
    print("\n" + "=" * 60)
    print("加载数据集")
    print("=" * 60)

    dataset = CWRUDataset(
        data_dir=data_path,
        sample_length=sample_length,
        stride=stride,
        normalize=normalize,
    )

    print_dataset_statistics(dataset, "CWRU 轴承数据集")

    # 划分训练集 / 测试集
    train_set, test_set = split_train_test(
        dataset,
        test_ratio=1 - train_ratio,
        random_seed=random_seed,
        stratify=True,
    )

    print(f"训练集: {len(train_set)} 样本")
    print(f"测试集: {len(test_set)} 样本")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ===== 构建模型 =====
    print("\n" + "=" * 60)
    print("构建模型")
    print("=" * 60)

    model = CNN1D(
        input_length=sample_length,
        in_channels=1,
        num_classes=4,
        kernel_size=16,
        first_channels=32,
    ).to(device)

    print_model_structure(model, input_length=sample_length)

    # ===== 初始化训练器 =====
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        scheduler_step=scheduler_step,
        scheduler_gamma=scheduler_gamma,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        grad_clip_threshold=grad_clip_threshold,
    )

    # ===== 训练循环 =====
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    start_time = datetime.now()
    best_train_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        metrics = trainer.train_epoch(
            train_loader=train_loader,
            val_loader=None,  # 本 baseline 不使用验证集
            epoch=epoch,
            log_interval=log_interval,
        )

        # 打印本轮指标
        print(
            f"  Train Loss: {metrics['train_loss']:.4f} | "
            f"Train Acc: {metrics['train_acc']:.2f}% | "
            f"LR: {metrics['learning_rate']:.6f}"
        )

        # 保存最佳模型（基于训练准确率）
        if metrics['train_acc'] > best_train_acc:
            best_train_acc = metrics['train_acc']
            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
                print(f"  ★ 模型已保存到: {model_save_path}")

        # 早停检查
        if trainer.should_stop_early():
            print(f"\n早停触发: 连续 {trainer.early_stopping_counter} 轮验证损失无改善")
            break

    training_time = (datetime.now() - start_time).total_seconds()

    # ===== 测试评估 =====
    print("\n" + "=" * 60)
    print("测试评估")
    print("=" * 60)

    # 加载最佳模型权重
    if model_save_path is not None and os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"已加载最佳模型: {model_save_path}")

    test_metrics, cm, pred_labels, true_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=list(CLASS_NAMES.values()),
    )

    print(f"\n测试集准确率 (Accuracy): {test_metrics['accuracy']:.2f}%")
    print(f"精确率 (Precision): {test_metrics['precision']:.2f}%")
    print(f"召回率 (Recall): {test_metrics['recall']:.2f}%")
    print(f"F1 分数: {test_metrics['f1_score']:.2f}%")

    # 打印详细分类报告
    report = print_classification_report(
        y_true=true_labels,
        y_pred=pred_labels,
        class_names=list(CLASS_NAMES.values()),
    )

    # ===== 保存结果 ======
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cnn1d_baseline_{timestamp}"

    # 训练曲线
    training_curve_path = os.path.join(FIGURES_DIR, f"{experiment_name}_training_curve.png")
    plot_training_curve(trainer.training_history, training_curve_path, show_val=False)

    # 混淆矩阵
    cm_path = os.path.join(FIGURES_DIR, f"{experiment_name}_confusion_matrix.png")
    plot_confusion_matrix(cm, list(CLASS_NAMES.values()), cm_path)

    # 学习率曲线
    lr_curve_path = os.path.join(FIGURES_DIR, f"{experiment_name}_lr_schedule.png")
    plot_learning_rate_curve(trainer.training_history, lr_curve_path)

    # 保存训练历史 JSON
    history_path = os.path.join(RESULTS_DIR, f"{experiment_name}_history.json")
    trainer.save_training_history(history_path)

    # 保存测试指标 JSON
    test_metrics.update({
        "model_name": "1D-CNN-Benchmark",
        "test_confusion_matrix": cm.tolist(),
        "training_time_seconds": training_time,
        "best_train_acc": best_train_acc,
        "epochs_trained": len(trainer.training_history["train_loss"]),
    })

    metrics_path = os.path.join(RESULTS_DIR, f"{experiment_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"训练轮数: {len(trainer.training_history['train_loss'])}")
    print(f"模型保存: {model_save_path}")
    print(f"结果保存目录: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    return model, test_metrics, trainer.training_history


# 别名（兼容旧代码）
main_training_function = train


# =============================================================================
# 命令行入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CWRU 轴承故障诊断 1D-CNN Baseline 训练脚本"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--sample_length", type=int, default=1024, help="样本长度（点数）")
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
    train(
        data_path=args.data_dir,
        sample_length=args.sample_length,
        stride=args.stride,
        normalize=args.normalize,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
        log_interval=args.log_interval,
        model_save_path=args.model_save_path,
        use_gpu=not args.no_gpu,
    )
