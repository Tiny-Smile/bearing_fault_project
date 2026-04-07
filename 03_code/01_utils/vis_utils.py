"""
轴承故障诊断可视化工具模块
作者：轴承故障诊断项目
功能：提供混淆矩阵、训练曲线等可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import os


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(cm: np.ndarray, 
                      save_path: str,
                      class_names: Optional[List[str]] = None,
                      title: str = "混淆矩阵") -> None:
    """
    绘制混淆矩阵热力图
    
    Args:
        cm (np.ndarray): 混淆矩阵，形状为 (n_classes, n_classes)
        save_path (str): 保存路径
        class_names (Optional[List[str]]): 类别名称列表
        title (str): 图表标题
        
    Example:
        >>> cm = np.array([[50, 2, 1, 0], [1, 48, 1, 0], [0, 1, 47, 2], [0, 0, 1, 49]])
        >>> plot_confusion_matrix(cm, "./confusion_matrix.png")
    """
    if class_names is None:
        class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数量'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    
    # 调整标签旋转角度
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"混淆矩阵已保存到: {save_path}")


def plot_train_curve(train_loss: List[float],
                   val_loss: List[float],
                   train_acc: List[float],
                   val_acc: List[float],
                   save_path: str,
                   title: str = "训练过程曲线") -> None:
    """
    绘制训练过程中的损失和准确率曲线
    
    Args:
        train_loss (List[float]): 训练损失列表
        val_loss (List[float]): 验证损失列表
        train_acc (List[float]): 训练准确率列表
        val_acc (List[float]): 验证准确率列表
        save_path (str): 保存路径
        title (str): 图表标题
        
    Example:
        >>> train_loss = [2.5, 2.0, 1.5, 1.0, 0.8]
        >>> val_loss = [2.6, 2.1, 1.6, 1.1, 0.9]
        >>> plot_train_curve(train_loss, val_loss, train_acc, val_acc, "./training_curves.png")
    """
    epochs = range(1, len(train_loss) + 1)
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制损失曲线
    ax1.plot(epochs, train_loss, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='验证损失', linewidth=2)
    ax1.set_title('损失曲线', fontsize=14)
    ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_acc, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='验证准确率', linewidth=2)
    ax2.set_title('准确率曲线', fontsize=14)
    ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax2.set_ylabel('准确率', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 设置y轴范围为0-1
    ax2.set_ylim([0, 1.05])
    
    plt.suptitle(title, fontsize=16, y=1.02)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")


def plot_class_metrics(per_class_metrics: dict, 
                    save_path: str,
                    title: str = "各类别性能指标") -> None:
    """
    绘制各类别的性能指标柱状图
    
    Args:
        per_class_metrics (dict): 各类别指标字典
        save_path (str): 保存路径
        title (str): 图表标题
        
    Example:
        >>> metrics = {
        ...     '正常': {'precision': 0.95, 'recall': 0.90, 'f1_score': 0.92},
        ...     '内圈故障': {'precision': 0.88, 'recall': 0.91, 'f1_score': 0.89}
        ... }
        >>> plot_class_metrics(metrics, "./class_metrics.png")
    """
    class_names = list(per_class_metrics.keys())
    precision = [per_class_metrics[cls]['precision'] for cls in class_names]
    recall = [per_class_metrics[cls]['recall'] for cls in class_names]
    f1_scores = [per_class_metrics[cls]['f1_score'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制柱状图
    bars1 = ax.bar(x - width, precision, width, label='精确率', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, recall, width, label='召回率', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1分数', alpha=0.8, color='lightgreen')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('故障类别', fontsize=14)
    ax.set_ylabel('指标值', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 在柱状图上显示数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"类别指标图已保存到: {save_path}")


def plot_cwt_samples(cwt_data: np.ndarray, 
                   labels: np.ndarray,
                   save_path: str,
                   n_samples: int = 6,
                   title: str = "CWT时频图样本") -> None:
    """
    绘制CWT时频图样本
    
    Args:
        cwt_data (np.ndarray): CWT数据，形状为 (n_samples, n_channels, height, width)
        labels (np.ndarray): 对应的标签
        save_path (str): 保存路径
        n_samples (int): 要显示的样本数量
        title (str): 图表标题
        
    Example:
        >>> cwt_data = np.random.randn(10, 3, 64, 2048)
        >>> labels = np.random.randint(0, 4, 10)
        >>> plot_cwt_samples(cwt_data, labels, "./cwt_samples.png")
    """
    class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    
    # 随机选择样本
    indices = np.random.choice(len(cwt_data), min(n_samples, len(cwt_data)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        # 取第一个通道显示
        sample = cwt_data[idx, 0]  # 第一通道
        
        # 绘制时频图
        im = axes[i].imshow(sample, aspect='auto', cmap='viridis')
        axes[i].set_title(f'{class_names[labels[idx]]} (样本{idx+1})', fontsize=12)
        axes[i].set_xlabel('时间点', fontsize=10)
        axes[i].set_ylabel('频率尺度', fontsize=10)
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"CWT样本图已保存到: {save_path}")


def create_experiment_report(metrics_dict: dict, 
                        save_dir: str = "./06_results/figures/") -> None:
    """
    创建实验报告的综合可视化
    
    Args:
        metrics_dict (dict): 包含所有指标的字典
        save_dir (str): 保存目录
        
    Example:
        >>> metrics = {
        ...     'accuracy': 0.95,
        ...     'f1_macro': 0.93,
        ...     'confusion_matrix': cm,
        ...     'per_class_metrics': per_class
        ... }
        >>> create_experiment_report(metrics, "./experiment_results/")
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 绘制混淆矩阵
    if 'confusion_matrix' in metrics_dict:
        plot_confusion_matrix(
            metrics_dict['confusion_matrix'],
            os.path.join(save_dir, "confusion_matrix.png"),
            title="混淆矩阵"
        )
    
    # 2. 绘制各类别指标
    if 'per_class_metrics' in metrics_dict:
        plot_class_metrics(
            metrics_dict['per_class_metrics'],
            os.path.join(save_dir, "class_metrics.png"),
            title="各类别性能指标"
        )
    
    # 3. 绘制训练曲线（如果有）
    if 'train_loss' in metrics_dict:
        plot_train_curve(
            metrics_dict['train_loss'],
            metrics_dict['val_loss'],
            metrics_dict['train_acc'],
            metrics_dict['val_acc'],
            os.path.join(save_dir, "training_curves.png"),
            title="训练过程曲线"
        )
    
    print(f"实验报告图表已保存到: {save_dir}")


if __name__ == "__main__":
    # 测试可视化功能
    print("=== 可视化工具测试 ===\n")
    
    # 测试混淆矩阵
    print("测试1: 混淆矩阵可视化")
    test_cm = np.array([
        [50, 2, 1, 0],
        [1, 48, 1, 0], 
        [0, 1, 47, 2],
        [0, 0, 1, 49]
    ])
    plot_confusion_matrix(test_cm, "./06_results/figures/test_confusion_matrix.png")
    
    print("\n" + "="*50 + "\n")
    
    # 测试训练曲线
    print("测试2: 训练曲线可视化")
    epochs = 20
    train_loss = [2.5 - 0.1*i + 0.1*np.sin(i/5) for i in range(epochs)]
    val_loss = [2.6 - 0.08*i + 0.05*np.cos(i/5) for i in range(epochs)]
    train_acc = [0.5 + 0.03*i - 0.02*np.sin(i/3) for i in range(epochs)]
    val_acc = [0.48 + 0.025*i - 0.01*np.cos(i/3) for i in range(epochs)]
    
    plot_train_curve(train_loss, val_loss, train_acc, val_acc, 
                   "./06_results/figures/test_training_curves.png")
    
    print("\n" + "="*50 + "\n")
    
    # 测试各类别指标
    print("测试3: 各类别指标可视化")
    test_metrics = {
        '正常': {'precision': 0.95, 'recall': 0.90, 'f1_score': 0.92},
        '内圈故障': {'precision': 0.88, 'recall': 0.91, 'f1_score': 0.89},
        '外圈故障': {'precision': 0.92, 'recall': 0.88, 'f1_score': 0.90},
        '滚动体故障': {'precision': 0.85, 'recall': 0.87, 'f1_score': 0.86}
    }
    
    plot_class_metrics(test_metrics, "./06_results/figures/test_class_metrics.png")
    
    print(f"\n=== 测试完成 ===")
    print("所有测试图表已保存到 ./06_results/figures/ 目录")
