"""
轴承故障诊断评价指标计算工具模块
作者：轴承故障诊断项目
功能：提供准确率、F1分数、混淆矩阵等评价指标计算
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, np.ndarray, Dict[str, Any]]:
    """
    计算分类模型的评价指标
    
    Args:
        y_true (np.ndarray): 真实标签，形状为 (n_samples,)
        y_pred (np.ndarray): 预测标签，形状为 (n_samples,)
        
    Returns:
        Tuple[float, float, np.ndarray, Dict[str, Any]]:
            - accuracy: 准确率
            - f1_macro: 宏平均F1分数
            - confusion_matrix: 混淆矩阵
            - report: 详细的分类报告字典
            
    Example:
        >>> y_true = np.array([0, 1, 2, 3])
        >>> y_pred = np.array([0, 1, 2, 3])
        >>> acc, f1, cm, report = calculate_metrics(y_true, y_pred)
        >>> print(f"准确率: {acc:.4f}")
    """
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算宏平均F1分数
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 生成详细的分类报告
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    print(f"评价指标计算完成:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  宏平均F1: {f1_macro:.4f}")
    print(f"  混淆矩阵形状: {cm.shape}")
    
    return accuracy, f1_macro, cm, report


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    计算每个类别的详细指标
    
    Args:
        y_true (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        
    Returns:
        Dict[str, Dict[str, float]]: 每个类别的详细指标
        
    Example:
        >>> metrics = calculate_per_class_metrics(y_true, y_pred)
        >>> print(metrics['正常']['precision'])
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # 类别名称映射
    class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    
    # 计算每个类别的详细指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 构建结果字典
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
    
    # 打印每个类别的指标
    print("\n各类别详细指标:")
    for class_name, metrics in per_class_metrics.items():
        print(f"  {class_name}:")
        print(f"    精确率: {metrics['precision']:.4f}")
        print(f"    召回率: {metrics['recall']:.4f}")
        print(f"    F1分数: {metrics['f1_score']:.4f}")
        print(f"    支持样本数: {metrics['support']}")
    
    return per_class_metrics


def calculate_confusion_matrix_stats(cm: np.ndarray) -> Dict[str, Any]:
    """
    分析混淆矩阵的统计信息
    
    Args:
        cm (np.ndarray): 混淆矩阵，形状为 (n_classes, n_classes)
        
    Returns:
        Dict[str, Any]: 混淆矩阵统计信息
        
    Example:
        >>> stats = calculate_confusion_matrix_stats(cm)
        >>> print(f"总体准确率: {stats['overall_accuracy']:.4f}")
    """
    n_classes = cm.shape[0]
    class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    
    # 计算各类别的统计信息
    stats = {}
    
    # 每个类别的真实样本数
    stats['true_counts'] = {}
    # 每个类别的预测样本数
    stats['pred_counts'] = {}
    # 每个类别的正确预测数
    stats['correct_counts'] = {}
    
    for i in range(n_classes):
        true_count = cm[i, :].sum()
        pred_count = cm[:, i].sum()
        correct_count = cm[i, i]
        
        stats['true_counts'][class_names[i]] = int(true_count)
        stats['pred_counts'][class_names[i]] = int(pred_count)
        stats['correct_counts'][class_names[i]] = int(correct_count)
    
    # 计算总体准确率
    total_correct = np.diag(cm).sum()
    total_samples = cm.sum()
    stats['overall_accuracy'] = total_correct / total_samples if total_samples > 0 else 0.0
    
    print("\n混淆矩阵统计:")
    print(f"  总体准确率: {stats['overall_accuracy']:.4f}")
    print(f"  总样本数: {total_samples}")
    print(f"  正确预测数: {total_correct}")
    
    return stats


if __name__ == "__main__":
    # 测试评价指标计算
    print("=== 评价指标计算测试 ===\n")
    
    # 模拟测试数据
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3, 0, 1])
    
    print("测试数据:")
    print(f"  真实标签: {y_true}")
    print(f"  预测标签: {y_pred}")
    
    # 计算评价指标
    accuracy, f1_macro, cm, report = calculate_metrics(y_true, y_pred)
    
    # 计算各类别详细指标
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred)
    
    # 分析混淆矩阵
    cm_stats = calculate_confusion_matrix_stats(cm)
    
    print(f"\n=== 测试完成 ===")
    print(f"准确率: {accuracy:.4f}")
    print(f"宏平均F1: {f1_macro:.4f}")
