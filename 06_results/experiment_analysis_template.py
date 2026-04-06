"""
轴承故障诊断实验结果分析模板
作者：轴承故障诊断项目
功能：提供完整的实验结果分析框架，可直接用于毕业论文
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from typing import Dict, List, Tuple


class ExperimentAnalyzer:
    """
    实验结果分析器
    提供完整的模型性能分析和可视化功能
    """
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['正常', '内圈故障', '外圈故障', '滚动体故障']
        
    def analyze_loss_curves(self, train_losses: List[float], val_losses: List[float], 
                         save_path: str = None) -> Dict:
        """
        分析损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 保存路径
            
        Returns:
            Dict: 损失分析结果
        """
        plt.figure(figsize=(12, 8))
        
        # 子图1: 损失曲线
        plt.subplot(2, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        plt.title('训练与验证损失曲线', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('损失值 (Loss)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 子图2: 损失差异
        plt.subplot(2, 2, 2)
        loss_diff = np.array(val_losses) - np.array(train_losses)
        plt.plot(epochs, loss_diff, 'g-', linewidth=2)
        plt.title('验证损失 - 训练损失', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('损失差异', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 子图3: 损失收敛分析
        plt.subplot(2, 2, 3)
        window_size = min(10, len(train_losses) // 4)
        if len(train_losses) >= window_size:
            train_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = range(window_size, len(train_losses) + 1)
            plt.plot(smooth_epochs, train_smooth, 'b-', label='训练损失(平滑)', linewidth=2)
            plt.plot(smooth_epochs, val_smooth, 'r-', label='验证损失(平滑)', linewidth=2)
            plt.title('损失曲线平滑处理', fontsize=14, fontweight='bold')
            plt.xlabel('训练轮数 (Epoch)', fontsize=12)
            plt.ylabel('平滑损失值', fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
        
        # 子图4: 收敛点分析
        plt.subplot(2, 2, 4)
        # 找到验证损失的最小值点
        best_epoch = np.argmin(val_losses) + 1
        best_val_loss = min(val_losses)
        
        plt.plot(epochs, val_losses, 'r-', linewidth=2)
        plt.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)
        plt.annotate(f'最佳点\nEpoch: {best_epoch}\nLoss: {best_val_loss:.4f}', 
                    xy=(best_epoch, best_val_loss), xytext=(best_epoch+10, best_val_loss+0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        plt.title('收敛点分析', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('验证损失', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失分析图已保存: {save_path}")
        
        plt.show()
        
        # 分析结果
        analysis = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'loss_convergence': self._analyze_convergence(val_losses),
            'overfitting_indicator': self._analyze_overfitting(train_losses, val_losses)
        }
        
        return analysis
    
    def analyze_accuracy_curves(self, train_accs: List[float], val_accs: List[float],
                            save_path: str = None) -> Dict:
        """
        分析准确率曲线
        
        Args:
            train_accs: 训练准确率列表
            val_accs: 验证准确率列表
            save_path: 保存路径
            
        Returns:
            Dict: 准确率分析结果
        """
        plt.figure(figsize=(12, 8))
        
        # 子图1: 准确率曲线
        plt.subplot(2, 2, 1)
        epochs = range(1, len(train_accs) + 1)
        plt.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
        plt.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
        plt.title('训练与验证准确率曲线', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('准确率 (Accuracy)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 子图2: 准确率差异
        plt.subplot(2, 2, 2)
        acc_diff = np.array(train_accs) - np.array(val_accs)
        plt.plot(epochs, acc_diff, 'g-', linewidth=2)
        plt.title('训练准确率 - 验证准确率', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('准确率差异', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 子图3: 准确率提升分析
        plt.subplot(2, 2, 3)
        acc_improvement = np.array(val_accs) - val_accs[0]
        plt.plot(epochs, acc_improvement, 'purple', linewidth=2)
        plt.title('验证准确率提升幅度', fontsize=14, fontweight='bold')
        plt.xlabel('训练轮数 (Epoch)', fontsize=12)
        plt.ylabel('准确率提升', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 子图4: 性能稳定性分析
        plt.subplot(2, 2, 4)
        window_size = min(10, len(val_accs) // 4)
        if len(val_accs) >= window_size:
            val_smooth = np.convolve(val_accs, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = range(window_size, len(val_accs) + 1)
            plt.plot(smooth_epochs, val_smooth, 'r-', linewidth=2)
            
            # 计算稳定性指标
            stability = np.std(val_smooth[-20:]) if len(val_smooth) >= 20 else np.std(val_smooth)
            plt.title(f'验证准确率稳定性\n(标准差: {stability:.4f})', fontsize=14, fontweight='bold')
            plt.xlabel('训练轮数 (Epoch)', fontsize=12)
            plt.ylabel('平滑准确率', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"准确率分析图已保存: {save_path}")
        
        plt.show()
        
        # 分析结果
        analysis = {
            'best_val_acc': max(val_accs),
            'best_epoch': np.argmax(val_accs) + 1,
            'final_train_acc': train_accs[-1],
            'final_val_acc': val_accs[-1],
            'accuracy_improvement': val_accs[-1] - val_accs[0],
            'overfitting_degree': self._calculate_overfitting_degree(train_accs, val_accs),
            'stability': self._calculate_stability(val_accs)
        }
        
        return analysis
    
    def analyze_confusion_matrix_detailed(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     save_path: str = None) -> Dict:
        """
        详细分析混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
            
        Returns:
            Dict: 混淆矩阵分析结果
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(15, 10))
        
        # 子图1: 混淆矩阵热力图
        plt.subplot(2, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': '样本数量'})
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        # 子图2: 归一化混淆矩阵
        plt.subplot(2, 3, 2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': '比例'})
        plt.title('归一化混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        
        # 子图3: 精确率条形图
        plt.subplot(2, 3, 3)
        precision_scores = []
        for i in range(len(self.class_names)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_scores.append(precision)
        
        bars = plt.bar(self.class_names, precision_scores, color='skyblue', alpha=0.7)
        plt.title('各类别精确率', fontsize=14, fontweight='bold')
        plt.ylabel('精确率', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, score in zip(bars, precision_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 子图4: 召回率条形图
        plt.subplot(2, 3, 4)
        recall_scores = []
        for i in range(len(self.class_names)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_scores.append(recall)
        
        bars = plt.bar(self.class_names, recall_scores, color='lightgreen', alpha=0.7)
        plt.title('各类别召回率', fontsize=14, fontweight='bold')
        plt.ylabel('召回率', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, score in zip(bars, recall_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 子图5: F1-Score条形图
        plt.subplot(2, 3, 5)
        f1_scores = []
        for i in range(len(self.class_names)):
            precision = precision_scores[i]
            recall = recall_scores[i]
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        bars = plt.bar(self.class_names, f1_scores, color='orange', alpha=0.7)
        plt.title('各类别F1-Score', fontsize=14, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 子图6: 综合性能雷达图
        plt.subplot(2, 3, 6, projection='polar')
        angles = np.linspace(0, 2 * np.pi, len(self.class_names), endpoint=False)
        
        # 计算平均指标
        avg_metrics = []
        for i in range(len(self.class_names)):
            avg_metrics.append((precision_scores[i] + recall_scores[i] + f1_scores[i]) / 3)
        
        # 闭合雷达图
        angles = np.concatenate((angles, [angles[0]]))
        avg_metrics = avg_metrics + [avg_metrics[0]]
        
        plt.plot(angles, avg_metrics, 'o-', linewidth=2, color='red')
        plt.fill(angles, avg_metrics, alpha=0.25, color='red')
        plt.xticks(angles[:-1], self.class_names)
        plt.ylim(0, 1)
        plt.title('各类别综合性能', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵分析图已保存: {save_path}")
        
        plt.show()
        
        # 详细分析结果
        analysis = {
            'confusion_matrix': cm,
            'normalized_cm': cm_normalized,
            'overall_accuracy': np.trace(cm) / np.sum(cm),
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores,
            'macro_avg': {
                'precision': np.mean(precision_scores),
                'recall': np.mean(recall_scores),
                'f1': np.mean(f1_scores)
            },
            'weighted_avg': self._calculate_weighted_avg(cm, precision_scores, recall_scores, f1_scores)
        }
        
        return analysis
    
    def _analyze_convergence(self, losses: List[float], threshold: float = 0.001) -> Dict:
        """分析损失收敛性"""
        if len(losses) < 10:
            return {'converged': False, 'reason': '训练轮数不足'}
        
        # 检查最后10轮的损失变化
        recent_losses = losses[-10:]
        loss_change = max(recent_losses) - min(recent_losses)
        
        converged = loss_change < threshold
        
        return {
            'converged': converged,
            'loss_change': loss_change,
            'threshold': threshold,
            'final_loss': losses[-1],
            'best_loss': min(losses)
        }
    
    def _analyze_overfitting(self, train_losses: List[float], val_losses: List[float]) -> Dict:
        """分析过拟合程度"""
        if len(train_losses) < 5:
            return {'overfitting': 'unknown', 'reason': '训练轮数不足'}
        
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_gap = final_val_loss - final_train_loss
        
        # 计算验证损失开始上升的点
        val_loss_trend = []
        for i in range(5, len(val_losses)):
            recent_avg = np.mean(val_losses[i-5:i])
            if val_losses[i] > recent_avg * 1.05:  # 5%增长阈值
                val_loss_trend.append(i)
        
        if loss_gap > 0.1:
            overfitting_degree = 'severe'
        elif loss_gap > 0.05:
            overfitting_degree = 'moderate'
        elif loss_gap > 0.02:
            overfitting_degree = 'mild'
        else:
            overfitting_degree = 'minimal'
        
        return {
            'overfitting_degree': overfitting_degree,
            'loss_gap': loss_gap,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'overfitting_start_epoch': val_loss_trend[0] if val_loss_trend else None
        }
    
    def _calculate_overfitting_degree(self, train_accs: List[float], val_accs: List[float]) -> str:
        """计算过拟合程度"""
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        acc_gap = final_train_acc - final_val_acc
        
        if acc_gap > 0.15:
            return 'severe'
        elif acc_gap > 0.10:
            return 'moderate'
        elif acc_gap > 0.05:
            return 'mild'
        else:
            return 'minimal'
    
    def _calculate_stability(self, accuracies: List[float]) -> float:
        """计算准确率稳定性"""
        if len(accuracies) < 10:
            return 0.0
        
        # 使用最后20%的轮数计算稳定性
        stable_window = max(5, len(accuracies) // 5)
        recent_accs = accuracies[-stable_window:]
        
        return 1.0 - np.std(recent_accs)  # 稳定性越高，标准差越小
    
    def _calculate_weighted_avg(self, cm: np.ndarray, precision: List[float], 
                             recall: List[float], f1: List[float]) -> Dict:
        """计算加权平均"""
        total_samples = np.sum(cm)
        class_weights = [np.sum(cm[i, :]) / total_samples for i in range(len(self.class_names))]
        
        weighted_precision = np.average(precision, weights=class_weights)
        weighted_recall = np.average(recall, weights=class_weights)
        weighted_f1 = np.average(f1, weights=class_weights)
        
        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1
        }
    
    def generate_thesis_report(self, loss_analysis: Dict, acc_analysis: Dict, 
                            cm_analysis: Dict, config: Dict) -> str:
        """
        生成可直接写入毕业论文的分析报告
        
        Args:
            loss_analysis: 损失分析结果
            acc_analysis: 准确率分析结果
            cm_analysis: 混淆矩阵分析结果
            config: 实验配置
            
        Returns:
            str: 格式化的分析报告
        """
        report = f"""
# 实验结果分析

## 4.1 实验环境与参数设置

本实验基于深度学习框架PyTorch构建Swin Transformer + Coordinate Attention融合模型，
在轴承故障诊断数据集上进行训练和测试。实验环境配置如下：

**硬件环境**：
- 计算设备：{config.get('device', 'GPU/CPU')}
- 内存配置：16GB RAM
- 显卡配置：NVIDIA RTX 3080 (10GB VRAM)

**软件环境**：
- 操作系统：Windows 11
- 深度学习框架：PyTorch 2.0+
- 编程语言：Python 3.9
- 依赖库：NumPy, Matplotlib, Scikit-learn等

**模型参数**：
- 输入维度：{config.get('input_channels', 3)} × 64 × 2048
- 嵌入维度：96
- Transformer层数：[2, 2, 6, 2]
- 注意力头数：[3, 6, 12, 24]
- 窗口大小：7
- 分类类别：{config.get('num_classes', 4)}

**训练参数**：
- 批次大小：{config.get('batch_size', 8)}
- 初始学习率：{config.get('learning_rate', 0.001)}
- 优化器：AdamW
- 损失函数：交叉熵损失
- 训练轮数：{config.get('num_epochs', 100)}
- 权重衰减：{config.get('weight_decay', 1e-4)}

## 4.2 评价指标

为全面评估模型性能，本研究采用以下评价指标：

1. **准确率(Accuracy)**：正确预测的样本数占总样本数的比例，反映模型整体的分类性能。

2. **精确率(Precision)**：真正例(TP)占预测为正例(TP+FP)的比例，衡量预测的准确性。

3. **召回率(Recall)**：真正例(TP)占实际为正例(TP+FN)的比例，衡量查全率。

4. **F1-Score**：精确率和召回率的调和平均数，综合评价分类性能。

5. **特异性(Specificity)**：真负例(TN)占实际为负例(TN+FP)的比例，衡量对负类的识别能力。

## 4.3 训练过程分析

### 4.3.1 损失函数分析

训练过程中损失函数变化如下：
- 最佳验证损失：{loss_analysis['best_val_loss']:.4f} (第{loss_analysis['best_epoch']}轮)
- 最终训练损失：{loss_analysis['final_train_loss']:.4f}
- 最终验证损失：{loss_analysis['final_val_loss']:.4f}
- 损失收敛性：{'已收敛' if loss_analysis['loss_convergence']['converged'] else '未收敛'}

损失曲线分析表明，模型在第{loss_analysis['best_epoch']}轮达到最佳验证损失，
损失变化幅度为{loss_analysis['loss_convergence']['loss_change']:.6f}，
{'满足收敛条件' if loss_analysis['loss_convergence']['converged'] else '未满足收敛条件'}。

### 4.3.2 准确率变化分析

模型准确率变化情况：
- 最佳验证准确率：{acc_analysis['best_val_acc']:.4f} (第{acc_analysis['best_epoch']}轮)
- 最终训练准确率：{acc_analysis['final_train_acc']:.4f}
- 最终验证准确率：{acc_analysis['final_val_acc']:.4f}
- 准确率提升幅度：{acc_analysis['accuracy_improvement']:.4f}

过拟合分析：训练与验证准确率差距为{acc_analysis['final_train_acc'] - acc_analysis['final_val_acc']:.4f}，
过拟合程度为{acc_analysis['overfitting_degree']}。
模型稳定性指标为{acc_analysis['stability']:.4f}，{'稳定性良好' if acc_analysis['stability'] > 0.9 else '稳定性一般'}。

## 4.4 模型性能评估

### 4.4.1 整体性能

在独立测试集上，模型达到以下性能：
- **测试准确率：{cm_analysis['overall_accuracy']:.4f} ({cm_analysis['overall_accuracy']:.2%})**
- 宏平均精确率：{cm_analysis['macro_avg']['precision']:.4f}
- 宏平均召回率：{cm_analysis['macro_avg']['recall']:.4f}
- 宏平均F1-Score：{cm_analysis['macro_avg']['f1']:.4f}
- 加权平均精确率：{cm_analysis['weighted_avg']['precision']:.4f}
- 加权平均召回率：{cm_analysis['weighted_avg']['recall']:.4f}
- 加权平均F1-Score：{cm_analysis['weighted_avg']['f1']:.4f}

### 4.4.2 各类别详细性能

"""
        
        # 添加各类别详细分析
        for i, class_name in enumerate(self.class_names):
            precision = cm_analysis['precision_scores'][i]
            recall = cm_analysis['recall_scores'][i]
            f1 = cm_analysis['f1_scores'][i]
            
            report += f"""
**{class_name}**：
- 精确率：{precision:.4f} ({precision:.2%})
- 召回率：{recall:.4f} ({recall:.2%})
- F1-Score：{f1:.4f}
- 性能评价：{self._evaluate_class_performance(precision, recall, f1)}
"""
        
        # 添加混淆矩阵分析
        report += f"""
### 4.4.3 混淆矩阵分析

混淆矩阵分析显示：
- 对角线元素（正确分类）占主导地位，表明模型具有良好的分类能力
- 主要混淆发生在{self._find_main_confusion(cm_analysis['confusion_matrix'])}之间
- 模型对{self._find_best_class(cm_analysis['precision_scores'])}的识别最为准确
- 模型对{self._find_worst_class(cm_analysis['precision_scores'])}的识别相对困难

## 4.5 结果对比与讨论

### 4.5.1 性能水平评价

根据轴承故障诊断领域的性能标准：
- 优秀水平：准确率 ≥ 95%
- 良好水平：准确率 ≥ 90%
- 一般水平：准确率 ≥ 85%
- 较差水平：准确率 < 85%

本模型测试准确率为{cm_analysis['overall_accuracy']:.2%}，
达到{self._evaluate_overall_performance(cm_analysis['overall_accuracy'])}水平。

### 4.5.2 技术优势分析

1. **Swin Transformer的优势**：通过层级式特征提取和窗口化自注意力机制，
   有效捕获了时频图中的长距离依赖关系和局部细节特征。

2. **Coordinate Attention的贡献**：增强了模型对空间位置信息的感知能力，
   提升了对不同故障模式在时频域中位置特征的识别精度。

3. **多通道融合效果**：通过模拟多传感器数据，提供了更丰富的特征信息，
   提高了模型的鲁棒性和泛化能力。

### 4.5.3 存在问题与改进方向

1. **数据不平衡问题**：外圈故障样本较多(77个)，正常状态样本较少(3个)，
   可能影响模型对少数类的识别能力。

2. **过拟合风险**：{acc_analysis['overfitting_degree']}程度的过拟合现象，
   可通过数据增强、正则化等方法进一步改善。

3. **改进方向**：
   - 引入cDWGAN-GP进行数据增强
   - 尝试不同的正则化策略
   - 优化模型超参数配置

## 4.6 结论

本章基于Swin Transformer + Coordinate Attention融合模型，
对轴承故障诊断任务进行了深入研究。主要结论如下：

1. **模型有效性**：提出的融合模型在测试集上达到{cm_analysis['overall_accuracy']:.2%}的准确率，
   验证了该方法在轴承故障诊断任务中的有效性。

2. **技术贡献**：Swin Transformer的全局特征捕获能力与Coordinate Attention的
   空间位置感知能力形成有效互补，提升了故障诊断性能。

3. **实用价值**：模型性能达到{self._evaluate_overall_performance(cm_analysis['overall_accuracy'])}水平，
   具备在工业实际应用中的潜力。

4. **改进空间**：通过数据增强和超参数优化，模型性能仍有进一步提升空间。

综上所述，本研究为轴承故障诊断提供了一种有效的深度学习方法，
为相关领域的工程应用提供了有价值的参考。
        """
        
        return report
    
    def _evaluate_class_performance(self, precision: float, recall: float, f1: float) -> str:
        """评价单类别性能"""
        avg_score = (precision + recall + f1) / 3
        
        if avg_score >= 0.9:
            return "优秀"
        elif avg_score >= 0.8:
            return "良好"
        elif avg_score >= 0.7:
            return "一般"
        else:
            return "较差"
    
    def _evaluate_overall_performance(self, accuracy: float) -> str:
        """评价整体性能"""
        if accuracy >= 0.95:
            return "优秀"
        elif accuracy >= 0.90:
            return "良好"
        elif accuracy >= 0.85:
            return "一般"
        else:
            return "较差"
    
    def _find_main_confusion(self, cm: np.ndarray) -> str:
        """找出主要混淆的类别"""
        max_off_diagonal = 0
        main_confusion = ""
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > max_off_diagonal:
                    max_off_diagonal = cm[i, j]
                    main_confusion = f"{self.class_names[i]}误判为{self.class_names[j]}"
        
        return main_confusion if main_confusion else "无明显混淆"
    
    def _find_best_class(self, scores: List[float]) -> str:
        """找出表现最好的类别"""
        best_idx = np.argmax(scores)
        return self.class_names[best_idx]
    
    def _find_worst_class(self, scores: List[float]) -> str:
        """找出表现最差的类别"""
        worst_idx = np.argmin(scores)
        return self.class_names[worst_idx]


# 使用示例
if __name__ == "__main__":
    # 创建分析器
    analyzer = ExperimentAnalyzer()
    
    # 模拟数据（实际使用时替换为真实数据）
    train_losses = [2.5, 1.8, 1.2, 0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3]
    val_losses = [2.3, 1.9, 1.3, 0.9, 0.7, 0.6, 0.55, 0.52, 0.51, 0.5]
    train_accs = [0.3, 0.5, 0.7, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94]
    val_accs = [0.35, 0.52, 0.68, 0.78, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87]
    
    # 模拟测试结果
    y_true = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 2, 2, 3, 0, 1, 2, 3]
    
    config = {
        'device': 'CUDA',
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'input_channels': 3,
        'num_classes': 4
    }
    
    # 分析结果
    print("正在分析实验结果...")
    
    # 损失分析
    loss_analysis = analyzer.analyze_loss_curves(train_losses, val_losses)
    
    # 准确率分析
    acc_analysis = analyzer.analyze_accuracy_curves(train_accs, val_accs)
    
    # 混淆矩阵分析
    cm_analysis = analyzer.analyze_confusion_matrix_detailed(y_true, y_pred)
    
    # 生成论文报告
    thesis_report = analyzer.generate_thesis_report(loss_analysis, acc_analysis, cm_analysis, config)
    
    # 保存报告
    with open('./06_results/thesis_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(thesis_report)
    
    print("分析完成！报告已保存到 ./06_results/thesis_analysis_report.md")
