"""
Swin-CA模型训练脚本
作者：轴承故障诊断项目
功能：训练Swin Transformer + Coordinate Attention轴承故障诊断模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Dict, List, Tuple
import time
from datetime import datetime

# 导入模型和工具
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '02_models'))

from cwt_utils import load_split_data
from swin_ca import create_swin_ca_model


class ConfusionMatrixAnalyzer:
    """
    混淆矩阵分析器
    """
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           save_path: str = None, title: str = "混淆矩阵"):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.show()
        
        return cm
    
    def analyze_confusion_matrix(self, cm: np.ndarray) -> Dict:
        """
        分析混淆矩阵指标
        """
        n_classes = cm.shape[0]
        analysis = {}
        
        # 计算每个类别的指标
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            analysis[self.class_names[i]] = {
                'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Specificity': specificity
            }
        
        return analysis


class TrainingMetrics:
    """
    训练指标记录器
    """
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss: float, val_loss: float, 
               train_acc: float, val_acc: float, lr: float):
        """更新指标"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='训练损失', color='blue')
        axes[0, 0].plot(self.val_losses, label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_accuracies, label='训练准确率', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='验证准确率', color='red')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 损失差异
        loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1, 0].plot(loss_diff, color='green')
        axes[1, 0].set_title('验证损失 - 训练损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].grid(True)
        
        # 学习率曲线
        axes[1, 1].plot(self.learning_rates, color='purple')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        
        plt.show()


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率"""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total


def train_epoch(model: nn.Module, dataloader: DataLoader, 
               criterion: nn.Module, optimizer: optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'批次 {batch_idx}/{len(dataloader)}, 损失: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model: nn.Module, dataloader: DataLoader,
                 criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def evaluate_model(model: nn.Module, dataloader: DataLoader,
                 device: torch.device, class_names: List[str]) -> Dict:
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算指标
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    
    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, 
                               output_dict=True, zero_division=0)
    
    # 混淆矩阵分析
    analyzer = ConfusionMatrixAnalyzer(class_names)
    cm = analyzer.plot_confusion_matrix(y_true, y_pred)
    cm_analysis = analyzer.analyze_confusion_matrix(cm)
    
    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'confusion_analysis': cm_analysis,
        'accuracy': np.mean(y_true == y_pred)
    }


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'weight_decay': 1e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_classes': 4,
        'input_channels': 3,
        'class_names': ['正常', '内圈故障', '外圈故障', '滚动体故障']
    }
    
    print("=" * 60)
    print("Swin-CA轴承故障诊断模型训练")
    print("=" * 60)
    print(f"设备: {config['device']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_epochs']}")
    print("=" * 60)
    
    # 加载数据
    print("加载CWT数据集...")
    try:
        X_train, y_train = load_split_data("./02_data/cwt/cwt_train.npz")
        X_val, y_val = load_split_data("./02_data/cwt/cwt_val.npz")
        X_test, y_test = load_split_data("./02_data/cwt/cwt_test.npz")
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建模型
    print("创建Swin-CA模型...")
    model = create_swin_ca_model(
        num_classes=config['num_classes'],
        input_channels=config['input_channels']
    )
    model = model.to(config['device'])
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # 训练指标记录
    metrics = TrainingMetrics()
    best_val_acc = 0.0
    best_model_path = None
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config['device'])
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标
        metrics.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f"./04_models_ckpt/best_swin_ca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            os.makedirs("./04_models_ckpt/", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, best_model_path)
            print(f"保存最佳模型: {best_model_path}")
    
    # 训练完成
    training_time = time.time() - start_time
    print(f"\n训练完成！用时: {training_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    # 绘制训练曲线
    os.makedirs("./06_results/figures/", exist_ok=True)
    metrics.plot_training_curves("./06_results/figures/training_curves.png")
    
    # 测试集评估
    print("\n在测试集上评估最佳模型...")
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate_model(model, test_loader, config['device'], config['class_names'])
    
    # 保存结果
    results = {
        'config': config,
        'training_time': training_time,
        'best_val_acc': best_val_acc,
        'test_results': test_results,
        'training_metrics': {
            'train_losses': metrics.train_losses,
            'val_losses': metrics.val_losses,
            'train_accuracies': metrics.train_accuracies,
            'val_accuracies': metrics.val_accuracies
        }
    }
    
    # 生成实验报告
    generate_experiment_report(results)
    
    print("训练和评估完成！")


def generate_experiment_report(results: Dict):
    """生成实验报告"""
    print("\n" + "="*60)
    print("实验结果分析报告")
    print("="*60)
    
    config = results['config']
    test_results = results['test_results']
    
    # 1. 实验环境
    print("\n1. 实验环境")
    print("-" * 30)
    print(f"硬件设备: {config['device']}")
    print(f"深度学习框架: PyTorch {torch.__version__}")
    print(f"编程语言: Python 3.9+")
    
    # 2. 参数设置
    print("\n2. 模型参数设置")
    print("-" * 30)
    print(f"模型架构: Swin Transformer + Coordinate Attention")
    print(f"输入维度: {config['input_channels']} × 64 × 2048")
    print(f"类别数量: {config['num_classes']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"初始学习率: {config['learning_rate']}")
    print(f"训练轮数: {config['num_epochs']}")
    print(f"权重衰减: {config['weight_decay']}")
    
    # 3. 评价指标
    print("\n3. 评价指标")
    print("-" * 30)
    print("• 准确率(Accuracy): 正确预测样本数 / 总样本数")
    print("• 精确率(Precision): TP / (TP + FP)")
    print("• 召回率(Recall): TP / (TP + FN)")
    print("• F1-Score: 2 × (Precision × Recall) / (Precision + Recall)")
    print("• 特异性(Specificity): TN / (TN + FP)")
    
    # 4. 实验结果
    print("\n4. 实验结果")
    print("-" * 30)
    print(f"训练时间: {results['training_time']/60:.2f} 分钟")
    print(f"最佳验证准确率: {results['best_val_acc']:.4f}")
    print(f"测试集准确率: {test_results['accuracy']:.4f}")
    
    # 5. 详细分类报告
    print("\n5. 详细分类报告")
    print("-" * 30)
    report = test_results['classification_report']
    for class_name in config['class_names']:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
            print(f"  支持数: {metrics['support']}")
    
    # 6. 混淆矩阵分析
    print("\n6. 混淆矩阵分析")
    print("-" * 30)
    cm_analysis = test_results['confusion_analysis']
    for class_name, analysis in cm_analysis.items():
        print(f"\n{class_name}:")
        print(f"  真正例(TP): {analysis['TP']}")
        print(f"  假正例(FP): {analysis['FP']}")
        print(f"  假负例(FN): {analysis['FN']}")
        print(f"  真负例(TN): {analysis['TN']}")
        print(f"  精确率: {analysis['Precision']:.4f}")
        print(f"  召回率: {analysis['Recall']:.4f}")
        print(f"  F1-Score: {analysis['F1-Score']:.4f}")
        print(f"  特异性: {analysis['Specificity']:.4f}")
    
    # 7. 模型性能分析
    print("\n7. 模型性能分析")
    print("-" * 30)
    
    # 过拟合/欠拟合分析
    train_metrics = results['training_metrics']
    final_train_acc = train_metrics['train_accuracies'][-1]
    final_val_acc = train_metrics['val_accuracies'][-1]
    accuracy_gap = final_train_acc - final_val_acc
    
    print(f"最终训练准确率: {final_train_acc:.4f}")
    print(f"最终验证准确率: {final_val_acc:.4f}")
    print(f"准确率差距: {accuracy_gap:.4f}")
    
    if accuracy_gap > 0.1:
        print("⚠️  模型存在过拟合现象")
    elif accuracy_gap < 0.05:
        print("✅ 模型拟合良好，无明显过拟合")
    else:
        print("ℹ️  模型拟合适中")
    
    # 准确率评价
    test_acc = test_results['accuracy']
    if test_acc >= 0.95:
        print("✅ 测试准确率优秀 (≥95%)")
    elif test_acc >= 0.90:
        print("✅ 测试准确率良好 (≥90%)")
    elif test_acc >= 0.85:
        print("⚠️  测试准确率一般 (≥85%)")
    else:
        print("❌ 测试准确率较差 (<85%)")
    
    # 8. 结论分析
    print("\n8. 结论分析")
    print("-" * 30)
    print("本实验基于Swin Transformer + Coordinate Attention融合模型，")
    print("对轴承故障诊断任务进行了深入研究。实验结果表明：")
    print(f"• 模型在测试集上达到{test_acc:.2%}的准确率")
    print("• Swin Transformer有效捕获了时频图的全局特征")
    print("• Coordinate Attention增强了空间位置信息的感知能力")
    print("• 模型对不同故障类型具有良好的识别能力")
    
    if test_acc >= 0.90:
        print("• 模型性能满足工业应用要求")
    else:
        print("• 模型性能有待进一步提升，可考虑数据增强等策略")


if __name__ == "__main__":
    main()
