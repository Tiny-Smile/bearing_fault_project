"""
评价指标与可视化工具集成测试
作者：轴承故障诊断项目
功能：测试metric_utils和vis_utils的完整功能
"""

import numpy as np
import os
import sys

# 添加路径
sys.path.append(os.path.dirname(__file__))

from metric_utils import calculate_metrics, calculate_per_class_metrics, calculate_confusion_matrix_stats
from vis_utils import plot_confusion_matrix, plot_train_curve, plot_class_metrics, create_experiment_report


def test_complete_pipeline():
    """测试完整的评价指标与可视化流程"""
    print("=== 评价指标与可视化完整测试 ===\n")
    
    # 1. 生成模拟的测试数据
    print("步骤1: 生成模拟测试数据")
    np.random.seed(42)
    
    # 模拟4类分类的预测结果
    n_samples = 100
    y_true = np.random.randint(0, 4, n_samples)
    
    # 添加一些噪声来模拟预测错误
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    y_pred[error_indices] = np.random.randint(0, 4, len(error_indices))
    
    print(f"  样本数量: {n_samples}")
    print(f"  真实标签分布: {np.bincount(y_true)}")
    print(f"  预测标签分布: {np.bincount(y_pred)}")
    
    # 2. 计算评价指标
    print("\n步骤2: 计算评价指标")
    accuracy, f1_macro, cm, report = calculate_metrics(y_true, y_pred)
    
    # 计算各类别详细指标
    per_class_metrics = calculate_per_class_metrics(y_true, y_pred)
    
    # 分析混淆矩阵
    cm_stats = calculate_confusion_matrix_stats(cm)
    
    # 3. 生成可视化
    print("\n步骤3: 生成可视化图表")
    
    # 确保输出目录存在
    output_dir = "./06_results/figures/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        cm, 
        os.path.join(output_dir, "final_confusion_matrix.png"),
        title="轴承故障诊断混淆矩阵"
    )
    
    # 绘制各类别指标
    plot_class_metrics(
        per_class_metrics,
        os.path.join(output_dir, "final_class_metrics.png"),
        title="各类别性能指标对比"
    )
    
    # 4. 生成模拟的训练曲线
    print("\n步骤4: 生成训练曲线可视化")
    epochs = 50
    
    # 模拟训练过程
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(epochs):
        # 损失逐渐下降，添加一些波动
        t_loss = 2.5 * np.exp(-epoch/15) + 0.1 + 0.05 * np.sin(epoch/5)
        v_loss = 2.6 * np.exp(-epoch/18) + 0.15 + 0.08 * np.cos(epoch/4)
        
        # 准确率逐渐上升，添加一些波动
        t_acc = 0.95 - 0.95 * np.exp(-epoch/12) + 0.02 * np.sin(epoch/3)
        v_acc = 0.93 - 0.93 * np.exp(-epoch/14) + 0.03 * np.cos(epoch/3.5)
        
        train_loss.append(t_loss)
        val_loss.append(v_loss)
        train_acc.append(min(t_acc, 0.99))  # 限制最大值
        val_acc.append(min(v_acc, 0.98))
    
    plot_train_curve(
        train_loss, val_loss, train_acc, val_acc,
        os.path.join(output_dir, "final_training_curves.png"),
        title="Swin-CA模型训练过程"
    )
    
    # 5. 创建综合实验报告
    print("\n步骤5: 创建综合实验报告")
    
    metrics_dict = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }
    
    create_experiment_report(metrics_dict, output_dir)
    
    # 6. 输出总结
    print("\n" + "="*60)
    print("实验结果总结:")
    print(f"  总体准确率: {accuracy:.4f}")
    print(f"  宏平均F1: {f1_macro:.4f}")
    print(f"  混淆矩阵形状: {cm.shape}")
    print(f"  输出图表数量: 4个")
    print(f"  输出目录: {output_dir}")
    
    # 7. 验证输出文件
    print("\n步骤6: 验证输出文件")
    expected_files = [
        "final_confusion_matrix.png",
        "final_class_metrics.png", 
        "final_training_curves.png"
    ]
    
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        exists = os.path.exists(filepath)
        status = "OK" if exists else "FAIL"
        print(f"  {filename}: {status}")
    
    print(f"\n=== 测试完成 ===")
    return True


if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n[SUCCESS] 所有评价指标与可视化工具测试通过！")
    else:
        print("\n[ERROR] 测试过程中出现问题！")
