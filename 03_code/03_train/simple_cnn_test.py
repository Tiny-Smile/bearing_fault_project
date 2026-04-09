"""
简单CNN测试 - 快速测试真实CWT数据
作者：轴承故障诊断项目
功能：简单直接地使用CWT数据测试CNN模型
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, '03_code', '02_models'))

from cnn_baseline import CNNBaseline, create_cnn_baseline_model


def simple_test():
    """简单的CNN模型测试"""
    print("=" * 50)
    print("简单CNN测试 - 使用真实CWT数据")
    print("=" * 50)
    
    # 1. 加载CWT测试数据
    print("1. 加载CWT数据...")
    data_path = "./02_data/cwt/cwt_test.npz"
    
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return
    
    # 加载数据
    data = np.load(data_path)
    X_numpy = data['X']  # 形状: (samples, 3, 64, 2048)
    y_numpy = data['y']  # 形状: (samples,)
    
    print(f"   数据形状: X={X_numpy.shape}, y={y_numpy.shape}")
    print(f"   数据类型: X={X_numpy.dtype}, y={y_numpy.dtype}")
    print(f"   标签范围: {y_numpy.min()} - {y_numpy.max()}")
    
    # 2. 转换为PyTorch张量
    print("\n2. 转换数据格式...")
    X_tensor = torch.FloatTensor(X_numpy)  # 转换为float32
    y_tensor = torch.LongTensor(y_numpy)   # 转换为long
    
    print(f"   张量形状: X={X_tensor.shape}, y={y_tensor.shape}")
    print(f"   张量类型: X={X_tensor.dtype}, y={y_tensor.dtype}")
    
    # 3. 创建CNN模型
    print("\n3. 创建CNN模型...")
    model, device = create_cnn_baseline_model(device='cpu')
    model.eval()
    
    print(f"   模型设备: {device}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 前向传播测试
    print("\n4. 前向传播测试...")
    
    # 测试前5个样本
    batch_size = 5
    test_batch = X_tensor[:batch_size].to(device)
    test_labels = y_tensor[:batch_size]
    
    print(f"   测试批次大小: {batch_size}")
    print(f"   输入形状: {test_batch.shape}")
    
    with torch.no_grad():
        outputs = model(test_batch)
        predictions = torch.argmax(outputs, dim=1)
    
    print(f"   输出形状: {outputs.shape}")
    print(f"   输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
    
    # 5. 显示预测结果
    print("\n5. 预测结果对比:")
    print("   样本 | 真实标签 | 预测标签 | 概率分布")
    print("   ----|---------|---------|----------")
    
    class_names = ['正常', '内圈', '外圈', '滚动体']
    
    for i in range(batch_size):
        true_label = test_labels[i].item()
        pred_label = predictions[i].item()
        probabilities = outputs[i].cpu().numpy()
        
        status = "OK" if true_label == pred_label else "WRONG"
        
        print(f"   {i+1:2d}   |    {true_label:d}    |    {pred_label:d}    | [{probabilities[0]:.3f}, {probabilities[1]:.3f}, {probabilities[2]:.3f}, {probabilities[3]:.3f}] {status}")
        print(f"       | {class_names[true_label]:6s} | {class_names[pred_label]:6s} |")
    
    # 6. 计算整体准确率
    print("\n6. 整体测试...")
    
    # 测试所有数据
    all_X = X_tensor.to(device)
    all_y = y_tensor
    
    with torch.no_grad():
        all_outputs = model(all_X)
        all_predictions = torch.argmax(all_outputs, dim=1)
    
    # 计算准确率
    correct = (all_predictions == all_y).sum().item()
    total = all_y.size(0)
    accuracy = correct / total * 100
    
    print(f"   总样本数: {total}")
    print(f"   正确预测: {correct}")
    print(f"   准确率: {accuracy:.2f}%")
    
    # 7. 按类别统计
    print("\n7. 各类别预测统计:")
    print("   类别 | 真实数量 | 正确预测 | 准确率")
    print("   ----|---------|---------|--------")
    
    for class_id in range(4):
        # 找到该类别的所有样本
        class_mask = (all_y == class_id)
        class_count = class_mask.sum().item()
        
        if class_count > 0:
            # 计算该类别的正确预测数
            class_correct = ((all_predictions == all_y) & class_mask).sum().item()
            class_accuracy = class_correct / class_count * 100
            
            print(f"   {class_id:d}   |    {class_count:d}    |    {class_correct:d}    |  {class_accuracy:.1f}%")
            print(f"       | {class_names[class_id]:6s} |         |         |")
        else:
            print(f"   {class_id:d}   |    0    |    0    |  N/A")
            print(f"       | {class_names[class_id]:6s} |         |         |")
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = simple_test()
        print(f"\n最终准确率: {accuracy:.2f}%")
        
        if accuracy > 50:
            print("模型表现良好！")
        elif accuracy > 25:
            print("模型表现一般，可能需要训练")
        else:
            print("模型表现较差，需要重新训练")
            
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
