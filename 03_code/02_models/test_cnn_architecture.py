"""
CNN基线模型架构可视化
作者：轴承故障诊断项目
功能：可视化CNN基线模型的架构
"""

# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块
import torch.nn as nn
# 从cnn_baseline模块导入模型和创建函数
from cnn_baseline import CNNBaseline, create_cnn_baseline_model


def print_architecture_details():
    """打印详细的架构信息"""
    print("=== CNN 基线模型架构详情 ===\n")
    
    # 创建模型
    model, device = create_cnn_baseline_model(device='cpu')
    
    # 逐层打印架构信息
    print("逐层架构信息:")
    print("-" * 50)
    
    # 测试输入
    test_input = torch.randn(1, 3, 64, 2048)  # 创建随机输入
    print(f"输入: {test_input.shape}")
    
    # 第一个卷积块：卷积 + ReLU + 池化
    x = model.conv1(test_input)  # 第一个卷积层
    print(f"经过Conv1 (3->16, 3x3): {x.shape}")
    
    x = model.relu1(x)  # 第一个ReLU激活函数
    print(f"经过ReLU1: {x.shape}")
    
    x = model.pool1(x)  # 第一个最大池化层
    print(f"经过MaxPool1 (2x2): {x.shape}")
    
    # 第二个卷积块：卷积 + ReLU + 池化
    x = model.conv2(x)  # 第二个卷积层
    print(f"经过Conv2 (16->32, 3x3): {x.shape}")
    
    x = model.relu2(x)  # 第二个ReLU激活函数
    print(f"经过ReLU2: {x.shape}")
    
    x = model.pool2(x)  # 第二个最大池化层
    print(f"经过MaxPool2 (2x2): {x.shape}")
    
    # 展平操作
    x = x.view(x.size(0), -1)  # 将2D特征图展平为1D向量
    print(f"经过Flatten: {x.shape}")
    
    # 第一个全连接块：全连接 + ReLU
    x = model.fc1(x)  # 第一个全连接层
    print(f"经过FC1 (->128): {x.shape}")
    
    x = model.relu3(x)  # 第三个ReLU激活函数
    print(f"经过ReLU3: {x.shape}")
    
    # 第二个全连接块：全连接 + Softmax
    x = model.fc2(x)  # 第二个全连接层
    print(f"经过FC2 (->4): {x.shape}")
    
    x = model.softmax(x)  # Softmax激活函数
    print(f"经过Softmax: {x.shape}")
    
    print("-" * 50)
    
    # 计算每层的参数数量
    print("\n每层参数数量:")
    print("-" * 50)
    
    # 遍历模型的所有命名参数
    for name, param in model.named_parameters():
        if param.requires_grad:  # 只显示可训练参数
            print(f"{name}: {param.numel():,} 参数")  # 使用千位分隔符
    
    print("-" * 50)
    
    # 内存使用估算
    print("\n内存使用估算:")
    print("-" * 50)
    
    # 输入内存
    input_memory = test_input.numel() * 4 / 1024 / 1024  # float32 = 4字节
    print(f"输入内存: {input_memory:.2f} MB")
    
    # 参数内存
    param_memory = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    print(f"参数内存: {param_memory:.2f} MB")
    
    # 输出内存
    output = model(test_input)  # 获取模型输出
    output_memory = output.numel() * 4 / 1024 / 1024
    print(f"输出内存: {output_memory:.4f} MB")
    
    # 总内存估算
    total_memory = input_memory + param_memory + output_memory
    print(f"总估算内存: {total_memory:.2f} MB")
    
    print("-" * 50)


def test_model_forward_pass():
    """使用详细输出测试模型前向传播"""
    print("\n=== 前向传播测试 ===\n")
    
    # 创建模型并设置为评估模式
    model, device = create_cnn_baseline_model(device='cpu')
    model.eval()  # 设置为评估模式，关闭dropout等
    
    # 创建测试输入
    batch_size = 4  # 批次大小
    test_input = torch.randn(batch_size, 3, 64, 2048)  # 随机输入
    
    print(f"测试输入形状: {test_input.shape}")
    print(f"测试输入范围: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # 前向传播
    with torch.no_grad():  # 不计算梯度，节省内存
        output = model(test_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"每个样本的输出和: {output.sum(dim=1)}")
    
    # 验证softmax属性
    print(f"所有输出和为1.0: {torch.allclose(output.sum(dim=1), torch.ones(batch_size))}")
    
    # 获取预测类别
    predicted_classes = torch.argmax(output, dim=1)  # 取最大概率的索引
    print(f"预测类别: {predicted_classes}")
    
    print("\n[OK] 前向传播测试成功完成！")


if __name__ == "__main__":
    # 打印架构详情
    print_architecture_details()
    # 测试前向传播
    test_model_forward_pass()
    
    print("\n" + "="*60)
    print("CNN 基线模型总结:")
    print("="*60)
    print("架构: 2个卷积层 + 2个全连接层")
    print("输入: (3, 64, 2048) 多通道时频图")
    print("输出: 4类概率分布")
    print("参数量: 约33.6M")
    print("设备兼容性: CPU/GPU（需要正确的CUDA设置）")
    print("="*60)
