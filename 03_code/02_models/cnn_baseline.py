"""
CNN基线模型 - 轴承故障诊断
作者：轴承故障诊断项目
功能：实现一个3D CNN模型用于多通道时频图分类
"""

# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络函数模块（如激活函数等）
import torch.nn.functional as F
# 导入类型提示工具
from typing import Tuple
# 导入数值计算库
import numpy as np


class CNNBaseline(nn.Module):
    """
    CNN基线模型，用于基于CWT时频图的轴承故障诊断
    
    网络结构：
    输入 (3, 64, 2048) 
    -> Conv2d(3->16, 3x3) -> ReLU -> MaxPool(2x2)
    -> Conv2d(16->32, 3x3) -> ReLU -> MaxPool(2x2) 
    -> Flatten -> Linear(->128) -> ReLU -> Linear(->4) -> Softmax
    """
    
    def __init__(self, num_classes: int = 4):
        """
        初始化CNN基线模型
        
        Args:
            num_classes (int): 故障类别数量（默认为4类）
        """
        # 调用父类的初始化方法
        super(CNNBaseline, self).__init__()
        
        # 第一个卷积块：卷积层 + 激活函数 + 池化层
        # 卷积层：输入3通道，输出16通道，卷积核3x3，填充1保持尺寸
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # ReLU激活函数：引入非线性，让网络能学习更复杂的特征
        self.relu1 = nn.ReLU()
        # 最大池化层：2x2窗口，步长2，将特征图尺寸减半
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块：卷积层 + 激活函数 + 池化层
        # 卷积层：输入16通道，输出32通道，卷积核3x3，填充1保持尺寸
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # ReLU激活函数
        self.relu2 = nn.ReLU()
        # 最大池化层：2x2窗口，步长2，再次将特征图尺寸减半
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 计算卷积层后的展平尺寸
        # 输入尺寸: (3, 64, 2048)
        # 经过第一个卷积块后: (16, 32, 1024) - 高度和宽度都减半
        # 经过第二个卷积块后: (32, 16, 512) - 高度和宽度再次减半
        self.flattened_size = 32 * 16 * 512
        
        # 全连接层：将卷积特征映射到分类结果
        # 第一个全连接层：将展平特征映射到128维特征空间
        self.fc1 = nn.Linear(self.flattened_size, 128)
        # ReLU激活函数
        self.relu3 = nn.ReLU()
        # 第二个全连接层：将128维特征映射到类别数量
        self.fc2 = nn.Linear(128, num_classes)
        # Softmax激活函数：将输出转换为概率分布，所有类别概率和为1
        self.softmax = nn.Softmax(dim=1)
        
        # 初始化网络权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化方法初始化模型权重"""
        # 遍历模型的所有模块（层）
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # 使用Xavier均匀分布初始化卷积层权重
                nn.init.xavier_uniform_(m.weight)
                # 如果有偏置项，初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是全连接层
            elif isinstance(m, nn.Linear):
                # 使用Xavier均匀分布初始化全连接层权重
                nn.init.xavier_uniform_(m.weight)
                # 偏置项初始化为0
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播过程
        
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, 3, 64, 2048)
            
        Returns:
            torch.Tensor: 输出概率，形状为 (batch_size, num_classes)
        """
        # 第一个卷积块：卷积 -> 激活 -> 池化
        x = self.conv1(x)  # 卷积操作，形状: (batch_size, 16, 64, 2048)
        x = self.relu1(x)  # ReLU激活，形状: (batch_size, 16, 64, 2048)
        x = self.pool1(x)  # 最大池化，形状: (batch_size, 16, 32, 1024)
        
        # 第二个卷积块：卷积 -> 激活 -> 池化
        x = self.conv2(x)  # 卷积操作，形状: (batch_size, 32, 32, 1024)
        x = self.relu2(x)  # ReLU激活，形状: (batch_size, 32, 32, 1024)
        x = self.pool2(x)  # 最大池化，形状: (batch_size, 32, 16, 512)
        
        # 展平操作：将2D特征图转换为1D向量，为全连接层做准备
        x = x.view(x.size(0), -1)  # 展平，形状: (batch_size, 32*16*512)
        
        # 全连接层：特征映射 -> 分类
        x = self.fc1(x)      # 第一个全连接层，形状: (batch_size, 128)
        x = self.relu3(x)    # ReLU激活，形状: (batch_size, 128)
        x = self.fc2(x)      # 第二个全连接层，形状: (batch_size, num_classes)
        x = self.softmax(x)  # Softmax激活，转换为概率分布，形状: (batch_size, num_classes)
        
        return x
    
    def get_model_info(self) -> dict:
        """
        获取模型信息，包括参数数量和架构详情
        
        Returns:
            dict: 模型信息字典
        """
        # 计算总参数数量
        total_params = sum(p.numel() for p in self.parameters())
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 构建模型信息字典
        info = {
            'model_name': 'CNN 基线模型',
            'input_shape': (3, 64, 2048),  # 输入形状：3通道，64x2048像素
            'output_classes': 4,  # 输出类别数：4类故障
            'total_parameters': total_params,  # 总参数数量
            'trainable_parameters': trainable_params,  # 可训练参数数量
            'architecture': [  # 网络架构描述
                'Conv2d(3->16, 3x3) -> ReLU -> MaxPool(2x2)',  # 第一层
                'Conv2d(16->32, 3x3) -> ReLU -> MaxPool(2x2)',  # 第二层
                'Flatten -> Linear(->128) -> ReLU -> Linear(->4) -> Softmax'  # 第三层
            ]
        }
        
        return info
    
    def print_model_summary(self):
        """打印详细的模型摘要信息"""
        # 获取模型信息
        info = self.get_model_info()
        
        # 打印模型摘要
        print("=" * 60)
        print("CNN 基线模型摘要")
        print("=" * 60)
        print(f"模型名称: {info['model_name']}")
        print(f"输入形状: {info['input_shape']}")
        print(f"输出类别数: {info['output_classes']}")
        print(f"总参数数量: {info['total_parameters']:,}")  # 使用千位分隔符
        print(f"可训练参数数量: {info['trainable_parameters']:,}")
        print("\n网络架构:")
        # 打印每一层的架构
        for i, layer in enumerate(info['architecture'], 1):
            print(f"  {i}. {layer}")
        print("=" * 60)


def create_cnn_baseline_model(num_classes: int = 4, device: str = 'auto') -> Tuple[nn.Module, str]:
    """
    创建CNN基线模型，自动检测设备
    
    Args:
        num_classes (int): 故障类别数量（默认为4类）
        device (str): 使用的设备（'auto', 'cuda', 'cpu'）
        
    Returns:
        Tuple[nn.Module, str]: 模型实例和设备名称
        
    Example:
        >>> model, device = create_cnn_baseline_model()
        >>> print(f"模型创建在设备上: {device}")
    """
    # 如果指定为自动检测，则自动选择设备
    if device == 'auto':
        # 检查是否有可用的CUDA设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型实例
    model = CNNBaseline(num_classes=num_classes)
    # 将模型移动到指定设备
    model = model.to(device)
    
    print(f"CNN 基线模型创建在设备上: {device}")
    
    return model, device


def test_cnn_baseline():
    """使用随机输入测试CNN基线模型"""
    print("=== CNN 基线模型测试 ===\n")
    
    # 创建模型（强制使用CPU以避免CUDA问题）
    model, device = create_cnn_baseline_model(device='cpu')
    
    # 打印模型摘要
    model.print_model_summary()
    
    # 使用随机输入进行测试
    print("\n测试前向传播...")
    batch_size = 1  # 批次大小
    input_shape = (batch_size, 3, 64, 2048)  # 输入形状
    
    # 创建随机输入张量
    random_input = torch.randn(input_shape).to(device)
    
    print(f"输入形状: {random_input.shape}")
    
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出和: {output.sum().item():.4f} (由于softmax应该约为1.0)")
    
    # 验证输出维度
    expected_shape = (batch_size, 4)  # 期望的输出形状
    if output.shape == expected_shape:
        print(f"[OK] 输出形状正确: {output.shape}")
    else:
        print(f"[ERROR] 期望形状 {expected_shape}，实际得到 {output.shape}")
        return False
    
    # 测试不同批次大小
    print("\n测试不同批次大小...")
    for bs in [2, 4, 8]:  # 测试不同的批次大小
        test_input = torch.randn(bs, 3, 64, 2048).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        
        expected_shape = (bs, 4)
        if test_output.shape == expected_shape:
            print(f"  批次大小 {bs}: [OK] {test_output.shape}")
        else:
            print(f"  批次大小 {bs}: [ERROR] 期望 {expected_shape}，实际得到 {test_output.shape}")
            return False
    
    # 测试模型参数
    print("\n测试模型参数...")
    param_count = sum(p.numel() for p in model.parameters())  # 计算总参数数量
    print(f"总参数数量: {param_count:,}")  # 使用千位分隔符
    
    if param_count > 0:
        print("[OK] 模型有参数")
    else:
        print("[ERROR] 模型没有参数")
        return False
    
    print(f"\n=== 测试成功完成 ===")
    return True


if __name__ == "__main__":
    # 运行模型测试
    success = test_cnn_baseline()
    
    if success:
        print("\n[成功] CNN 基线模型测试通过！")
    else:
        print("\n[错误] CNN 基线模型测试失败！")
    
    # 额外测试：为不同类别数量创建模型
    print("\n=== 测试不同类别数量 ===")
    for num_classes in [2, 3, 5]:  # 测试2类、3类、5类
        try:
            model, device = create_cnn_baseline_model(num_classes=num_classes, device='cpu')
            test_input = torch.randn(1, 3, 64, 2048).to(device)
            
            with torch.no_grad():  # 不计算梯度
                output = model(test_input)
            
            expected_shape = (1, num_classes)  # 期望的输出形状
            if output.shape == expected_shape:
                print(f"  {num_classes} 类: [OK] {output.shape}")
            else:
                print(f"  {num_classes} 类: [ERROR] 期望 {expected_shape}，实际得到 {output.shape}")
                
        except Exception as e:
            print(f"  {num_classes} 类: [ERROR] {e}")
