"""
CNN模型测试 - 使用真实CWT时频图数据
作者：轴承故障诊断项目
功能：使用项目中生成的CWT数据直接测试CNN模型
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from typing import Tuple, Dict

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 导入模型和工具
sys.path.append(os.path.join(project_root, '03_code', '02_models'))
sys.path.append(os.path.join(project_root, '03_code', '01_utils'))

from cnn_baseline import CNNBaseline, create_cnn_baseline_model
from metric_utils import calculate_metrics
from vis_utils import plot_confusion_matrix, plot_class_metrics


class CWTDataLoader:
    """CWT时频图数据加载器"""
    
    def __init__(self, data_dir: str = "./02_data/cwt"):
        """
        初始化数据加载器
        
        Args:
            data_dir (str): CWT数据目录路径
        """
        self.data_dir = data_dir
        self.data = {}
        self._load_data()
    
    def _load_data(self):
        """加载所有CWT数据文件"""
        print("正在加载CWT数据...")
        
        # 加载训练数据
        train_path = os.path.join(self.data_dir, "cwt_train.npz")
        if os.path.exists(train_path):
            train_data = np.load(train_path)
            self.data['train'] = {
                'X': train_data['X'],  # 形状: (samples, 3, 64, 2048)
                'y': train_data['y']   # 形状: (samples,)
            }
            print(f"  训练数据: {self.data['train']['X'].shape}, 标签: {self.data['train']['y'].shape}")
        
        # 加载验证数据
        val_path = os.path.join(self.data_dir, "cwt_val.npz")
        if os.path.exists(val_path):
            val_data = np.load(val_path)
            self.data['val'] = {
                'X': val_data['X'],
                'y': val_data['y']
            }
            print(f"  验证数据: {self.data['val']['X'].shape}, 标签: {self.data['val']['y'].shape}")
        
        # 加载测试数据
        test_path = os.path.join(self.data_dir, "cwt_test.npz")
        if os.path.exists(test_path):
            test_data = np.load(test_path)
            self.data['test'] = {
                'X': test_data['X'],
                'y': test_data['y']
            }
            print(f"  测试数据: {self.data['test']['X'].shape}, 标签: {self.data['test']['y'].shape}")
        
        print("CWT数据加载完成！\n")
    
    def get_data(self, split: str = 'test') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定数据集
        
        Args:
            split (str): 数据集类型 ('train', 'val', 'test')
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (数据, 标签)
        """
        if split not in self.data:
            raise ValueError(f"数据集 '{split}' 不存在。可用选项: {list(self.data.keys())}")
        
        # 获取numpy数据
        X_numpy = self.data[split]['X']  # 形状: (samples, 3, 64, 2048)
        y_numpy = self.data[split]['y']  # 形状: (samples,)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_numpy)  # 转换为float32
        y_tensor = torch.LongTensor(y_numpy)   # 转换为long
        
        print(f"{split}数据张量:")
        print(f"  X形状: {X_tensor.shape}, 类型: {X_tensor.dtype}")
        print(f"  y形状: {y_tensor.shape}, 类型: {y_tensor.dtype}")
        print(f"  标签范围: {y_tensor.min().item()} - {y_tensor.max().item()}")
        
        return X_tensor, y_tensor


def test_cnn_with_real_data():
    """使用真实CWT数据测试CNN模型"""
    print("=" * 60)
    print("CNN模型测试 - 使用真实CWT时频图数据")
    print("=" * 60)
    
    # 1. 加载CWT数据
    data_loader = CWTDataLoader()
    
    # 2. 创建CNN模型
    print("创建CNN模型...")
    model, device = create_cnn_baseline_model(device='cpu')  # 使用CPU避免CUDA问题
    model.eval()  # 设置为评估模式
    
    # 打印模型信息
    model.print_model_summary()
    
    # 3. 测试不同数据集
    datasets = ['train', 'val', 'test']
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"测试 {dataset_name.upper()} 数据集")
        print(f"{'='*50}")
        
        try:
            # 获取数据
            X, y = data_loader.get_data(dataset_name)
            
            # 移动到设备
            X = X.to(device)
            y = y.to(device)
            
            print(f"\n开始前向传播...")
            print(f"批次大小: {X.shape[0]}")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(X)  # 输出形状: (batch_size, 4)
                predictions = torch.argmax(outputs, dim=1)  # 获取预测类别
            
            print(f"输出形状: {outputs.shape}")
            print(f"输出范围: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
            print(f"预测类别: {predictions.cpu().numpy()}")
            print(f"真实标签: {y.cpu().numpy()}")
            
            # 计算准确率
            correct = (predictions == y).sum().item()
            total = y.size(0)
            accuracy = correct / total * 100
            
            print(f"\n准确率: {correct}/{total} = {accuracy:.2f}%")
            
            # 保存结果
            results[dataset_name] = {
                'accuracy': accuracy,
                'predictions': predictions.cpu().numpy(),
                'true_labels': y.cpu().numpy(),
                'outputs': outputs.cpu().numpy()
            }
            
            print(f"[OK] {dataset_name} 数据集测试完成！")
            
        except Exception as e:
            print(f"[ERROR] {dataset_name} 数据集测试失败: {e}")
            results[dataset_name] = {'error': str(e)}
    
    # 4. 生成测试报告
    print(f"\n{'='*60}")
    print("测试结果总结")
    print(f"{'='*60}")
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"{dataset_name.upper()}: 测试失败 - {result['error']}")
        else:
            print(f"{dataset_name.upper()}: 准确率 = {result['accuracy']:.2f}%")
    
    # 5. 详细分析测试集结果
    if 'test' in results and 'accuracy' in results['test']:
        print(f"\n{'='*50}")
        print("测试集详细分析")
        print(f"{'='*50}")
        
        test_result = results['test']
        y_true = test_result['true_labels']
        y_pred = test_result['predictions']
        
        # 计算详细指标
        accuracy, f1_macro, cm, report = calculate_metrics(y_true, y_pred)
        
        print(f"准确率: {accuracy:.4f}")
        print(f"宏平均F1: {f1_macro:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        # 保存可视化结果
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
            plt.rcParams['axes.unicode_minus'] = False
            
            # 绘制混淆矩阵
            class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
            plot_confusion_matrix(
                cm, 
                "./06_results/figures/cnn_test_confusion_matrix.png",
                class_names=class_names,
                title="CNN模型测试混淆矩阵"
            )
            
            print(f"\n混淆矩阵已保存到: ./06_results/figures/cnn_test_confusion_matrix.png")
            
        except Exception as e:
            print(f"可视化保存失败: {e}")
    
    return results


def test_single_sample():
    """测试单个样本"""
    print(f"\n{'='*50}")
    print("单个样本测试")
    print(f"{'='*50}")
    
    # 加载数据
    data_loader = CWTDataLoader()
    X_test, y_test = data_loader.get_data('test')
    
    # 创建模型
    model, device = create_cnn_baseline_model(device='cpu')
    model.eval()
    
    # 选择第一个样本
    single_sample = X_test[:1]  # 形状: (1, 3, 64, 2048)
    true_label = y_test[0].item()
    
    print(f"单个样本形状: {single_sample.shape}")
    print(f"真实标签: {true_label}")
    
    # 前向传播
    with torch.no_grad():
        output = model(single_sample.to(device))
        probability = output.cpu().numpy()[0]
        predicted_class = np.argmax(probability)
    
    print(f"输出概率: {probability}")
    print(f"预测类别: {predicted_class}")
    print(f"预测正确: {'是' if predicted_class == true_label else '否'}")
    
    # 类别名称映射
    class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
    print(f"真实: {class_names[true_label]}")
    print(f"预测: {class_names[predicted_class]}")


if __name__ == "__main__":
    print("开始CNN模型真实数据测试...\n")
    
    # 运行完整测试
    results = test_cnn_with_real_data()
    
    # 运行单样本测试
    test_single_sample()
    
    print(f"\n{'='*60}")
    print("所有测试完成！")
    print(f"{'='*60}")
