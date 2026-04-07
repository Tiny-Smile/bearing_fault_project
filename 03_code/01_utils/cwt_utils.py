"""
Morlet小波CWT变换工具模块
作者：轴承故障诊断项目
功能：提供Morlet小波连续小波变换，生成时频图用于深度学习
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from typing import List, Tuple, Optional, Union
import glob
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
import pandas as pd

# 智能获取项目根目录
def get_project_root():
    """智能获取项目根目录"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # 向上查找项目根目录（包含02_data和03_code的目录）
    search_dir = current_dir
    while search_dir != os.path.dirname(search_dir):
        parent = os.path.dirname(search_dir)
        if (os.path.exists(os.path.join(parent, "02_data")) and 
            os.path.exists(os.path.join(parent, "03_code"))):
            return parent
        search_dir = parent
    
    # 如果没找到，返回当前目录的父目录
    return os.path.dirname(current_dir)

# 获取项目根目录和数据根目录
PROJECT_ROOT = get_project_root()
DATA_ROOT = os.path.join(PROJECT_ROOT, "02_data")

print(f"CWT - 检测到的项目根目录: {PROJECT_ROOT}")
print(f"CWT - 数据根目录: {DATA_ROOT}")


def cwt_morlet_transform(signal_data: np.ndarray, 
                        scales: Optional[np.ndarray] = None,
                        omega0: float = 5.0) -> np.ndarray:
    """
    对一维信号进行Morlet小波连续小波变换
    
    生成64×2048的时频幅度谱图，用于深度学习模型输入
    
    Args:
        signal_data (np.ndarray): 输入的一维信号，形状为 (2048,)
        scales (Optional[np.ndarray]): 尺度序列，默认64个对数尺度
        omega0 (float): Morlet小波的中心频率参数，默认5.0
        
    Returns:
        np.ndarray: CWT系数幅度谱，形状为 (64, 2048)
        
    Example:
        >>> signal_data = np.random.randn(2048)
        >>> cwt_coeffs = cwt_morlet_transform(signal_data)
        >>> print(cwt_coeffs.shape)
        (64, 2048)
    """
    if len(signal_data) != 2048:
        raise ValueError(f"输入信号长度必须为2048，当前长度: {len(signal_data)}")
    
    # 生成尺度序列
    if scales is None:
        # 生成64个对数尺度，覆盖故障冲击特征频率范围
        # 尺度范围对应频率从1Hz到1000Hz（假设采样率12kHz）
        min_scale = 1
        max_scale = 128
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 64)
    
    print(f"开始CWT变换，信号长度: {len(signal_data)}, 尺度数量: {len(scales)}")
    
    # 定义Morlet小波函数
    def morlet_wavelet(x, omega0=5.0):
        """Morlet小波函数"""
        return np.exp(1j * omega0 * x) * np.exp(-0.5 * x**2)
    
    # 执行CWT变换
    cwt_coeffs = []
    
    for i, scale in enumerate(scales):
        # 生成小波函数
        wavelet_length = int(8 * scale)  # 小波长度为尺度的8倍
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        
        # 创建时间轴
        t = np.arange(-wavelet_length//2, wavelet_length//2 + 1) / scale
        
        # 生成Morlet小波
        wavelet = morlet_wavelet(t, omega0)
        
        # 归一化小波
        wavelet = wavelet / np.sqrt(scale)
        
        # 卷积计算CWT系数
        if len(wavelet) <= len(signal_data):
            cwt_coeff = signal.convolve(signal_data, wavelet, mode='same')
        else:
            # 如果小波比信号长，进行信号卷积
            cwt_coeff = signal.convolve(signal_data, wavelet, mode='same')
        
        cwt_coeffs.append(cwt_coeff)
        
        if (i + 1) % 16 == 0:
            print(f"已完成 {i + 1}/{len(scales)} 个尺度变换")
    
    # 转换为numpy数组并计算幅度谱
    cwt_coeffs = np.array(cwt_coeffs)
    cwt_magnitude = np.abs(cwt_coeffs)
    
    print(f"CWT变换完成，输出幅度谱形状: {cwt_magnitude.shape}")
    
    return cwt_magnitude


def multi_channel_cwt(signals: List[np.ndarray], 
                     scales: Optional[np.ndarray] = None,
                     omega0: float = 5.0) -> np.ndarray:
    """
    对多通道信号进行CWT变换，生成三维时频张量
    
    Args:
        signals (List[np.ndarray]): 多通道信号列表，每个信号长度为2048
        scales (Optional[np.ndarray]): 尺度序列
        omega0 (float): Morlet小波中心频率参数
        
    Returns:
        np.ndarray: 多通道CWT幅度谱，形状为 (n_channels, 64, 2048)
        
    Example:
        >>> signals = [np.random.randn(2048) for _ in range(3)]
        >>> multi_cwt = multi_channel_cwt(signals)
        >>> print(multi_cwt.shape)
        (3, 64, 2048)
    """
    if not signals:
        raise ValueError("输入信号列表不能为空")
    
    n_channels = len(signals)
    print(f"开始多通道CWT变换，通道数: {n_channels}")
    
    # 验证所有信号长度
    for i, sig in enumerate(signals):
        if len(sig) != 2048:
            raise ValueError(f"第{i+1}个信号长度必须为2048，当前长度: {len(sig)}")
    
    # 对每个通道进行CWT变换
    channel_cwts = []
    for i, signal in enumerate(signals):
        print(f"处理第{i+1}/{n_channels}通道...")
        cwt_magnitude = cwt_morlet_transform(signal, scales, omega0)
        channel_cwts.append(cwt_magnitude)
    
    # 堆叠为三维张量
    multi_cwt = np.array(channel_cwts)
    
    print(f"多通道CWT完成，输出张量形状: {multi_cwt.shape}")
    
    return multi_cwt


def simulate_multi_channel_signal(signal: np.ndarray, n_channels: int = 3) -> List[np.ndarray]:
    """
    模拟生成多通道信号（用于单通道数据模拟多通道效果）
    
    Args:
        signal (np.ndarray): 原始单通道信号
        n_channels (int): 模拟的通道数量
        
    Returns:
        List[np.ndarray]: 多通道信号列表
        
    Example:
        >>> signal = np.random.randn(2048)
        >>> multi_signals = simulate_multi_channel_signal(signal, 3)
        >>> print(len(multi_signals))
        3
    """
    print(f"模拟生成{n_channels}通道信号...")
    
    multi_signals = []
    
    for i in range(n_channels):
        if i == 0:
            # 第一通道：原始信号
            channel_signal = signal.copy()
        else:
            # 其他通道：添加轻微变化模拟不同传感器
            # 添加噪声和相位偏移模拟多传感器差异
            noise_level = 0.05 * i  # 噪声水平递增
            phase_shift = i * 0.1    # 相位偏移
            
            # 添加高斯噪声
            noise = np.random.randn(len(signal)) * noise_level * np.std(signal)
            
            # 模拟相位偏移（通过时移近似）
            shift_samples = int(phase_shift * 10)  # 相位偏移转换为样本偏移
            shifted_signal = np.roll(signal, shift_samples)
            
            channel_signal = shifted_signal + noise
        
        multi_signals.append(channel_signal)
    
    print(f"多通道信号模拟完成，通道数: {len(multi_signals)}")
    
    return multi_signals


def batch_cwt_cwru(split_dir: str, out_dir: str, n_channels: int = 3) -> dict:
    """
    批量处理CWRU数据集的CWT变换
    
    对划分好的训练/验证/测试集进行CWT变换，生成时频图数据集
    
    Args:
        split_dir (str): 划分数据集目录路径
        out_dir (str): CWT数据集输出目录路径
        n_channels (int): 模拟的通道数量
        
    Returns:
        dict: 处理统计信息
        
    Example:
        >>> stats = batch_cwt_cwru("./02_data/preprocessed/split/", "./02_data/cwt/")
        >>> print(f"处理完成: {stats}")
    """
    print("=== 批量CWT变换开始 ===")
    print(f"输入目录: {split_dir}")
    print(f"输出目录: {out_dir}")
    print(f"模拟通道数: {n_channels}")
    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("./06_results/figures/", exist_ok=True)
    
    # 查找NPZ文件
    npz_files = glob.glob(os.path.join(split_dir, "*.npz"))
    
    if not npz_files:
        print(f"[ERROR] 在目录 {split_dir} 中未找到NPZ文件")
        return {}
    
    print(f"找到 {len(npz_files)} 个NPZ文件")
    
    stats = {}
    
    for npz_file in npz_files:
        filename = os.path.basename(npz_file)
        dataset_name = filename.replace('.npz', '')
        
        print(f"\n处理数据集: {dataset_name}")
        
        try:
            # 加载数据
            data = np.load(npz_file)
            X = data['X']  # 特征数据
            y = data['y']  # 标签数据
            
            print(f"加载数据: X{X.shape}, y{y.shape}")
            
            # 对每个样本进行CWT变换
            cwt_data = []
            valid_indices = []
            
            for i, (signal, label) in enumerate(zip(X, y)):
                try:
                    # 模拟多通道信号
                    multi_signals = simulate_multi_channel_signal(signal, n_channels)
                    
                    # 进行多通道CWT变换
                    cwt_tensor = multi_channel_cwt(multi_signals)
                    
                    cwt_data.append(cwt_tensor)
                    valid_indices.append(i)
                    
                    if (i + 1) % 20 == 0:
                        print(f"已处理 {i + 1}/{len(X)} 个样本")
                        
                except Exception as e:
                    print(f"处理第{i}个样本时出错: {e}")
                    continue
            
            if cwt_data:
                # 转换为numpy数组
                cwt_array = np.array(cwt_data)  # 形状: (n_samples, n_channels, 64, 2048)
                valid_labels = y[valid_indices]
                
                # 保存CWT数据
                output_path = os.path.join(out_dir, f"cwt_{dataset_name}.npz")
                np.savez_compressed(output_path, X=cwt_array, y=valid_labels)
                
                print(f"CWT数据已保存: {output_path}")
                print(f"输出形状: X{cwt_array.shape}, y{valid_labels.shape}")
                
                stats[dataset_name] = {
                    'input_samples': len(X),
                    'output_samples': len(cwt_array),
                    'shape': cwt_array.shape,
                    'success_rate': len(cwt_array) / len(X) * 100
                }
                
            else:
                print(f"[ERROR] 数据集 {dataset_name} 处理失败")
                
        except Exception as e:
            print(f"[ERROR] 处理文件 {npz_file} 时出错: {e}")
            continue
    
    # 生成统计信息
    print(f"\n=== 批量CWT变换完成 ===")
    for dataset, stat in stats.items():
        print(f"{dataset}: {stat['output_samples']}/{stat['input_samples']} 样本 ({stat['success_rate']:.1f}%)")
    
    return stats


def visualize_cwt_comparison(split_dir: str, 
                           cwt_dir: str,
                           save_path: str = "./06_results/figures/cwt_compare.png"):
    """
    可视化正常和故障信号的CWT时频图对比
    
    Args:
        split_dir (str): 原始数据集目录
        cwt_dir (str): CWT数据集目录
        save_path (str): 图片保存路径
    """
    print(f"生成CWT时频图对比，保存到: {save_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # 加载原始数据
        train_data = np.load(os.path.join(split_dir, "train.npz"))
        X_train = train_data['X']
        y_train = train_data['y']
        
        # 加载CWT数据
        cwt_train_data = np.load(os.path.join(cwt_dir, "cwt_train.npz"))
        X_cwt = cwt_train_data['X']
        y_cwt = cwt_train_data['y']
        
        # 找到正常和故障样本
        normal_indices = np.where(y_train == 0)[0]
        fault_indices = np.where(y_train > 0)[0]
        
        if len(normal_indices) == 0 or len(fault_indices) == 0:
            print("[ERROR] 未找到正常或故障样本")
            return
        
        # 选择样本进行对比
        normal_idx = normal_indices[0]
        fault_idx = fault_indices[0]
        
        # 创建对比图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # 正常信号
        normal_signal = X_train[normal_idx]
        normal_cwt = X_cwt[normal_idx, 0]  # 第一个通道
        
        # 故障信号
        fault_signal = X_train[fault_idx]
        fault_cwt = X_cwt[fault_idx, 0]  # 第一个通道
        
        # 绘制正常信号
        axes[0, 0].plot(normal_signal)
        axes[0, 0].set_title('正常信号 - 时域')
        axes[0, 0].set_xlabel('采样点')
        axes[0, 0].set_ylabel('幅度')
        axes[0, 0].grid(True)
        
        # 正常信号频谱
        freq = fftfreq(len(normal_signal), 1/12000)[:len(normal_signal)//2]
        spectrum = np.abs(fft(normal_signal))[:len(normal_signal)//2]
        axes[0, 1].plot(freq[:500], spectrum[:500])
        axes[0, 1].set_title('正常信号 - 频域')
        axes[0, 1].set_xlabel('频率 (Hz)')
        axes[0, 1].set_ylabel('幅度')
        axes[0, 1].grid(True)
        
        # 正常信号CWT时频图
        im1 = axes[0, 2].imshow(normal_cwt, aspect='auto', cmap='viridis', 
                               extent=[0, 2048, 64, 1])
        axes[0, 2].set_title('正常信号 - CWT时频图')
        axes[0, 2].set_xlabel('时间采样点')
        axes[0, 2].set_ylabel('尺度')
        plt.colorbar(im1, ax=axes[0, 2])
        
        # 正常信号CWT平均能量
        cwt_energy = np.mean(normal_cwt**2, axis=1)
        axes[0, 3].plot(cwt_energy)
        axes[0, 3].set_title('正常信号 - CWT能量分布')
        axes[0, 3].set_xlabel('尺度')
        axes[0, 3].set_ylabel('平均能量')
        axes[0, 3].grid(True)
        
        # 绘制故障信号
        axes[1, 0].plot(fault_signal)
        axes[1, 0].set_title('故障信号 - 时域')
        axes[1, 0].set_xlabel('采样点')
        axes[1, 0].set_ylabel('幅度')
        axes[1, 0].grid(True)
        
        # 故障信号频谱
        spectrum_fault = np.abs(fft(fault_signal))[:len(fault_signal)//2]
        axes[1, 1].plot(freq[:500], spectrum_fault[:500])
        axes[1, 1].set_title('故障信号 - 频域')
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('幅度')
        axes[1, 1].grid(True)
        
        # 故障信号CWT时频图
        im2 = axes[1, 2].imshow(fault_cwt, aspect='auto', cmap='viridis',
                               extent=[0, 2048, 64, 1])
        axes[1, 2].set_title('故障信号 - CWT时频图')
        axes[1, 2].set_xlabel('时间采样点')
        axes[1, 2].set_ylabel('尺度')
        plt.colorbar(im2, ax=axes[1, 2])
        
        # 故障信号CWT平均能量
        cwt_energy_fault = np.mean(fault_cwt**2, axis=1)
        axes[1, 3].plot(cwt_energy_fault)
        axes[1, 3].set_title('故障信号 - CWT能量分布')
        axes[1, 3].set_xlabel('尺度')
        axes[1, 3].set_ylabel('平均能量')
        axes[1, 3].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"CWT对比图已保存到: {save_path}")
        
    except Exception as e:
        print(f"[ERROR] 生成CWT对比图时出错: {e}")


def test_cwt_transform():
    """
    测试CWT变换功能
    """
    print("=== CWT变换测试 ===\n")
    
    # 测试1: 单信号CWT变换
    print("测试1: 单信号CWT变换")
    
    try:
        # 生成测试信号
        test_signal = np.random.randn(2048)
        print(f"生成测试信号，长度: {len(test_signal)}")
        
        # 进行CWT变换
        cwt_result = cwt_morlet_transform(test_signal)
        print(f"[OK] 单信号CWT变换成功")
        print(f"   输出形状: {cwt_result.shape}")
        
    except Exception as e:
        print(f"[ERROR] 单信号CWT变换失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 多通道CWT变换
    print("测试2: 多通道CWT变换")
    
    try:
        # 生成多通道测试信号
        multi_signals = [np.random.randn(2048) for _ in range(3)]
        print(f"生成多通道测试信号，通道数: {len(multi_signals)}")
        
        # 进行多通道CWT变换
        multi_cwt_result = multi_channel_cwt(multi_signals)
        print(f"[OK] 多通道CWT变换成功")
        print(f"   输出形状: {multi_cwt_result.shape}")
        
    except Exception as e:
        print(f"[ERROR] 多通道CWT变换失败: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试3: 批量CWT处理
    print("测试3: 批量CWT处理")
    
    split_dir = os.path.join(DATA_ROOT, "preprocessed", "split")
    cwt_dir = os.path.join(DATA_ROOT, "cwt")
    
    if os.path.exists(split_dir):
        try:
            stats = batch_cwt_cwru(split_dir, cwt_dir, n_channels=3)
            
            if stats:
                print(f"[OK] 批量CWT处理成功")
                for dataset, stat in stats.items():
                    print(f"   {dataset}: {stat['output_samples']} 样本, 形状 {stat['shape']}")
                
                # 生成对比图
                visualize_cwt_comparison(split_dir, cwt_dir)
                
            else:
                print(f"[ERROR] 批量CWT处理失败")
                
        except Exception as e:
            print(f"[ERROR] 批量CWT处理过程中出错: {e}")
    else:
        print(f"[ERROR] 数据集目录不存在: {split_dir}")
    
    print("\n=== CWT变换测试完成 ===")


# 当直接运行此脚本时，执行测试
if __name__ == "__main__":
    test_cwt_transform()
