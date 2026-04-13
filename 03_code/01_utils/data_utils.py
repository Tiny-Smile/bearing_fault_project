"""
CWRU轴承数据集读取与批量处理工具模块
作者：轴承故障诊断项目
功能：提供CWRU数据集的读取、预处理和批量处理功能
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

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

print(f"检测到的项目根目录: {PROJECT_ROOT}")
print(f"数据根目录: {DATA_ROOT}")


def parse_fault_type_from_path(file_path: str) -> int:
    """
    从文件路径解析故障类型标签
    
    Args:
        file_path (str): .mat文件的完整路径
        
    Returns:
        int: 故障类型标签
            0 - 正常 (Normal)
            1 - 内圈故障 (Inner Race)  
            2 - 外圈故障 (Outer Race)
            3 - 滚动体故障 (Ball)
    
    Example:
        >>> parse_fault_type_from_path(".../Normal Baseline Data/97_0.mat")
        0
        >>> parse_fault_type_from_path(".../Inner Race/118.mat")
        1
    """
    # 将路径转换为小写，便于匹配
    path_lower = file_path.lower()
    
    # 检查是否为正常数据
    if "normal" in path_lower or "baseline" in path_lower:
        return 0  # 正常状态
    
    # 检查故障类型
    elif "inner" in path_lower or "ir" in path_lower:
        return 1  # 内圈故障
    
    elif "outer" in path_lower or "or" in path_lower:
        return 2  # 外圈故障
    
    elif "ball" in path_lower:
        return 3  # 滚动体故障
    
    else:
        # 如果无法从路径判断，返回-1表示未知
        print(f"警告：无法从路径解析故障类型: {file_path}")
        return -1


def read_cwru_mat(file_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    读取单个CWRU .mat文件，提取振动信号和故障标签
    
    Args:
        file_path (str): .mat文件的完整路径
        
    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: 
            - 第一个元素：振动信号的一维numpy数组，如果读取失败则为None
            - 第二个元素：故障类型标签(0-3)，如果解析失败则为None
    
    Raises:
        FileNotFoundError: 当文件不存在时抛出
        ValueError: 当文件格式不正确时抛出
    
    Example:
        >>> signal, label = read_cwru_mat("./02_data/raw/cwru/Normal Baseline Data/97_0.mat")
        >>> print(f"信号长度: {len(signal)}, 故障标签: {label}")
        信号长度: 121104, 故障标签: 0
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件扩展名
        if not file_path.endswith('.mat'):
            raise ValueError(f"文件格式错误，需要.mat文件: {file_path}")
        
        # 使用scipy.io加载mat文件
        print(f"正在读取文件: {file_path}")
        mat_data = loadmat(file_path)
        
        # 查找驱动端振动信号 (DE_time)
        # CWRU数据集中，驱动端信号通常存储在'DE_time'键中
        if 'DE_time' in mat_data:
            signal = mat_data['DE_time']
        elif 'DE' in mat_data:
            signal = mat_data['DE']
        else:
            # 如果找不到DE_time，打印所有可用的键
            print(f"警告：在文件 {file_path} 中未找到DE_time，可用键: {list(mat_data.keys())}")
            # 尝试找到包含数据的键（排除元数据）
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
            if data_keys:
                signal = mat_data[data_keys[0]]
                print(f"使用键 '{data_keys[0]}' 作为振动信号")
            else:
                raise ValueError(f"无法在文件中找到振动信号数据: {file_path}")
        
        # 确保信号是一维数组
        if signal.ndim > 1:
            signal = signal.flatten()  # 将多维数组展平为一维
        
        # 解析故障类型标签
        label = parse_fault_type_from_path(file_path)
        
        print(f"成功读取 - 信号长度: {len(signal)}, 故障标签: {label}")
        
        return signal, label
        
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return None, None
    except ValueError as e:
        print(f"错误：{e}")
        return None, None
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
        return None, None


def batch_read_cwru(raw_dir: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    批量读取CWRU数据集目录下的所有.mat文件
    
    Args:
        raw_dir (str): CWRU原始数据目录路径 (例如: "./02_data/raw/cwru/")
        
    Returns:
        Tuple[List[np.ndarray], List[int], List[str]]:
            - 信号列表：包含所有成功读取的振动信号数组
            - 标签列表：对应的故障类型标签 (0-3)
            - 文件列表：成功读取的文件路径列表
    
    Example:
        >>> signals, labels, files = batch_read_cwru("./02_data/raw/cwru/")
        >>> print(f"成功读取 {len(signals)} 个文件")
        >>> print(f"故障类型分布: {np.bincount(labels)}")
    """
    # 初始化返回列表
    signals: List[np.ndarray] = []  # 存储所有振动信号
    labels: List[int] = []          # 存储对应的故障标签
    files: List[str] = []           # 存储成功读取的文件路径
    
    try:
        # 检查目录是否存在
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"数据目录不存在: {raw_dir}")
        
        print(f"开始批量读取目录: {raw_dir}")
        
        # 使用glob递归查找所有.mat文件
        # **/* 表示递归匹配所有子目录
        mat_files = glob.glob(os.path.join(raw_dir, "**/*.mat"), recursive=True)
        
        if not mat_files:
            print(f"警告：在目录 {raw_dir} 中未找到任何.mat文件")
            return signals, labels, files
        
        print(f"找到 {len(mat_files)} 个.mat文件")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        
        # 遍历所有找到的.mat文件
        for i, file_path in enumerate(mat_files, 1):
            print(f"\n处理进度: {i}/{len(mat_files)}")
            
            # 读取单个文件
            signal, label = read_cwru_mat(file_path)
            
            # 检查读取是否成功
            if signal is not None and label is not None and label != -1:
                signals.append(signal)      # 添加信号到列表
                labels.append(label)        # 添加标签到列表
                files.append(file_path)     # 添加文件路径到列表
                success_count += 1
            else:
                fail_count += 1
                print(f"跳过文件: {file_path}")
        
        # 打印批量读取结果统计
        print(f"\n=== 批量读取完成 ===")
        print(f"总文件数: {len(mat_files)}")
        print(f"成功读取: {success_count} 个文件")
        print(f"失败/跳过: {fail_count} 个文件")
        
        # 统计各类故障的样本数量
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            print("\n故障类型分布:")
            label_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
            for label, count in zip(unique_labels, counts):
                if label < len(label_names):
                    print(f"  {label_names[label]} (标签{label}): {count} 个样本")
        
        return signals, labels, files
        
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return signals, labels, files
    except Exception as e:
        print(f"批量读取过程中发生未知错误: {e}")
        return signals, labels, files


def test_cwru_reading():
    """
    测试CWRU数据读取功能
    包含单个文件读取和批量读取的测试用例
    """
    print("=== CWRU数据读取测试 ===\n")
    
    # 测试1: 读取单个正常数据文件
    print("测试1: 读取单个正常数据文件")
    normal_file = os.path.join(DATA_ROOT, "raw", "cwru", "Normal Baseline Data", "97_0.mat")
    
    if os.path.exists(normal_file):
        signal, label = read_cwru_mat(normal_file)
        if signal is not None:
            print(f"[OK] 单个文件读取成功")
            print(f"   文件路径: {normal_file}")
            print(f"   信号长度: {len(signal)}")
            print(f"   信号数据类型: {signal.dtype}")
            print(f"   信号范围: [{signal.min():.4f}, {signal.max():.4f}]")
            print(f"   故障标签: {label}")
        else:
            print(f"[ERROR] 单个文件读取失败")
    else:
        print(f"[ERROR] 测试文件不存在: {normal_file}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 批量读取所有数据文件
    print("测试2: 批量读取所有CWRU数据文件")
    raw_dir = os.path.join(DATA_ROOT, "raw", "cwru")
    
    if os.path.exists(raw_dir):
        signals, labels, files = batch_read_cwru(raw_dir)
        
        if signals:
            print(f"[OK] 批量读取成功")
            print(f"   总信号数量: {len(signals)}")
            print(f"   总标签数量: {len(labels)}")
            
            # 显示信号长度统计
            signal_lengths = [len(sig) for sig in signals]
            print(f"   信号长度范围: [{min(signal_lengths)}, {max(signal_lengths)}]")
            print(f"   平均信号长度: {np.mean(signal_lengths):.0f}")
            
            # 显示故障类型分布
            if labels:
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"   故障类型分布: {dict(zip(unique_labels, counts))}")
                
        else:
            print(f"[ERROR] 批量读取失败或无有效数据")
    else:
        print(f"[ERROR] 数据目录不存在: {raw_dir}")
    
    print("\n=== 测试完成 ===")


def preprocess_signal(signal: np.ndarray, fs: int = 48000) -> np.ndarray:
    """
    对原始振动信号进行预处理，转换为2048维标准化信号
    
    预处理步骤：
    1. 去趋势处理：消除基线漂移
    2. Butterworth低通滤波：截止频率1kHz，阶数4
    3. 裁剪/补零到2048维：固定长度
    4. 归一化：标准化到[-1,1]范围
    
    Args:
        signal (np.ndarray): 输入的一维振动信号
        fs (int): 采样频率，默认48000Hz
        
    Returns:
        np.ndarray: 预处理后的2048维信号
        
    Example:
        >>> original_signal = np.random.randn(10000)
        >>> processed_signal = preprocess_signal(original_signal, fs=48000)
        >>> print(f"处理后信号长度: {len(processed_signal)}")
        处理后信号长度: 2048
    """
    print(f"开始信号预处理，原始长度: {len(signal)}, 采样频率: {fs}Hz")
    
    # 步骤1: 去趋势处理 - 消除基线漂移
    # 使用线性拟合去除趋势，保留信号的波动特征
    from scipy.signal import detrend
    detrended_signal = detrend(signal, type='linear')
    print(f"步骤1 - 去趋势完成")
    
    # 步骤2: Butterworth低通滤波 - 去除高频噪声
    # 设计4阶Butterworth低通滤波器，截止频率1kHz
    nyquist_freq = fs / 2  # 奈奎斯特频率
    cutoff_freq = 1000  # 截止频率1kHz
    normalized_cutoff = cutoff_freq / nyquist_freq  # 归一化截止频率
    
    # 设计Butterworth滤波器
    b, a = butter(N=4, Wn=normalized_cutoff, btype='low', analog=False)
    
    # 应用滤波器，zero-phase滤波避免相位失真
    filtered_signal = filtfilt(b, a, detrended_signal)
    print(f"步骤2 - Butterworth低通滤波完成，截止频率: {cutoff_freq}Hz")
    
    # 步骤3: 裁剪/补零到2048维 - 统一信号长度
    target_length = 2048
    signal_length = len(filtered_signal)
    
    if signal_length > target_length:
        # 如果信号长度超过2048，从中间截取2048个点
        start_idx = (signal_length - target_length) // 2
        processed_signal = filtered_signal[start_idx:start_idx + target_length]
        print(f"步骤3 - 信号裁剪完成，从{signal_length}点裁剪到{target_length}点")
    elif signal_length < target_length:
        # 如果信号长度不足2048，在两端补零
        pad_length = target_length - signal_length
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        processed_signal = np.pad(filtered_signal, (pad_left, pad_right), mode='constant')
        print(f"步骤3 - 信号补零完成，从{signal_length}点补零到{target_length}点")
    else:
        # 如果信号长度正好是2048，直接使用
        processed_signal = filtered_signal
        print(f"步骤3 - 信号长度已是{target_length}点，无需调整")
    
    # 步骤4: 归一化到[-1,1]范围 - 消除幅度差异
    # 使用最大值归一化，确保信号在[-1,1]范围内
    max_value = np.max(np.abs(processed_signal))
    if max_value > 0:
        normalized_signal = processed_signal / max_value
    else:
        # 如果信号全为零，保持原样
        normalized_signal = processed_signal.copy()
    
    print(f"步骤4 - 归一化完成，信号范围: [{normalized_signal.min():.4f}, {normalized_signal.max():.4f}]")
    print(f"预处理完成！最终信号长度: {len(normalized_signal)}")
    
    return normalized_signal


def batch_preprocess_cwru(raw_dir: str, out_dir: str) -> Tuple[List[str], List[int]]:
    """
    批量预处理CWRU数据集的所有信号文件
    
    Args:
        raw_dir (str): 原始数据目录路径 (例如: "./02_data/raw/cwru/")
        out_dir (str): 预处理数据输出目录 (例如: "./02_data/preprocessed/cwru/")
        
    Returns:
        Tuple[List[str], List[int]]:
            - 文件名列表：成功处理的文件名（不含路径）
            - 标签列表：对应的故障类型标签
            
    Example:
        >>> files, labels = batch_preprocess_cwru("./02_data/raw/cwru/", "./02_data/preprocessed/cwru/")
        >>> print(f"成功处理 {len(files)} 个文件")
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"开始批量预处理，源目录: {raw_dir}")
    print(f"输出目录: {out_dir}")
    
    # 首先读取所有原始数据文件
    signals, labels, file_paths = batch_read_cwru(raw_dir)
    
    if not signals:
        print("错误：未找到有效的原始数据文件")
        return [], []
    
    processed_files = []
    processed_labels = []
    
    # 保存一些样本用于可视化对比
    sample_signals = []
    sample_labels = []
    sample_file_names = []
    
    print(f"\n开始处理 {len(signals)} 个信号文件...")
    
    for i, (signal, label, file_path) in enumerate(zip(signals, labels, file_paths), 1):
        print(f"\n处理进度: {i}/{len(signals)}")
        
        try:
            # 判断采样频率
            # 根据文件路径判断采样频率
            if "48k" in file_path:
                fs = 48000
            elif "12k" in file_path:
                fs = 12000
            else:
                fs = 48000  # 默认采样频率
            
            # 预处理信号
            processed_signal = preprocess_signal(signal, fs)
            
            # 生成输出文件名
            # 将相对路径转换为文件名，替换目录分隔符
            relative_path = os.path.relpath(file_path, raw_dir)
            output_filename = relative_path.replace(os.sep, '_').replace('.mat', '.npy')
            output_path = os.path.join(out_dir, output_filename)
            
            # 确保输出文件的目录存在
            output_file_dir = os.path.dirname(output_path)
            if output_file_dir:
                os.makedirs(output_file_dir, exist_ok=True)
            
            # 保存预处理后的信号为.npy文件
            np.save(output_path, processed_signal)
            
            processed_files.append(output_filename)
            processed_labels.append(label)
            
            print(f"保存成功: {output_filename}")
            
            # 保存前几个样本用于可视化对比
            if len(sample_signals) < 4:  # 最多保存4个样本
                sample_signals.append(signal)
                sample_labels.append(label)
                sample_file_names.append(os.path.basename(file_path))
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    # 打印处理结果统计
    print(f"\n=== 批量预处理完成 ===")
    print(f"总文件数: {len(signals)}")
    print(f"成功处理: {len(processed_files)} 个文件")
    print(f"失败: {len(signals) - len(processed_files)} 个文件")
    
    # 统计各类故障的样本数量
    if processed_labels:
        unique_labels, counts = np.unique(processed_labels, return_counts=True)
        print("\n预处理后故障类型分布:")
        label_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
        for label, count in zip(unique_labels, counts):
            if label < len(label_names):
                print(f"  {label_names[label]} (标签{label}): {count} 个样本")
    
    # 生成预处理前后对比图
    if sample_signals:
        print("\n生成预处理前后对比图...")
        visualize_preprocess_comparison(
            sample_signals, 
            sample_labels, 
            sample_file_names,
            save_path="./06_results/figures/preprocess_compare.png"
        )
    
    return processed_files, processed_labels


def visualize_preprocess_comparison(original_signals: List[np.ndarray], 
                              labels: List[int], 
                              file_names: List[str],
                              save_path: str = None):
    """
    可视化预处理前后的信号波形与频谱对比
    
    Args:
        original_signals (List[np.ndarray]): 原始信号列表
        labels (List[int]): 对应的故障标签
        file_names (List[str]): 文件名列表
        save_path (str): 图片保存路径
    """
    # 默认保存路径
    if save_path is None:
        figures_dir = os.path.join(PROJECT_ROOT, "06_results", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, "preprocess_compare.png")
    
    print(f"生成预处理对比图，保存到: {save_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建子图
    fig, axes = plt.subplots(len(original_signals), 4, figsize=(20, 5*len(original_signals)))
    if len(original_signals) == 1:
        axes = axes.reshape(1, -1)
    
    label_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
    
    for i, (orig_signal, label, file_name) in enumerate(zip(original_signals, labels, file_names)):
        # 预处理信号
        processed_signal = preprocess_signal(orig_signal, fs=48000)  # 默认采样频率
        
        # 计算频谱
        def compute_spectrum(sig, fs):
            """计算信号的频谱"""
            n = len(sig)
            yf = fft(sig)
            xf = fftfreq(n, 1/fs)
            # 只取正频率部分
            pos_mask = xf >= 0
            return xf[pos_mask], 2.0/n * np.abs(yf[pos_mask])
        
        # 原始信号时域
        axes[i, 0].plot(orig_signal[:1000])  # 只显示前1000个点
        axes[i, 0].set_title(f'原始时域\n{file_name}\n{label_names[label]}')
        axes[i, 0].set_xlabel('采样点')
        axes[i, 0].set_ylabel('幅度')
        axes[i, 0].grid(True)
        
        # 原始信号频域
        freq_orig, amp_orig = compute_spectrum(orig_signal, fs=48000)
        axes[i, 1].plot(freq_orig[:500], amp_orig[:500])  # 只显示到500Hz
        axes[i, 1].set_title('原始频域')
        axes[i, 1].set_xlabel('频率 (Hz)')
        axes[i, 1].set_ylabel('幅度')
        axes[i, 1].grid(True)
        
        # 预处理后信号时域
        axes[i, 2].plot(processed_signal)
        axes[i, 2].set_title(f'预处理时域\n长度: {len(processed_signal)}')
        axes[i, 2].set_xlabel('采样点')
        axes[i, 2].set_ylabel('幅度')
        axes[i, 2].grid(True)
        
        # 预处理后信号频域
        freq_proc, amp_proc = compute_spectrum(processed_signal, fs=48000)
        axes[i, 3].plot(freq_proc[:500], amp_proc[:500])  # 只显示到500Hz
        axes[i, 3].set_title('预处理频域')
        axes[i, 3].set_xlabel('频率 (Hz)')
        axes[i, 3].set_ylabel('幅度')
        axes[i, 3].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预处理对比图已保存到: {save_path}")


def test_preprocessing():
    """
    测试信号预处理功能
    包含单个信号预处理和批量预处理的测试用例
    """
    print("=== 信号预处理测试 ===\n")
    
    # 测试1: 单个信号预处理
    print("测试1: 单个信号预处理")
    
    # 读取一个测试文件
    test_file = os.path.join(DATA_ROOT, "raw", "cwru", "Normal Baseline Data", "97_0.mat")
    if os.path.exists(test_file):
        original_signal, label = read_cwru_mat(test_file)
        if original_signal is not None:
            print(f"原始信号长度: {len(original_signal)}")
            print(f"原始信号范围: [{original_signal.min():.4f}, {original_signal.max():.4f}]")
            
            # 预处理信号
            processed_signal = preprocess_signal(original_signal, fs=12000)  # 12kHz采样
            
            print(f"[OK] 单个信号预处理成功")
            print(f"处理后信号长度: {len(processed_signal)}")
            print(f"处理后信号范围: [{processed_signal.min():.4f}, {processed_signal.max():.4f}]")
        else:
            print(f"[ERROR] 无法读取测试文件")
    else:
        print(f"[ERROR] 测试文件不存在: {test_file}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 批量预处理
    print("测试2: 批量预处理CWRU数据")
    raw_dir = os.path.join(DATA_ROOT, "raw", "cwru")
    out_dir = os.path.join(DATA_ROOT, "preprocessed", "cwru")
    
    if os.path.exists(raw_dir):
        processed_files, processed_labels = batch_preprocess_cwru(raw_dir, out_dir)
        
        if processed_files:
            print(f"[OK] 批量预处理成功")
            print(f"   处理文件数量: {len(processed_files)}")
            print(f"   输出目录: {out_dir}")
            
            # 验证几个输出文件
            print(f"\n验证输出文件:")
            for i, filename in enumerate(processed_files[:3]):  # 只验证前3个
                file_path = os.path.join(out_dir, filename)
                if os.path.exists(file_path):
                    data = np.load(file_path)
                    print(f"  {filename}: 长度{len(data)}, 范围[{data.min():.4f}, {data.max():.4f}]")
                else:
                    print(f"  {filename}: 文件不存在")
        else:
            print(f"[ERROR] 批量预处理失败")
    else:
        print(f"[ERROR] 原始数据目录不存在: {raw_dir}")
    
    print("\n=== 预处理测试完成 ===")


def split_and_save_cwru(preprocessed_dir: str, 
                        out_dir: str, 
                        val_size: float = 0.1, 
                        test_size: float = 0.2, 
                        random_state: int = 42) -> Tuple[dict, pd.DataFrame]:
    """
    划分CWRU数据集并保存为NPZ格式
    
    使用分层抽样保证各类别在训练/验证/测试集中分布一致
    划分比例：训练集70%，验证集10%，测试集20%
    
    Args:
        preprocessed_dir (str): 预处理数据目录路径
        out_dir (str): 输出目录路径
        val_size (float): 验证集比例，默认0.1 (10%)
        test_size (float): 测试集比例，默认0.2 (20%)
        random_state (int): 随机种子，默认42
        
    Returns:
        Tuple[dict, pd.DataFrame]:
            - 数据集统计信息字典
            - 样本分布统计表格
            
    Example:
        >>> stats, df = split_and_save_cwru("./02_data/preprocessed/cwru/", "./02_data/preprocessed/split/")
        >>> print(f"训练集样本数: {stats['train']['total']}")
    """
    print("=== CWRU数据集划分开始 ===")
    print(f"预处理数据目录: {preprocessed_dir}")
    print(f"输出目录: {out_dir}")
    print(f"划分比例 - 训练:{(1-val_size-test_size)*100:.0f}% 验证:{val_size*100:.0f}% 测试:{test_size*100:.0f}%")
    
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    
    # 确保结果目录存在
    metrics_dir = os.path.join(PROJECT_ROOT, "06_results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. 读取所有预处理后的数据文件
    print("\n步骤1: 读取预处理数据...")
    
    # 查找所有.npy文件
    npy_files = glob.glob(os.path.join(preprocessed_dir, "**/*.npy"), recursive=True)
    
    if not npy_files:
        print(f"[ERROR] 在目录 {preprocessed_dir} 中未找到.npy文件")
        return {}, pd.DataFrame()
    
    print(f"找到 {len(npy_files)} 个预处理文件")
    
    # 加载所有数据和标签
    all_signals = []
    all_labels = []
    all_filenames = []
    
    label_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
    
    for i, npy_file in enumerate(npy_files, 1):
        try:
            # 从文件名解析标签
            filename = os.path.basename(npy_file)
            relative_path = os.path.relpath(npy_file, preprocessed_dir)
            
            # 根据文件路径解析故障类型
            label = parse_fault_type_from_path(relative_path.replace('.npy', '.mat'))
            
            if label == -1:  # 如果无法解析标签，跳过
                print(f"警告：无法解析文件 {filename} 的标签，跳过")
                continue
            
            # 加载信号数据
            signal_data = np.load(npy_file)
            
            all_signals.append(signal_data)
            all_labels.append(label)
            all_filenames.append(filename)
            
            if i % 50 == 0:
                print(f"已加载 {i}/{len(npy_files)} 个文件")
                
        except Exception as e:
            print(f"加载文件 {npy_file} 时出错: {e}")
            continue
    
    if not all_signals:
        print("[ERROR] 没有成功加载任何数据文件")
        return {}, pd.DataFrame()
    
    print(f"成功加载 {len(all_signals)} 个数据文件")
    
    # 转换为numpy数组
    X = np.array(all_signals)  # 形状: (n_samples, 2048)
    y = np.array(all_labels)    # 形状: (n_samples,)
    
    print(f"数据形状 - X: {X.shape}, y: {y.shape}")
    
    # 2. 显示原始数据分布
    print("\n步骤2: 原始数据分布分析")
    unique_labels, counts = np.unique(y, return_counts=True)
    print("原始数据故障类型分布:")
    for label, count in zip(unique_labels, counts):
        if label < len(label_names):
            print(f"  {label_names[label]} (标签{label}): {count} 个样本 ({count/len(y)*100:.1f}%)")
    
    # 3. 分层划分数据集
    print("\n步骤3: 分层划分数据集...")
    
    # 计算训练集比例
    train_size = 1.0 - val_size - test_size
    
    # 第一次划分：分离出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # 第二次划分：从剩余数据中分离出验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size/(train_size + val_size),  # 调整验证集比例
        stratify=y_temp, 
        random_state=random_state
    )
    
    print(f"划分完成:")
    print(f"  训练集: {len(X_train)} 样本 ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  验证集: {len(X_val)} 样本 ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  测试集: {len(X_test)} 样本 ({len(X_test)/len(X)*100:.1f}%)")
    
    # 4. 保存数据集为NPZ格式
    print("\n步骤4: 保存数据集...")
    
    # 保存训练集
    train_path = os.path.join(out_dir, "train.npz")
    np.savez_compressed(train_path, X=X_train, y=y_train)
    print(f"训练集已保存: {train_path}")
    
    # 保存验证集
    val_path = os.path.join(out_dir, "val.npz")
    np.savez_compressed(val_path, X=X_val, y=y_val)
    print(f"验证集已保存: {val_path}")
    
    # 保存测试集
    test_path = os.path.join(out_dir, "test.npz")
    np.savez_compressed(test_path, X=X_test, y=y_test)
    print(f"测试集已保存: {test_path}")
    
    # 5. 生成统计信息
    print("\n步骤5: 生成统计信息...")
    
    # 计算各数据集的类别分布
    def calculate_distribution(labels, name):
        unique_labels, counts = np.unique(labels, return_counts=True)
        distribution = {}
        total = len(labels)
        
        for label, count in zip(unique_labels, counts):
            if label < len(label_names):
                distribution[label_names[label]] = {
                    'count': count,
                    'percentage': count/total*100
                }
        
        distribution['total'] = total
        return distribution
    
    # 统计各数据集分布
    train_dist = calculate_distribution(y_train, "训练集")
    val_dist = calculate_distribution(y_val, "验证集")
    test_dist = calculate_distribution(y_test, "测试集")
    
    # 创建统计信息字典
    stats = {
        'train': train_dist,
        'val': val_dist,
        'test': test_dist,
        'total': {
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test),
            'overall': len(X)
        }
    }
    
    # 创建详细的统计表格
    print("各数据集类别分布:")
    
    # 准备表格数据
    table_data = []
    for label_idx in range(len(label_names)):
        label_name = label_names[label_idx]
        row = {
            '故障类型': label_name,
            '标签': label_idx,
            '训练集数量': train_dist.get(label_name, {}).get('count', 0),
            '训练集比例(%)': f"{train_dist.get(label_name, {}).get('percentage', 0):.1f}",
            '验证集数量': val_dist.get(label_name, {}).get('count', 0),
            '验证集比例(%)': f"{val_dist.get(label_name, {}).get('percentage', 0):.1f}",
            '测试集数量': test_dist.get(label_name, {}).get('count', 0),
            '测试集比例(%)': f"{test_dist.get(label_name, {}).get('percentage', 0):.1f}",
        }
        table_data.append(row)
    
    # 添加总计行
    total_row = {
        '故障类型': '总计',
        '标签': '',
        '训练集数量': len(X_train),
        '训练集比例(%)': '100.0',
        '验证集数量': len(X_val),
        '验证集比例(%)': '100.0',
        '测试集数量': len(X_test),
        '测试集比例(%)': '100.0',
    }
    table_data.append(total_row)
    
    # 创建DataFrame
    df_stats = pd.DataFrame(table_data)
    
    # 保存统计表格
    metrics_dir = os.path.join(PROJECT_ROOT, "06_results", "metrics")
    csv_path = os.path.join(metrics_dir, "dataset_split.csv")
    df_stats.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"统计表格已保存: {csv_path}")
    
    # 显示统计表格
    print("\n数据集划分统计表:")
    print(df_stats.to_string(index=False))
    
    # 6. 验证保存的文件
    print("\n步骤6: 验证保存的文件...")
    
    for split_name, file_path in [("训练集", train_path), ("验证集", val_path), ("测试集", test_path)]:
        if os.path.exists(file_path):
            loaded_data = np.load(file_path)
            X_loaded, y_loaded = loaded_data['X'], loaded_data['y']
            print(f"{split_name}验证成功: X{X_loaded.shape}, y{y_loaded.shape}")
        else:
            print(f"[ERROR] {split_name}文件不存在: {file_path}")
    
    print("\n=== 数据集划分完成 ===")
    
    return stats, df_stats


def load_split_data(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载划分好的数据集NPZ文件
    
    Args:
        npz_path (str): NPZ文件路径
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: 特征数据，形状为 (n_samples, 2048)
            - y: 标签数据，形状为 (n_samples,)
            
    Example:
        >>> X_train, y_train = load_split_data("./02_data/preprocessed/split/train.npz")
        >>> print(f"训练集形状: X{X_train.shape}, y{y_train.shape}")
    """
    try:
        data = np.load(npz_path)
        X = data['X']
        y = data['y']
        
        print(f"成功加载数据集: {npz_path}")
        print(f"数据形状 - X: {X.shape}, y: {y.shape}")
        
        # 显示标签分布
        unique_labels, counts = np.unique(y, return_counts=True)
        label_names = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
        
        print("标签分布:")
        for label, count in zip(unique_labels, counts):
            if label < len(label_names):
                print(f"  {label_names[label]} (标签{label}): {count} 个样本")
        
        return X, y
        
    except Exception as e:
        print(f"[ERROR] 加载数据集失败: {e}")
        return np.array([]), np.array([])


def test_dataset_split():
    """
    测试数据集划分功能
    """
    print("=== 数据集划分测试 ===\n")
    
    # 测试1: 数据集划分
    print("测试1: 数据集划分功能")
    
    preprocessed_dir = os.path.join(DATA_ROOT, "preprocessed", "cwru")
    out_dir = os.path.join(DATA_ROOT, "preprocessed", "split")
    
    if os.path.exists(preprocessed_dir):
        try:
            stats, df_stats = split_and_save_cwru(
                preprocessed_dir, 
                out_dir, 
                val_size=0.1, 
                test_size=0.2, 
                random_state=42
            )
            
            if stats:
                print(f"[OK] 数据集划分成功")
                print(f"   训练集: {stats['total']['train']} 样本")
                print(f"   验证集: {stats['total']['val']} 样本")
                print(f"   测试集: {stats['total']['test']} 样本")
                print(f"   总计: {stats['total']['overall']} 样本")
            else:
                print(f"[ERROR] 数据集划分失败")
                
        except Exception as e:
            print(f"[ERROR] 数据集划分过程中出错: {e}")
    else:
        print(f"[ERROR] 预处理数据目录不存在: {preprocessed_dir}")
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 数据加载功能
    print("测试2: 数据加载功能")
    
    try:
        # 加载训练集
        X_train, y_train = load_split_data(os.path.join(out_dir, "train.npz"))
        
        # 加载验证集
        X_val, y_val = load_split_data(os.path.join(out_dir, "val.npz"))
        
        # 加载测试集
        X_test, y_test = load_split_data(os.path.join(out_dir, "test.npz"))
        
        if len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0:
            print(f"[OK] 数据加载功能正常")
            print(f"   所有数据集均成功加载")
            
            # 验证数据形状
            assert X_train.shape[1] == 2048, "训练集特征维度不正确"
            assert X_val.shape[1] == 2048, "验证集特征维度不正确"
            assert X_test.shape[1] == 2048, "测试集特征维度不正确"
            
            print(f"   特征维度验证通过: 2048维")
            
        else:
            print(f"[ERROR] 数据加载失败")
            
    except Exception as e:
        print(f"[ERROR] 数据加载测试失败: {e}")
    
    print("\n=== 数据集划分测试完成 ===")


# 当直接运行此脚本时，执行测试
if __name__ == "__main__":
    test_cwru_reading()
    print("\n" + "="*80 + "\n")
    test_preprocessing()
    print("\n" + "="*80 + "\n")
    test_dataset_split()
