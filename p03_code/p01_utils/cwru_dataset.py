"""
CWRU 轴承数据集加载器

数据集：CWRU (Case Western Reserve University) 凯斯西储大学轴承数据集
https://engineering.case.edu/bearingdatacenter

故障类型：
    - 0: 正常 (Normal)
    - 1: 内圈故障 (Inner Race Fault, IRF)
    - 2: 外圈故障 (Outer Race Fault, ORF)
    - 3: 滚动体故障 (Ball Fault, BF)

数据格式：
    - 驱动端采样频率: 12,000 Hz
    - 风扇端采样频率: 48,000 Hz
    - 原始数据格式: .mat 文件 (MATLAB 格式)

数据集目录结构 (待创建):
    p02_data/raw/cwru/
    ├── 12kHz_DE/
    │   ├── normal_0.mat
    │   ├── inner_race_0.mat
    │   ├── outer_race_0.mat
    │   └── ball_0.mat
    └── 48kHz_DE/
        ├── normal_0.mat
        ├── inner_race_0.mat
        └── ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional, Dict
import glob


# ===== 数据集路径配置 =====
# 项目根目录（自动检测）
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
_search_dir = _current_dir
PROJECT_ROOT = None

while _search_dir != os.path.dirname(_search_dir):
    _parent = os.path.dirname(_search_dir)
    if (os.path.exists(os.path.join(_parent, "p02_data")) and
            os.path.exists(os.path.join(_parent, "p03_code"))):
        PROJECT_ROOT = _parent
        break
    _search_dir = _parent

if PROJECT_ROOT is None:
    PROJECT_ROOT = os.path.dirname(_current_dir)

# 原始数据目录
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "p02_data", "raw", "cwru")
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "p02_data", "preprocessed")

# ===== CWRU 数据集元信息 =====
# 采样频率配置（Hz）
SAMPLING_RATES = {
    "12k": 12000,
    "48k": 48000,
}

# 类别名称映射
CLASS_NAMES = {
    0: "Normal",
    1: "Inner_Race",
    2: "Outer_Race",
    3: "Ball",
}

# 类别到标签的映射
# CWRU 数据集中 .mat 文件内变量的命名规则
CLASS_MAPPING = {
    "normal": 0,
    "inner_race": 1,
    "outer_race": 2,
    "ball": 3,
}

# ===== 信号处理参数 =====
# 默认样本长度
DEFAULT_SAMPLE_LENGTH = 1024
# 默认步长（滑动窗口）
DEFAULT_STRIDE = 512
# 默认数据归一化方法
NORMALIZATION_METHODS = ["minmax", "zscore", "none"]


def 获取数据目录(采样频率: str = "12k") -> str:
    """
    获取指定采样频率的数据目录

    Args:
        采样频率: "12k" 或 "48k"

    Returns:
        数据目录路径
    """
    return os.path.join(DATA_RAW_DIR, f"{采样频率}_DE")


def 列出可用数据文件(采样频率: str = "12k") -> Dict[str, List[str]]:
    """
    列出指定采样频率下所有可用的 .mat 数据文件

    Args:
        采样频率: "12k" 或 "48k"

    Returns:
        字典，键为故障类型，值为文件路径列表
    """
    data_dir = 获取数据目录(采样频率)

    if not os.path.exists(data_dir):
        print(f"[警告] 数据目录不存在: {data_dir}")
        print("请下载 CWRU 数据集并放置到 02_data/raw/cwru/ 目录")
        return {}

    available_files = {}
    for class_name, label in CLASS_MAPPING.items():
        pattern = os.path.join(data_dir, f"*{class_name}*.mat")
        files = glob.glob(pattern)
        if files:
            available_files[class_name] = files

    return available_files


class CWRUDataset(Dataset):
    """
    CWRU 轴承数据集加载器

    支持：
    - 从 .mat 文件加载振动信号
    - 自动分段（滑动窗口）
    - 数据归一化
    - 数据增强（可选）

    使用示例：
        dataset = CWRUDataset(
            data_dir="02_data/raw/cwru/12kHz_DE",
            sample_length=1024,
            stride=512,
            normalize="minmax",
        )
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        sample_length: int = DEFAULT_SAMPLE_LENGTH,
        stride: int = DEFAULT_STRIDE,
        normalize: str = "minmax",
        采样频率: str = "12k",
        数据文件列表: Optional[List[Tuple[str, str]]] = None,
        验证比例: float = 0.0,
        加载所有数据到内存: bool = True,
    ) -> None:
        """
        初始化 CWRU 数据集

        Args:
            data_dir: 数据目录路径，默认使用 PROJECT_ROOT/02_data/raw/cwru/
            sample_length: 每个样本的序列长度，默认 1024
            stride: 滑动窗口步长，默认 512
            normalize: 归一化方法，"minmax", "zscore", 或 "none"
            采样频率: 采样频率设置，"12k" 或 "48k"
            数据文件列表: 手动指定数据文件 [(文件路径, 类别名), ...]
            验证比例: 验证集比例（暂未使用）
            加载所有数据到内存: 是否将所有数据加载到内存
        """
        self.sample_length = sample_length
        self.stride = stride
        self.normalize = normalize
        self.data_dir = data_dir or 获取数据目录(采样频率)
        self.加载所有数据到内存 = 加载所有数据到内存

        self.samples = []  # 存储 (数据, 标签) 元组
        self.file_info = []  # 存储文件信息用于调试

        # 如果提供了数据文件列表，直接使用
        if 数据文件列表 is not None:
            self._从文件列表加载(数据文件列表)
        elif os.path.exists(self.data_dir):
            self._扫描并加载数据目录()
        else:
            print(f"[警告] 数据目录不存在: {self.data_dir}")
            self._创建模拟数据()

    def _扫描并加载数据目录(self) -> None:
        """
        扫描数据目录并加载所有 .mat 文件
        支持CWRU官方目录结构
        """
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("需要 scipy 库来读取 .mat 文件: pip install scipy")

        print(f"\n正在扫描数据目录: {self.data_dir}")

        # 首先尝试标准结构（12kHz_DE/normal_0.mat等）
        found_files = False
        for class_name, label in CLASS_MAPPING.items():
            # 匹配文件名的模式
            patterns = [
                f"*{class_name}*.mat",
                f"*{class_name.replace('_', '')}*.mat",
            ]

            for pattern in patterns:
                mat_files = glob.glob(os.path.join(self.data_dir, pattern))
                for mat_file in mat_files:
                    try:
                        self._从mat文件加载(mat_file, label, class_name)
                        found_files = True
                    except Exception as e:
                        print(f"[跳过] 无法加载 {mat_file}: {e}")
                        continue

        # 如果标准结构没找到，尝试CWRU官方目录结构
        if not found_files:
            print("[信息] 标准目录结构未找到，尝试扫描CWRU官方目录结构...")
            self._扫描CWRU官方结构()

        if len(self.samples) == 0:
            print("[警告] 未找到任何数据文件，将使用模拟数据")
            self._创建模拟数据()
        else:
            print(f"加载完成: 共 {len(self.samples)} 个样本")
            self._打印类别分布()

    def _扫描CWRU官方结构(self) -> None:
        """
        扫描CWRU官方目录结构
        支持的结构：
        - Normal Baseline Data/*.mat
        - 12k Drive End Bearing Fault Data/0.007/0/Ball/*.mat
        - 12k Drive End Bearing Fault Data/0.007/0/Inner Race/*.mat
        - 12k Drive End Bearing Fault Data/0.007/0/Outer Race/*.mat
        """
        import scipy.io as sio

        cwru_root = self.data_dir

        # 扫描正常数据
        normal_dir = os.path.join(cwru_root, "Normal Baseline Data")
        if os.path.exists(normal_dir):
            print(f"找到正常数据目录: {normal_dir}")
            for mat_file in glob.glob(os.path.join(normal_dir, "*.mat")):
                try:
                    self._从mat文件加载(mat_file, 0, "normal")
                except Exception as e:
                    print(f"[跳过] 无法加载 {mat_file}: {e}")

        # 扫描故障数据
        fault_base_dirs = [
            os.path.join(cwru_root, "12k Drive End Bearing Fault Data"),
            os.path.join(cwru_root, "48k Drive End Bearing Fault Data"),
            os.path.join(cwru_root, "12k Fan End Bearing Fault Data"),
        ]

        for fault_base_dir in fault_base_dirs:
            if not os.path.exists(fault_base_dir):
                continue

            print(f"扫描故障数据目录: {fault_base_dir}")

            # 递归扫描所有.mat文件
            for mat_file in glob.glob(os.path.join(fault_base_dir, "**", "*.mat"), recursive=True):
                # 根据路径判断故障类型
                mat_file_lower = mat_file.lower()
                label = None
                class_name = None

                if "ball" in mat_file_lower:
                    label = 3
                    class_name = "ball"
                elif "inner" in mat_file_lower:
                    label = 1
                    class_name = "inner_race"
                elif "outer" in mat_file_lower:
                    label = 2
                    class_name = "outer_race"

                if label is not None:
                    try:
                        self._从mat文件加载(mat_file, label, class_name)
                    except Exception as e:
                        print(f"[跳过] 无法加载 {mat_file}: {e}")

    def _从mat文件加载(self, mat_file: str, label: int, class_name: str) -> None:
        """
        从 .mat 文件加载数据

        Args:
            mat_file: .mat 文件路径
            label: 类别标签 (0-3)
            class_name: 类别名称
        """
        import scipy.io as sio

        # 读取 .mat 文件
        mat_data = sio.loadmat(mat_file)

        # 查找数据字段（CWRU 数据格式）
        data = None
        data_key = None

        for key in mat_data.keys():
            # 跳过系统字段
            if key.startswith("__"):
                continue
            # 查找包含数据的字段
            if isinstance(mat_data[key], np.ndarray):
                if mat_data[key].ndim == 1 or (
                    mat_data[key].ndim == 2 and mat_data[key].shape[0] > 100
                ):
                    data = mat_data[key].flatten()
                    data_key = key
                    break

        if data is None:
            raise ValueError(f"无法从 {mat_file} 提取数据")

        # 滑窗分段
        n_samples = 0
        for start_idx in range(0, len(data) - self.sample_length + 1, self.stride):
            end_idx = start_idx + self.sample_length
            segment = data[start_idx:end_idx].copy()

            # 归一化
            segment = self._归一化(segment)

            self.samples.append((segment.astype(np.float32), label))
            self.file_info.append(
                {
                    "file": os.path.basename(mat_file),
                    "class": class_name,
                    "label": label,
                    "start_idx": start_idx,
                }
            )
            n_samples += 1

        print(f"  [{class_name}] {os.path.basename(mat_file)}: {n_samples} 样本")

    def _从文件列表加载(self, 文件列表: List[Tuple[str, str]]) -> None:
        """
        从文件列表加载数据

        Args:
            文件列表: [(文件路径, 类别名), ...]
        """
        for file_path, class_name in 文件列表:
            if class_name not in CLASS_MAPPING:
                print(f"[跳过] 未知类别: {class_name}")
                continue
            label = CLASS_MAPPING[class_name]
            self._从mat文件加载(file_path, label, class_name)

    def _创建模拟数据(self) -> None:
        """
        创建模拟数据用于测试（当真实数据不存在时）
        """
        print("\n[信息] 正在创建模拟数据进行测试...")
        print("      真实使用时，请下载 CWRU 数据集并放置到 02_data/raw/cwru/")

        np.random.seed(42)

        # 每个类别生成相同数量的样本
        samples_per_class = 200

        for label in range(4):
            class_name = CLASS_NAMES[label]

            for i in range(samples_per_class):
                # 模拟不同故障类型的振动信号特征
                t = np.linspace(0, 1, self.sample_length)

                # 基础信号（模拟不同频率成分）
                freq_base = 30 + label * 20  # 不同故障有不同的基频
                signal = np.sin(2 * np.pi * freq_base * t)

                # 添加不同故障的特征频率成分
                if label == 1:  # 内圈故障 - 高频冲击
                    fault_freq = 150
                    signal += 0.5 * np.sin(2 * np.pi * fault_freq * t)
                    signal += 0.3 * np.random.randn(self.sample_length)
                elif label == 2:  # 外圈故障 - 调制特征
                    mod_freq = 80
                    carrier = np.sin(2 * np.pi * 120 * t)
                    signal += 0.4 * carrier * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
                elif label == 3:  # 滚动体故障 - 随机冲击
                    signal += 0.6 * np.random.randn(self.sample_length)
                else:  # 正常 - 轻微噪声
                    signal += 0.1 * np.random.randn(self.sample_length)

                # 归一化
                signal = self._归一化(signal)
                self.samples.append((signal.astype(np.float32), label))

        print(f"模拟数据创建完成: 共 {len(self.samples)} 个样本")
        self._打印类别分布()

    def _归一化(self, data: np.ndarray) -> np.ndarray:
        """
        数据归一化

        Args:
            data: 输入数据

        Returns:
            归一化后的数据
        """
        if self.normalize == "minmax":
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val > 1e-8:
                return (data - min_val) / (max_val - min_val)
            return data
        elif self.normalize == "zscore":
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-8:
                return (data - mean) / std
            return data
        else:  # "none"
            return data

    def _打印类别分布(self) -> None:
        """
        打印类别分布统计
        """
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        print("\n类别分布:")
        for label, count in sorted(class_counts.items()):
            class_name = CLASS_NAMES.get(label, "Unknown")
            print(f"  {label} ({class_name:12s}): {count:>5} 样本")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (信号数据, 标签) 元组
        """
        data, label = self.samples[idx]

        # 转换为 PyTorch 张量，添加通道维度
        data = torch.from_numpy(data).unsqueeze(0)  # shape: (1, sample_length)

        return data, label

    def 获取类别名称(self, label: int) -> str:
        """获取类别名称"""
        return CLASS_NAMES.get(label, "Unknown")

    def 获取样本统计(self) -> Dict:
        """获取数据集统计信息"""
        if len(self.samples) == 0:
            return {}

        all_data = np.array([s[0] for s in self.samples])
        return {
            "样本总数": len(self.samples),
            "样本长度": self.sample_length,
            "数据形状": all_data.shape,
            "数据范围": (float(np.min(all_data)), float(np.max(all_data))),
            "数据均值": float(np.mean(all_data)),
            "数据标准差": float(np.std(all_data)),
        }


def 划分训练测试集(
    dataset: Dataset,
    test_ratio: float = 0.3,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[Subset, Subset]:
    """
    划分训练集和测试集

    Args:
        dataset: 完整数据集
        test_ratio: 测试集比例，默认 0.3 (30%)
        random_seed: 随机种子，默认 42
        stratify: 是否分层采样（保持类别比例）

    Returns:
        (训练集子集, 测试集子集)
    """
    from sklearn.model_selection import train_test_split

    n_samples = len(dataset)
    indices = list(range(n_samples))
    labels = [dataset[i][1] for i in range(n_samples)]

    if stratify:
        # 分层采样
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=labels,
        )
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=random_seed,
        )

    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)

    return train_subset, test_subset


def 创建数据加载器(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和测试集 DataLoader

    Args:
        dataset: 完整数据集
        batch_size: 批次大小
        train_ratio: 训练集比例，默认 0.7 (70% 训练，30% 测试)
        random_seed: 随机种子
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存

    Returns:
        (训练集 DataLoader, 测试集 DataLoader)
    """
    train_subset, test_subset = 划分训练测试集(
        dataset,
        test_ratio=1 - train_ratio,
        random_seed=random_seed,
        stratify=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def 打印数据统计(dataset: Dataset, name: str = "数据集") -> None:
    """
    打印数据集统计信息

    Args:
        dataset: 数据集实例
        name: 数据集名称
    """
    stats = dataset.获取样本统计()
    print(f"\n{'='*50}")
    print(f"{name} 统计信息")
    print(f"{'='*50}")
    for key, value in stats.items():
        if isinstance(value, tuple):
            print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:,.4f}" if isinstance(value, float) else f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 测试数据加载器
    print("测试 CWRU 数据集加载器...\n")

    # 创建数据集
    dataset = CWRUDataset(
        data_dir=None,
        sample_length=1024,
        stride=512,
        normalize="minmax",
    )

    # 打印统计信息
    打印数据统计(dataset, "CWRU 轴承数据集")

    # 划分数据集
    train_subset, test_subset = 划分训练测试集(dataset, test_ratio=0.3)

    print(f"训练集大小: {len(train_subset)}")
    print(f"测试集大小: {len(test_subset)}")

    # 测试数据加载器
    train_loader, test_loader = 创建数据加载器(
        dataset,
        batch_size=16,
        train_ratio=0.7,
    )

    # 获取一个批次
    for batch_data, batch_labels in train_loader:
        print(f"\n批次形状: {batch_data.shape}")
        print(f"标签形状: {batch_labels.shape}")
        print(f"标签内容: {batch_labels}")
        break
