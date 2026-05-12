"""
CWRU 轴承数据集加载器 - 适配真实 CWRU 数据集目录结构

数据集来源：CWRU (Case Western Reserve University) 凯斯西储大学轴承数据集
官方下载地址：https://engineering.case.edu/bearingdatacenter

本文件支持的目录结构（下载 CWRU 数据后解压到 p02_data/raw/cwru/）：
    p02_data/raw/cwru/
    ├── Normal Baseline Data/
    │   └── *.mat              ← 正常数据（标签 0）
    ├── 12k Drive End Bearing Fault Data/
    │   ├── 0.007 inches/
    │   │   ├── Ball/*.mat     ← 滚动体故障 0.007 英寸（标签 3）
    │   │   ├── Inner Race/*.mat  ← 内圈故障 0.007 英寸（标签 1）
    │   │   └── Outer Race/*.mat  ← 外圈故障 0.007 英寸（标签 2）
    │   ├── 0.014 inches/
    │   │   ├── Ball/*.mat
    │   │   ├── Inner Race/*.mat
    │   │   └── Outer Race/*.mat
    │   └── 0.021 inches/
    │       ├── Ball/*.mat
    │       ├── Inner Race/*.mat
    │       └── Outer Race/*.mat
    └── 12k Fan End Bearing Fault Data/
        ├── 0.007 inches/
        │   ├── Ball/*.mat
        │   ├── Inner Race/*.mat
        │   └── Outer Race/*.mat
        ├── 0.014 inches/
        │   ├── Ball/*.mat
        │   ├── Inner Race/*.mat
        │   └── Outer Race/*.mat
        └── 0.021 inches/
            ├── Ball/*.mat
            ├── Inner Race/*.mat
            └── Outer Race/*.mat

故障类型（4 分类）：
    - 标签 0: 正常 (Normal)
    - 标签 1: 内圈故障 (Inner Race Fault, IRF)
    - 标签 2: 外圈故障 (Outer Race Fault, ORF)
    - 标签 3: 滚动体故障 (Ball Fault, BF)

采样频率：
    - 驱动端 (Drive End): 12,000 Hz
    - 风扇端 (Fan End): 48,000 Hz
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Optional, Dict
import glob


# ===== 项目路径自动检测 =====
# 原理：从当前文件出发，逐层往上找，直到找到同时包含 p02_data 和 p03_code 的目录
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

# 原始数据根目录
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "p02_data", "raw", "cwru")
DATA_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "p02_data", "preprocessed")


# ===== CWRU 数据集常量 =====

# 采样频率配置（Hz）
SAMPLING_RATES = {
    "12k": 12000,
    "48k": 48000,
}

# 类别名称映射（标签 → 名称）
CLASS_NAMES = {
    0: "Normal",
    1: "Inner_Race",
    2: "Outer_Race",
    3: "Ball",
}

# 目录名 → 标签的映射（大小写不敏感）
# 用于自动识别目录结构中的故障类型
FAULT_DIR_MAPPING = {
    "normal": 0,
    "inner race": 1,
    "inner_race": 1,
    "inner": 1,
    "outer race": 2,
    "outer_race": 2,
    "outer": 2,
    "ball": 3,
}

# 默认信号处理参数
DEFAULT_SAMPLE_LENGTH = 1024  # 每个样本的点数
DEFAULT_STRIDE = 512          # 滑动窗口步长（重叠 50%）
NORMALIZATION_METHODS = ["minmax", "zscore", "none"]


# ===== 工具函数 =====

def get_data_dir(sampling_rate: str = "12k") -> str:
    """
    获取指定采样频率的原始数据根目录。

    Args:
        sampling_rate: 采样频率，"12k" 或 "48k"

    Returns:
        数据目录的绝对路径
    """
    return os.path.join(DATA_RAW_DIR, f"{sampling_rate}_DE")


def list_available_files(sampling_rate: str = "12k") -> Dict[str, List[str]]:
    """
    列出指定采样频率下所有可用的 .mat 数据文件（按故障类型分组）。

    Args:
        sampling_rate: 采样频率，"12k" 或 "48k"

    Returns:
        字典，键为故障类型名称，值为文件路径列表
    """
    data_dir = get_data_dir(sampling_rate)

    if not os.path.exists(data_dir):
        print(f"[警告] 数据目录不存在: {data_dir}")
        print("请下载 CWRU 数据集并解压到 02_data/raw/cwru/ 目录")
        return {}

    available_files = {}
    for class_name_lower, label in FAULT_DIR_MAPPING.items():
        class_name = CLASS_NAMES[label]
        pattern = os.path.join(data_dir, f"*{class_name_lower}*.mat")
        files = glob.glob(pattern)
        if files:
            available_files[class_name] = files

    return available_files


def scan_cwru_directory(data_dir: str) -> List[Tuple[str, int, str]]:
    """
    扫描 CWRU 数据集目录结构，收集所有 .mat 文件及其对应标签。

    支持的目录结构（适配用户本地真实目录）：
        ├── Normal Baseline Data/*.mat           → 标签 0
        ├── 12k Drive End Bearing Fault Data/
        │   └── (Ball | Inner Race | Outer Race)/*.mat  → 标签 1/2/3
        └── 12k Fan End Bearing Fault Data/
            └── (Ball | Inner Race | Outer Race)/*.mat

    Args:
        data_dir: 数据集根目录（p02_data/raw/cwru）

    Returns:
        [(文件路径, 标签, 类别名), ...] 的列表
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    files_found: List[Tuple[str, int, str]] = []

    # --- 第一步：扫描 Normal Baseline Data（正常数据，标签 0）---
    normal_dir = os.path.join(data_dir, "Normal Baseline Data")
    if os.path.exists(normal_dir):
        for mat_file in glob.glob(os.path.join(normal_dir, "*.mat")):
            files_found.append((mat_file, 0, "Normal"))
            print(f"  [normal] {os.path.basename(mat_file)}")

    # --- 第二步：扫描故障数据目录 ---
    # 适配 CWRU 官方目录命名（含中英文字符和不同英寸深度）
    fault_base_patterns = [
        "12k Drive End Bearing Fault Data",
        "48k Drive End Bearing Fault Data",
        "12k Fan End Bearing Fault Data",
        # 备用模式（防止目录名有细微差异）
        "*Drive End*Fault*",
        "*Fan End*Fault*",
    ]

    for pattern in fault_base_patterns:
        matched_dirs = glob.glob(os.path.join(data_dir, pattern))
        for fault_base_dir in matched_dirs:
            if not os.path.isdir(fault_base_dir):
                continue
            _scan_fault_subdirectory(fault_base_dir, files_found)

    return files_found


def _scan_fault_subdirectory(base_dir: str, files_found: List[Tuple[str, int, str]]) -> None:
    """
    递归扫描故障数据子目录，收集所有 .mat 文件。

    CWRU 目录结构可能有多层：
        12k Drive End Bearing Fault Data/
        └── 0.007 inches/
            └── Ball/
                └── *.mat

    Args:
        base_dir: 当前扫描的目录
        files_found: 收集到的文件列表（引用传递）
    """
    # 遍历当前目录中的所有子目录和文件
    for entry_name in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry_name)

        # 如果是 .mat 文件（直接放在当前目录下）
        if entry_name.lower().endswith(".mat"):
            label, class_name = _detect_label_from_path(entry_path)
            if label is not None:
                files_found.append((entry_path, label, class_name))

        # 如果是目录，递归扫描
        elif os.path.isdir(entry_path):
            # 检查目录名是否直接对应故障类型
            label, class_name = _detect_label_from_path(entry_path)
            if label is not None:
                # 目录本身是故障类型目录，扫描其中的 .mat 文件
                for mat_file in glob.glob(os.path.join(entry_path, "*.mat")):
                    files_found.append((mat_file, label, class_name))
            else:
                # 目录是深度目录（如 "0.007 inches"），继续递归
                _scan_fault_subdirectory(entry_path, files_found)


def _detect_label_from_path(file_or_dir_path: str) -> Tuple[Optional[int], Optional[str]]:
    """
    根据文件路径或目录名推断故障类型标签。

    匹配规则（按优先级）：
    1. 路径中包含 "normal" → 标签 0
    2. 路径中包含 "inner race" / "inner_race" / "inner" → 标签 1
    3. 路径中包含 "outer race" / "outer_race" / "outer" → 标签 2
    4. 路径中包含 "ball" → 标签 3

    Args:
        file_or_dir_path: 文件或目录的完整路径

    Returns:
        (标签, 类别名) 或 (None, None) 如果无法识别
    """
    path_lower = file_or_dir_path.lower()

    # 匹配顺序很重要：先匹配长的，再匹配短的，避免 "outer" 先于 "outer race" 被匹配
    # 按路径分段匹配，确保 "Ball" 不会误匹配到其他类别
    path_parts = path_lower.replace("\\", "/").split("/")

    for part in path_parts:
        part_clean = part.strip()
        if part_clean in FAULT_DIR_MAPPING:
            label = FAULT_DIR_MAPPING[part_clean]
            return label, CLASS_NAMES[label]

    return None, None


# ===== 数据集类 =====

class CWRUDataset(Dataset):
    """
    CWRU 轴承数据集加载器

    功能：
    - 自动扫描并读取真实 .mat 数据文件（支持 CWRU 官方目录结构）
    - 滑动窗口切分信号为固定长度样本
    - Min-Max 归一化到 [0, 1]
    - 自动标签识别（目录名 → 标签）

    使用示例：
        dataset = CWRUDataset(
            data_dir="p02_data/raw/cwru",
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
        load_to_memory: bool = True,
    ) -> None:
        """
        初始化 CWRU 数据集。

        Args:
            data_dir: 数据集根目录路径，默认使用 p02_data/raw/cwru/
            sample_length: 每个样本的序列长度（点数），默认 1024
            stride: 滑动窗口步长，默认 512（50% 重叠）
            normalize: 归一化方法，"minmax" | "zscore" | "none"
            load_to_memory: 是否将所有数据预加载到内存，默认 True
        """
        self.data_dir = data_dir or DATA_RAW_DIR
        self.sample_length = sample_length
        self.stride = stride
        self.normalize = normalize
        self.load_to_memory = load_to_memory

        self.samples: List[Tuple[np.ndarray, int]] = []  # (信号数据, 标签)
        self.file_info: List[Dict] = []                   # 文件元信息（用于调试）

        # 扫描目录并加载数据
        self._load_data()

    def _load_data(self) -> None:
        """扫描目录并加载所有 .mat 文件的核心逻辑。"""
        if not os.path.exists(self.data_dir):
            print(f"[错误] 数据目录不存在: {self.data_dir}")
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                "请下载 CWRU 数据集并解压到 p02_data/raw/cwru/ 目录"
            )

        print(f"\n开始扫描数据目录: {self.data_dir}")

        # 扫描收集所有 .mat 文件
        all_files = scan_cwru_directory(self.data_dir)

        if len(all_files) == 0:
            print("[错误] 未找到任何 .mat 数据文件！")
            raise FileNotFoundError(
                "未找到任何 .mat 数据文件。\n"
                "请确认 CWRU 数据集已正确解压到 p02_data/raw/cwru/ 目录"
            )

        print(f"\n共发现 {len(all_files)} 个 .mat 文件，开始加载...")

        # 逐个加载文件
        for mat_file, label, class_name in all_files:
            try:
                self._load_mat_file(mat_file, label, class_name)
            except Exception as e:
                print(f"[跳过] 无法加载 {os.path.basename(mat_file)}: {e}")

        if len(self.samples) == 0:
            print("[错误] 所有 .mat 文件加载失败！")
            raise RuntimeError("数据加载失败，请检查 .mat 文件格式")

        print(f"\n数据加载完成: 共 {len(self.samples)} 个样本")
        self._print_class_distribution()

    def _load_mat_file(self, mat_file: str, label: int, class_name: str) -> None:
        """
        从单个 .mat 文件加载数据并滑动窗口切分。

        Args:
            mat_file: .mat 文件路径
            label: 故障类型标签 (0-3)
            class_name: 故障类型名称
        """
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("需要 scipy 库读取 .mat 文件: pip install scipy")

        # 读取 .mat 文件（MATLAB 格式）
        mat_data = sio.loadmat(mat_file)

        # 从 .mat 文件中提取振动信号数据
        # CWRU 数据通常以 DE_time（驱动端）或 FE_time（风扇端）命名
        signal = None
        data_key = None

        for key in mat_data.keys():
            # 跳过 MATLAB 内部变量（以 __ 开头）
            if key.startswith("__"):
                continue

            value = mat_data[key]
            # 查找包含时域振动信号的字段（通常是 1D 或 2D 数组，且长度 > 100）
            if isinstance(value, np.ndarray) and value.ndim <= 2:
                if value.size > 100:
                    # 优先选择包含时间序列的字段
                    if "time" in key.lower() or key in ["X", "Y", "DE", "FE"]:
                        signal = value.flatten()
                        data_key = key
                        break
                    elif signal is None:
                        signal = value.flatten()
                        data_key = key

        if signal is None:
            raise ValueError(f"无法从 {mat_file} 提取振动信号数据")

        # 如果信号长度小于样本长度，直接复制填充
        if len(signal) < self.sample_length:
            signal = np.tile(signal, (self.sample_length // len(signal) + 1))[:self.sample_length]

        # 滑动窗口切分
        n_segments = 0
        for start_idx in range(0, len(signal) - self.sample_length + 1, self.stride):
            end_idx = start_idx + self.sample_length
            segment = signal[start_idx:end_idx].copy().astype(np.float32)

            # 归一化
            segment = self._normalize(segment)

            self.samples.append((segment, label))
            self.file_info.append({
                "file": os.path.basename(mat_file),
                "class": class_name,
                "label": label,
                "start_idx": start_idx,
                "data_key": data_key,
            })
            n_segments += 1

        print(f"  [{class_name:12s}] {os.path.basename(mat_file)}: {n_segments} 样本")

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        数据归一化。

        Args:
            data: 输入信号数组

        Returns:
            归一化后的信号数组
        """
        if self.normalize == "minmax":
            # Min-Max 归一化到 [0, 1]
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val > 1e-8:
                return (data - min_val) / (max_val - min_val)
            return data
        elif self.normalize == "zscore":
            # Z-Score 标准化
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-8:
                return (data - mean) / std
            return data
        else:
            # 不归一化
            return data

    def _print_class_distribution(self) -> None:
        """打印各类别的样本数量分布。"""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        print("\n" + "=" * 50)
        print("类别分布 (Class Distribution)")
        print("=" * 50)
        for label, count in sorted(class_counts.items()):
            name = CLASS_NAMES.get(label, "Unknown")
            print(f"  {label} ({name:12s}): {count:>6} 样本")
        print("=" * 50)

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本（支持索引和切片）。

        Args:
            idx: 样本索引

        Returns:
            (信号张量, 标签) 元组
            - 信号张量形状: (1, sample_length)，添加了通道维度
            - 标签: 整数 (0-3)
        """
        data, label = self.samples[idx]

        # 转换为 PyTorch 张量，添加通道维度：(sample_length,) → (1, sample_length)
        data = torch.from_numpy(data).unsqueeze(0)

        return data, label

    def get_class_name(self, label: int) -> str:
        """根据标签获取类别名称。"""
        return CLASS_NAMES.get(label, "Unknown")

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息。

        Returns:
            包含样本总数、样本长度、数据范围等统计信息的字典
        """
        if len(self.samples) == 0:
            return {}

        all_data = np.array([s[0] for s in self.samples])
        return {
            "total_samples": len(self.samples),
            "sample_length": self.sample_length,
            "data_shape": all_data.shape,
            "data_range": (float(np.min(all_data)), float(np.max(all_data))),
            "data_mean": float(np.mean(all_data)),
            "data_std": float(np.std(all_data)),
        }


# ===== 数据集划分函数 =====

def split_train_test(
    dataset: Dataset,
    test_ratio: float = 0.3,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[Subset, Subset]:
    """
    将数据集划分为训练集和测试集。

    Args:
        dataset: 完整数据集
        test_ratio: 测试集比例，默认 0.3（30% 测试，70% 训练）
        random_seed: 随机种子，用于结果复现，默认 42
        stratify: 是否分层采样（保持各类别比例一致），默认 True

    Returns:
        (训练集子集, 测试集子集)
    """
    from sklearn.model_selection import train_test_split

    n_samples = len(dataset)
    indices = list(range(n_samples))
    labels = [dataset[i][1] for i in range(n_samples)]

    if stratify:
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


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和测试集的 DataLoader。

    Args:
        dataset: 完整数据集
        batch_size: 每批次样本数，默认 32
        train_ratio: 训练集比例，默认 0.7（70% 训练，30% 测试）
        random_seed: 随机种子，默认 42
        num_workers: 数据加载线程数，默认 0（主进程加载）
        pin_memory: 是否固定内存以加速 GPU 传输，默认 True

    Returns:
        (训练集 DataLoader, 测试集 DataLoader)
    """
    train_subset, test_subset = split_train_test(
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
        drop_last=True,  # 丢弃最后一个不完整批次
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader


def print_dataset_statistics(dataset: Dataset, name: str = "Dataset") -> None:
    """
    打印数据集的统计信息。

    Args:
        dataset: 数据集实例
        name: 数据集名称（用于打印标题）
    """
    stats = dataset.get_statistics()
    print(f"\n{'='*50}")
    print(f"{name} 统计信息 (Statistics)")
    print(f"{'='*50}")
    for key, value in stats.items():
        if isinstance(value, tuple):
            print(f"  {key}: [{value[0]:.4f}, {value[1]:.4f}]")
        elif isinstance(value, (int, float)):
            print(f"  {key}: {value:,.4f}" if isinstance(value, float) else f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*50}\n")


# ===== 中文别名（兼容旧代码）=====
# 保留中文函数名作为别名，确保现有调用不会报错
# 后续应逐步迁移到英文函数名

def 获取数据目录(采样频率: str = "12k") -> str:
    """获取指定采样频率的数据目录（中文别名）。"""
    return get_data_dir(sampling_rate=采样频率)


def 列出可用数据文件(采样频率: str = "12k") -> Dict[str, List[str]]:
    """列出可用数据文件（中文别名）。"""
    return list_available_files(sampling_rate=采样频率)


def 划分训练测试集(
    dataset: Dataset,
    test_ratio: float = 0.3,
    random_seed: int = 42,
    stratify: bool = True,
) -> Tuple[Subset, Subset]:
    """划分训练测试集（中文别名）。"""
    return split_train_test(dataset, test_ratio, random_seed, stratify)


def 创建数据加载器(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """创建数据加载器（中文别名）。"""
    return create_dataloaders(dataset, batch_size, train_ratio, random_seed, num_workers, pin_memory)


def 打印数据统计(dataset: Dataset, name: str = "数据集") -> None:
    """打印数据统计（中文别名）。"""
    print_dataset_statistics(dataset, name)


# ===== 入口测试 =====
if __name__ == "__main__":
    print("=" * 60)
    print("CWRU 数据集加载器测试")
    print("=" * 60)

    try:
        # 创建数据集（使用真实数据目录）
        dataset = CWRUDataset(
            data_dir=None,  # 使用默认路径 p02_data/raw/cwru
            sample_length=1024,
            stride=512,
            normalize="minmax",
        )

        # 打印统计信息
        print_dataset_statistics(dataset, "CWRU 轴承数据集")

        # 划分数据集
        train_subset, test_subset = split_train_test(dataset, test_ratio=0.3)

        print(f"训练集大小: {len(train_subset)}")
        print(f"测试集大小: {len(test_subset)}")

        # 创建 DataLoader
        train_loader, test_loader = create_dataloaders(dataset, batch_size=16)

        # 获取一个批次进行测试
        for batch_data, batch_labels in train_loader:
            print(f"\n批次数据形状: {batch_data.shape}")
            print(f"批次标签形状: {batch_labels.shape}")
            print(f"批次标签内容: {batch_labels}")
            break

        print("\n" + "=" * 60)
        print("测试通过！数据集加载正常。")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("\n请按以下步骤操作：")
        print("1. 下载 CWRU 数据集: https://engineering.case.edu/bearingdatacenter")
        print("2. 解压到 p02_data/raw/cwru/ 目录")
        print("3. 确保目录结构包含 Normal Baseline Data 和 *Fault Data 子目录")
    except Exception as e:
        print(f"\n[未知错误] {e}")
        import traceback
        traceback.print_exc()
