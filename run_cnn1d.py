"""
1D-CNN CWRU 轴承故障诊断 - 主运行脚本

一键运行完整流程：
1. 数据加载与预处理
2. 模型训练
3. 测试评估
4. 结果可视化

使用方法：
    # 运行完整流程（使用模拟数据测试）
    python run_cnn1d.py

    # 指定真实数据路径
    python run_cnn1d.py --data_dir p02_data/raw/cwru/12kHz_DE

    # 自定义参数
    python run_cnn1d.py --epochs 100 --batch_size 64 --lr 0.0005

    # 仅评估已有模型
    python run_cnn1d.py --mode eval --model_path p04_models_ckpt/cnn1d_baseline.pth
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目路径
_current_file = os.path.abspath(__file__)
_current_dir = os.path.dirname(_current_file)
sys.path.insert(0, _current_dir)

from p03_code.p03_train.train_cnn1d import main_training_function as train_main
from p03_code.p04_test.eval_cnn1d import quick_eval


# =============================================================================
# 路径配置
# =============================================================================
PROJECT_ROOT = _current_dir
sys.path.insert(0, PROJECT_ROOT)


def print_welcome():
    """打印欢迎信息和实验配置"""
    print("\n" + "=" * 70)
    print("=" * 70)
    print("       CWRU 轴承故障诊断 - 1D-CNN Baseline 模型")
    print("=" * 70)
    print()
    print("  任务: 4分类轴承故障诊断")
    print("  类别: 正常(Normal) | 内圈故障(Inner Race) | 外圈故障(Outer Race) | 滚动体故障(Ball)")
    print()
    print("  模型: 1D-CNN (标准 Baseline)")
    print("  结构: Conv1d x 3 + BatchNorm + MaxPool + GlobalAvgPool + FC")
    print()
    print("=" * 70)


def print_complete(training_time, test_metrics):
    """打印训练完成信息"""
    print("\n" + "=" * 70)
    print("  训练完成!")
    print("=" * 70)
    print(f"\n  [RESULT] 最终测试结果:")
    print(f"     - Accuracy:  {test_metrics['accuracy']:.2f}%")
    print(f"     - Precision: {test_metrics['precision']:.2f}%")
    print(f"     - Recall:    {test_metrics['recall']:.2f}%")
    print(f"     - F1-Score:  {test_metrics['f1_score']:.2f}%")
    print(f"\n  [TIME] 训练时间: {training_time:.2f} 秒")
    print()
    print("  [FILES] 输出文件:")
    print("     - Model: p04_models_ckpt/cnn1d_baseline.pth")
    print("     - Training Curve: p06_results/figures/*_training_curve.png")
    print("     - Confusion Matrix: p06_results/figures/*_confusion_matrix.png")
    print("     - Metrics: p06_results/*_metrics.json")
    print("=" * 70)
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CWRU 轴承故障诊断 1D-CNN Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings (simulated data)
  python run_cnn1d.py

  # Use real data directory
  python run_cnn1d.py --data_dir p02_data/raw/cwru/12kHz_DE

  # Custom training parameters
  python run_cnn1d.py --epochs 100 --batch_size 64 --lr 0.0005

  # Eval only
  python run_cnn1d.py --mode eval --model_path p04_models_ckpt/cnn1d_baseline.pth
        """,
    )

    # 模式选择
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="run mode: train or eval",
    )

    # 数据配置
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "p02_data", "raw", "cwru"),
        help="数据目录路径 (默认: p02_data/raw/cwru)",
    )
    parser.add_argument(
        "--sample_length",
        type=int,
        default=1024,
        help="sample length (default: 1024)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="sliding window stride (default: 512)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax",
        choices=["minmax", "zscore", "none"],
        help="normalization method (default: minmax)",
    )

    # 数据集划分
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="train set ratio (default: 0.7)",
    )

    # 训练配置
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="optimizer (default: adam)",
    )

    # 学习率调度
    parser.add_argument(
        "--scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "none"],
        help="lr scheduler (default: step)",
    )

    # 早停配置
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="early stopping patience (default: 15)",
    )

    # 其他配置
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="log interval (default: 10)",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="force CPU mode",
    )

    # 模型路径
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "p04_models_ckpt", "cnn1d_baseline.pth"),
        help="model save/load path",
    )

    args = parser.parse_args()

    # 打印欢迎信息
    print_welcome()

    # 确保输出目录存在
    os.makedirs(os.path.join(PROJECT_ROOT, "p04_models_ckpt"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "p05_logs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "p06_results", "figures"), exist_ok=True)

    if args.mode == "train":
        # ===== 训练模式 =====
        print("[MODE] Training Mode\n")

        # 打印配置
        print("Training Config:")
        print(f"  - Data Dir: {args.data_dir or '[SIMULATED DATA - no real data found]'}")
        print(f"  - Sample Length: {args.sample_length}")
        print(f"  - Stride: {args.stride}")
        print(f"  - Normalize: {args.normalize}")
        print(f"  - Train Ratio: {args.train_ratio}")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Learning Rate: {args.lr}")
        print(f"  - Optimizer: {args.optimizer}")
        print(f"  - Scheduler: {args.scheduler}")
        print(f"  - Early Stopping Patience: {args.patience}")
        print(f"  - Random Seed: {args.seed}")
        print(f"  - GPU: {'enabled' if not args.no_gpu else 'disabled'}")
        print()

        # 运行训练
        start_time = datetime.now()
        model, test_metrics, train_history = train_main(
            data_path=args.data_dir,
            sample_length=args.sample_length,
            stride=args.stride,
            normalize=args.normalize,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler,
            early_stopping_patience=args.patience,
            random_seed=args.seed,
            log_interval=args.log_interval,
            model_save_path=args.model_path,
            use_gpu=not args.no_gpu,
        )

        training_time = (datetime.now() - start_time).total_seconds()

        # 打印完成信息
        print_complete(training_time, test_metrics)

    else:
        # ===== 评估模式 =====
        print("[MODE] Evaluation Mode\n")

        if not os.path.exists(args.model_path):
            print(f"[ERROR] Model not found: {args.model_path}")
            print("Please train first:")
            print(f"  python run_cnn1d.py --data_dir {args.data_dir}")
            return

        quick_eval(
            model_path=args.model_path,
            data_path=args.data_dir,
            output_dir=os.path.join(PROJECT_ROOT, "p06_results"),
        )


if __name__ == "__main__":
    main()
