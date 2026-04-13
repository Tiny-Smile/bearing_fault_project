"""
轴承故障诊断 - 完整流程运行脚本

这个脚本串联整个项目流程：
1. 数据加载和预处理
2. 数据集划分
3. CWT变换
4. 模型训练
5. 模型评估和可视化

使用方法：
    python run_all.py
"""

import os
import sys
import time
import numpy as np
import torch
print(torch.cuda.is_available())  # 输出 True 表示GPU可用
print(torch.version.cuda)         # 查看PyTorch对应的CUDA版本
print(torch.cuda.get_device_name(0))  # 查看显卡型号
# 设置项目根目录（手动指定，方便理解）
PROJECT_ROOT = r"E:\PycharmProjects\gradualationProjects\bearing_fault_project"

# 添加项目路径到系统路径
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "03_code"))
sys.path.append(os.path.join(PROJECT_ROOT, "03_code", "01_utils"))
sys.path.append(os.path.join(PROJECT_ROOT, "03_code", "02_models"))


def print_header(title: str):
    """打印一个醒目的标题"""
    print("\n")
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(step_num: int, step_name: str):
    """打印步骤编号"""
    print(f"\n>>> 第{step_num}步: {step_name}")


def 创建必要的文件夹():
    """创建项目需要的文件夹"""
    print_step(0, "创建必要的文件夹")
    
    # 需要创建的文件夹列表
    folders = [
        os.path.join(PROJECT_ROOT, "02_data", "preprocessed"),
        os.path.join(PROJECT_ROOT, "02_data", "preprocessed", "split"),
        os.path.join(PROJECT_ROOT, "02_data", "cwt"),
        os.path.join(PROJECT_ROOT, "04_models_ckpt"),
        os.path.join(PROJECT_ROOT, "05_logs"),
        os.path.join(PROJECT_ROOT, "06_results", "figures"),
        os.path.join(PROJECT_ROOT, "06_results", "metrics"),
        os.path.join(PROJECT_ROOT, "06_results", "report"),
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  创建文件夹: {folder}")
    
    print("  文件夹创建完成！")


def 第1步_加载原始数据():
    """第1步：加载原始CWRU数据"""
    print_step(1, "加载原始CWRU数据")
    
    try:
        from data_utils import read_cwru_mat, batch_read_cwru
        
        # 检查原始数据目录
        raw_dir = os.path.join(PROJECT_ROOT, "02_data", "raw", "cwru")
        
        if not os.path.exists(raw_dir):
            print(f"  [跳过] 原始数据目录不存在: {raw_dir}")
            print("  请先准备好原始CWRU数据集！")
            return None, None
        
        # 批量读取数据
        print(f"  数据目录: {raw_dir}")
        signals, labels, files = batch_read_cwru(raw_dir)
        
        if signals:
            print(f"\n  [成功] 加载了 {len(signals)} 个数据文件")
            return signals, labels
        else:
            print("  [警告] 没有加载到任何数据")
            return None, None
            
    except Exception as e:
        print(f"  [错误] 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def 第2步_预处理数据():
    """第2步：预处理原始数据"""
    print_step(2, "预处理原始数据")
    
    try:
        from data_utils import batch_preprocess_cwru
        
        # 设置输入输出目录
        raw_dir = os.path.join(PROJECT_ROOT, "02_data", "raw", "cwru")
        out_dir = os.path.join(PROJECT_ROOT, "02_data", "preprocessed", "cwru")
        
        if not os.path.exists(raw_dir):
            print(f"  [跳过] 原始数据目录不存在")
            return None, None
        
        # 检查预处理数据是否已存在
        npy_files = []
        if os.path.exists(out_dir):
            import glob
            npy_files = glob.glob(os.path.join(out_dir, "**/*.npy"), recursive=True)
        
        if npy_files:
            print(f"  [跳过] 预处理数据已存在 ({len(npy_files)} 个文件)")
            print(f"  如需重新处理，请先删除目录: {out_dir}")
            return out_dir, None
        
        # 执行预处理
        print(f"  输入目录: {raw_dir}")
        print(f"  输出目录: {out_dir}")
        processed_files, processed_labels = batch_preprocess_cwru(raw_dir, out_dir)
        
        if processed_files:
            print(f"\n  [成功] 预处理完成，共 {len(processed_files)} 个文件")
            return out_dir, processed_labels
        else:
            print("  [警告] 预处理失败")
            return None, None
            
    except Exception as e:
        print(f"  [错误] 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def 第3步_划分数据集():
    """第3步：划分训练集、验证集、测试集"""
    print_step(3, "划分数据集")
    
    try:
        from data_utils import split_and_save_cwru
        
        # 设置输入输出目录
        preprocessed_dir = os.path.join(PROJECT_ROOT, "02_data", "preprocessed", "cwru")
        out_dir = os.path.join(PROJECT_ROOT, "02_data", "preprocessed", "split")
        
        # 检查预处理数据是否存在
        if not os.path.exists(preprocessed_dir):
            print(f"  [跳过] 预处理数据目录不存在")
            return None
        
        # 检查划分数据是否已存在
        train_file = os.path.join(out_dir, "train.npz")
        val_file = os.path.join(out_dir, "val.npz")
        test_file = os.path.join(out_dir, "test.npz")
        
        if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file):
            print(f"  [跳过] 数据集已划分完成")
            print(f"  如需重新划分，请先删除文件:")
            print(f"    - {train_file}")
            print(f"    - {val_file}")
            print(f"    - {test_file}")
            return out_dir
        
        # 执行划分
        print(f"  预处理数据目录: {preprocessed_dir}")
        print(f"  输出目录: {out_dir}")
        
        stats, df_stats = split_and_save_cwru(
            preprocessed_dir,
            out_dir,
            val_size=0.15,
            test_size=0.15,
            random_state=42
        )
        
        if stats:
            print(f"\n  [成功] 数据集划分完成")
            print(f"    训练集: {stats['total']['train']} 样本")
            print(f"    验证集: {stats['total']['val']} 样本")
            print(f"    测试集: {stats['total']['test']} 样本")
            return out_dir
        else:
            print("  [警告] 数据集划分失败")
            return None
            
    except Exception as e:
        print(f"  [错误] 数据集划分失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def 第4步_CWT变换():
    """第4步：进行CWT小波变换"""
    print_step(4, "进行CWT小波变换")
    
    try:
        from cwt_utils import batch_cwt_cwru, visualize_cwt_comparison
        
        # 设置输入输出目录
        split_dir = os.path.join(PROJECT_ROOT, "02_data", "preprocessed", "split")
        cwt_dir = os.path.join(PROJECT_ROOT, "02_data", "cwt")
        
        # 检查划分数据是否存在
        train_file = os.path.join(split_dir, "train.npz")
        if not os.path.exists(train_file):
            print(f"  [跳过] 划分数据集不存在")
            return None
        
        # 检查CWT数据是否已存在
        cwt_train_file = os.path.join(cwt_dir, "cwt_train.npz")
        if os.path.exists(cwt_train_file):
            print(f"  [跳过] CWT数据已存在")
            print(f"  如需重新生成，请先删除目录: {cwt_dir}")
            return cwt_dir
        
        # 执行CWT变换
        print(f"  输入目录: {split_dir}")
        print(f"  输出目录: {cwt_dir}")
        print(f"  模拟通道数: 3")
        
        stats = batch_cwt_cwru(split_dir, cwt_dir, n_channels=3)
        
        if stats:
            print(f"\n  [成功] CWT变换完成")
            for dataset_name, stat in stats.items():
                print(f"    {dataset_name}: {stat['output_samples']} 样本")
            
            # 生成对比图
            print("\n  生成CWT对比图...")
            try:
                visualize_cwt_comparison(split_dir, cwt_dir)
                print("  [成功] CWT对比图已生成")
            except Exception as e:
                print(f"  [警告] 生成对比图失败: {e}")
            
            return cwt_dir
        else:
            print("  [警告] CWT变换失败")
            return None
            
    except Exception as e:
        print(f"  [错误] CWT变换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def 第5步_训练模型():
    """第5步：训练CNN模型"""
    print_step(5, "训练CNN模型")
    
    try:
        import torch
        from cnn_baseline import CNNBaseline
    except ImportError as e:
        print(f"  [跳过] 缺少必要依赖: {e}")
        print("  请先安装: pip install torch")
        return None
    
    try:
        # 检查CWT数据是否存在
        cwt_dir = os.path.join(PROJECT_ROOT, "02_data", "cwt")
        train_file = os.path.join(cwt_dir, "cwt_train.npz")
        val_file = os.path.join(cwt_dir, "cwt_val.npz")
        
        if not os.path.exists(train_file):
            print(f"  [跳过] CWT训练数据不存在: {train_file}")
            print("  请先完成前面的步骤生成CWT数据")
            return None
        
        # 加载数据
        print("  加载训练数据...")
        train_data = np.load(train_file)
        X_train = train_data['X']
        y_train = train_data['y']
        print(f"    训练数据: X={X_train.shape}, y={y_train.shape}")
        
        # 加载验证数据
        print("  加载验证数据...")
        if os.path.exists(val_file):
            val_data = np.load(val_file)
            X_val = val_data['X']
            y_val = val_data['y']
            print(f"    验证数据: X={X_val.shape}, y={y_val.shape}")
        else:
            print("  [警告] 验证数据不存在，跳过验证")
            X_val, y_val = None, None
        
        # 创建模型
        print("\n  创建CNN模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  使用设备: {device}")
        
        model = CNNBaseline(num_classes=4)
        model = model.to(device)
        
        # 打印模型信息
        info = model.get_model_info()
        print(f"    模型参数量: {info['total_parameters']:,}")
        
        # 训练配置
        batch_size = 32
        learning_rate = 0.001
        num_epochs = 10  # 演示用，实际可以设大一些
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print(f"\n  训练配置:")
        print(f"    批次大小: {batch_size}")
        print(f"    学习率: {learning_rate}")
        print(f"    训练轮次: {num_epochs}")
        
        # 开始训练
        print("\n  开始训练...")
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
        
        train_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs):
            model.train()
            
            # 训练一个epoch
            total_loss = 0
            correct = 0
            total = 0
            
            num_batches = (len(X_train) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                
                batch_X = X_train_tensor[start_idx:end_idx].to(device)
                batch_y = y_train_tensor[start_idx:end_idx].to(device)
                
                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_loss = total_loss / num_batches
            accuracy = correct / total
            
            train_losses.append(avg_loss)
            train_accs.append(accuracy)
            
            # 验证
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor.to(device))
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_accuracy = (val_predicted == y_val_tensor.to(device)).sum().item() / len(y_val_tensor)
                    val_accs.append(val_accuracy)
                
                print(f"  Epoch [{epoch+1:2d}/{num_epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"Train Acc: {accuracy:.4f} "
                      f"Val Acc: {val_accuracy:.4f}")
            else:
                print(f"  Epoch [{epoch+1:2d}/{num_epochs}] Loss: {avg_loss:.4f} Train Acc: {accuracy:.4f}")
        
        print("\n  [成功] 模型训练完成！")
        
        # 保存模型
        model_path = os.path.join(PROJECT_ROOT, "04_models_ckpt", "cnn_baseline_final.pth")
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
        }, model_path)
        print(f"  模型已保存: {model_path}")
        
        return model, train_losses, train_accs, val_accs
        
    except Exception as e:
        print(f"  [错误] 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def 第6步_评估模型(model):
    """第6步：评估模型性能"""
    print_step(6, "评估模型性能")
    
    if model is None:
        print("  [跳过] 没有训练好的模型")
        return
    
    try:
        import torch
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
        
        # 加载测试数据
        cwt_dir = os.path.join(PROJECT_ROOT, "02_data", "cwt")
        test_file = os.path.join(cwt_dir, "cwt_test.npz")
        
        if not os.path.exists(test_file):
            print(f"  [跳过] 测试数据不存在")
            return
        
        print("  加载测试数据...")
        test_data = np.load(test_file)
        X_test = test_data['X']
        y_test = test_data['y']
        print(f"    测试数据: X={X_test.shape}, y={y_test.shape}")
        
        # 加载模型
        model_path = os.path.join(PROJECT_ROOT, "04_models_ckpt", "cnn_baseline_final.pth")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 评估
        print("\n  开始评估...")
        device = next(model.parameters()).device
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
        
        predictions = predictions.cpu().numpy()
        
        # 计算指标
        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro')
        cm = confusion_matrix(y_test, predictions)
        
        print("\n  " + "=" * 40)
        print("  评估结果")
        print("  " + "=" * 40)
        print(f"  准确率 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  F1分数 (Macro): {f1_macro:.4f}")
        print("\n  混淆矩阵:")
        print(cm)
        
        # 打印每个类别的指标
        print("\n  详细分类报告:")
        class_names = ['正常', '内圈故障', '外圈故障', '滚动体故障']
        report = classification_report(y_test, predictions, target_names=class_names)
        print(report)
        
        # 保存评估结果
        results_file = os.path.join(PROJECT_ROOT, "06_results", "metrics", "evaluation_report.txt")
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("轴承故障诊断模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"F1分数: {f1_macro:.4f}\n\n")
            f.write("混淆矩阵:\n")
            f.write(str(cm) + "\n\n")
            f.write("分类报告:\n")
            f.write(report)
        
        print(f"\n  评估报告已保存: {results_file}")
        
        # 绘制混淆矩阵图
        try:
            from vis_utils import plot_confusion_matrix
            
            fig_dir = os.path.join(PROJECT_ROOT, "06_results", "figures")
            plot_confusion_matrix(cm, os.path.join(fig_dir, "confusion_matrix.png"))
        except Exception as e:
            print(f"  [警告] 绘制混淆矩阵失败: {e}")
        
    except Exception as e:
        print(f"  [错误] 评估模型失败: {e}")
        import traceback
        traceback.print_exc()


def run_pipeline():
    """运行完整流程"""
    print_header("轴承故障诊断 - 完整流程")
    
    print(f"\n项目根目录: {PROJECT_ROOT}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 第0步：创建必要的文件夹
    创建必要的文件夹()
    
    # 第1步：加载原始数据
    signals, labels = 第1步_加载原始数据()
    
    # 第2步：预处理数据
    preprocessed_dir, _ = 第2步_预处理数据()
    
    # 第3步：划分数据集
    split_dir = 第3步_划分数据集()
    
    # 第4步：CWT变换
    cwt_dir = 第4步_CWT变换()
    
    # 第5步：训练模型
    result = 第5步_训练模型()
    if result:
        model, train_losses, train_accs, val_accs = result
    else:
        model = None
    
    # 第6步：评估模型
    第6步_评估模型(model)
    
    # 结束
    total_time = time.time() - start_time
    print_header("流程执行完成")
    print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    print("\n" + "=" * 60)
    print("  结果文件位置:")
    print("  - 模型文件: 04_models_ckpt/")
    print("  - 评估报告: 06_results/metrics/")
    print("  - 图片结果: 06_results/figures/")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
