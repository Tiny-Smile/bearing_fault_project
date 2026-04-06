# 环境验证代码
import torch
import pywt
import sklearn

print("Python环境：", torch.__version__)
print("GPU可用：", torch.cuda.is_available())
print("PyWavelets版本：", pywt.__version__)
print("Scikit-learn版本：", sklearn.__version__)

# 路径验证
import os
print("数据集目录存在：", os.path.exists("./02_data/raw"))
print("工具目录存在：", os.path.exists("./03_code/01_utils"))
