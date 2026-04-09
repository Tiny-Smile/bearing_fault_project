import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 参数设置
fs = 1000          # 采样率 (Hz)
duration = 0.2     # 信号时长 (秒)，选择0.2秒以清晰显示波形
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# 生成三个频率的正弦波
f1, f2, f3 = 70, 80, 90
# signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)
signal = np.sin(2 * np.pi * f1 * t)+ np.sin(2 * np.pi * f2 * t)+ np.sin(2 * np.pi * f3 * t)+ np.sin(2 * np.pi * 200 * t)

# 计算FFT
N = len(signal)
yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]        # 频率轴（正频率部分）
magnitude = 2.0/N * np.abs(yf[:N//2]) # 幅值谱（单边）

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 时域图
ax1.plot(t, signal, color='b', linewidth=1)
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('幅值')
ax1.set_title('70Hz + 80Hz + 90Hz 合成信号的时域波形')
ax1.grid(True)
ax1.set_xlim(0, duration)   # 显示完整时域范围

# 频谱图
ax2.stem(xf, magnitude, basefmt=" ", linefmt='r-', markerfmt='ro')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('幅值')
ax2.set_title('合成信号的频谱图')
ax2.set_xlim(0, 150)        # 只显示到150Hz，突出三个频率成分
ax2.grid(True)

plt.tight_layout()
plt.show()