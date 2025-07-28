import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec, waverec, threshold

# 生成示例 ICG 信号（含噪）
np.random.seed(0)
t = np.linspace(0, 1, 1000)
clean_icg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
noise = np.random.normal(0, 0.5, t.shape)
# 添加模拟基线漂移
baseline_wander = 0.3 * np.sin(2 * np.pi * 0.3 * t)
noisy_icg = clean_icg + baseline_wander + noise


# 小波分解
wavelet = 'db4'
level = 3
coeffs = wavedec(noisy_icg, wavelet, level=level)

# 使用 MAD 估计噪声标准差（使用最后一个细节系数）
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
uthresh = sigma * np.sqrt(2 * np.log(len(noisy_icg)))

# 对每个细节系数应用软阈值
coeffs_thresh = [coeffs[0]] + [threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]

# 重构信号
denoised_icg = waverec(coeffs_thresh, wavelet)

# 绘图对比
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_icg, label='Noisy ICG', alpha=0.6)
plt.plot(t, denoised_icg[:len(t)], label='Denoised ICG (Wavelet + Adaptive Soft Threshold)', linewidth=2)
plt.plot(t, clean_icg, label='Clean ICG (Reference)', linestyle='--', linewidth=1)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('ICG Signal Denoising using Wavelet + Adaptive Soft Thresholding')
plt.grid(True)
plt.tight_layout()
plt.show()
