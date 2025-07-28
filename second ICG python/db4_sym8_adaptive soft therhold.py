import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec, waverec

# === 自适应软阈值函数 ===
def adaptive_soft_threshold(coeffs, method='mad'):
    detail_coeffs = coeffs[1:]  # 忽略近似系数
    sigma = np.median(np.abs(detail_coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(detail_coeffs[-1])))

    coeffs_thresh = [coeffs[0]]  # 保留近似系数不变
    for c in detail_coeffs:
        coeffs_thresh.append(pywt.threshold(c, value=uthresh, mode='soft'))
    return coeffs_thresh

# === 构造含噪模拟 ICG 信号 ===
np.random.seed(0)
t = np.linspace(0, 1, 1000)
clean_icg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
baseline_drift = 0.3 * np.sin(2 * np.pi * 0.5 * t)
noise = np.random.normal(0, 0.5, t.shape)
noisy_icg = clean_icg + baseline_drift + noise

# === 第一步：db4 小波去除基线漂移 ===
coeffs_db4 = wavedec(noisy_icg, 'db4', level=3)
coeffs_db4_thresh = adaptive_soft_threshold(coeffs_db4)
denoised_db4 = waverec(coeffs_db4_thresh, 'db4')

# === 第二步：sym8 小波去除高频噪声与伪影 ===
coeffs_sym8 = wavedec(denoised_db4, 'sym8', level=2)
coeffs_sym8_thresh = adaptive_soft_threshold(coeffs_sym8)
final_denoised = waverec(coeffs_sym8_thresh, 'sym8')

# === 绘图比较 ===
plt.figure(figsize=(14, 6))
plt.plot(t, noisy_icg, label='Noisy ICG (with baseline & noise)', alpha=0.5)
plt.plot(t, final_denoised[:len(t)], label='Denoised ICG (db4 + sym8)', linewidth=2)
plt.plot(t, clean_icg, label='Clean ICG (Reference)', linestyle='--', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('ICG Denoising using Cascaded db4 + sym8 Wavelet with Adaptive Soft Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
