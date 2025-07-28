import pywt
import numpy as np
import matplotlib.pyplot as plt

# === 示例数据加载 ===
# 这里只模拟一段ICG信号（真实应用中建议使用你的ICG信号导入）
fs = 1000  # 采样率
np.random.seed(0)
t = np.linspace(0, 1, fs)
pure_icg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t) # 模拟ICG主波
noise = np.random.normal(0, 0.5, t.shape)  # 高斯白噪声
baseline = 0.3 * np.sin(2 * np.pi * 0.3 * t)  # 模拟基线漂移
noisy_icg = pure_icg + noise + baseline  # 叠加后信号

# === 小波去噪函数 ===
def wavelet_denoise(signal, wavelet='db4', level=3, threshold_type='soft'):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # robust noise estimate
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

    denoised_coeffs = [coeffs[0]]  # 保留近似分量（低频）
    for i in range(1, len(coeffs)):
        denoised = pywt.threshold(coeffs[i], value=uthresh, mode=threshold_type)
        denoised_coeffs.append(denoised)

    return pywt.waverec(denoised_coeffs, wavelet)

# === 去噪处理 ===
denoised_icg = wavelet_denoise(noisy_icg, wavelet='db4', level=3, threshold_type='soft')

# === 可视化 ===
plt.figure(figsize=(12, 6))
plt.plot(t, noisy_icg, label='Noisy ICG', alpha=0.5)
plt.plot(t, denoised_icg, label='Denoised ICG', linewidth=2)
plt.plot(t, pure_icg, label='Original ICG (simulated)', linestyle='--')
plt.title('Wavelet Denoising for Simulated ICG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
