import numpy as np
import matplotlib.pyplot as plt
import pywt

# === 工具函数 1：自适应软阈值小波去噪 ===
def wavelet_denoise(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_thresh, wavelet)

# === 工具函数 2：串联小波去噪（db4 + sym8） ===
def cascade_wavelet_denoise(signal):
    denoised1 = wavelet_denoise(signal, wavelet='db4', level=3)
    denoised2 = wavelet_denoise(denoised1, wavelet='sym8', level=3)
    return denoised2

# === 工具函数 3：简化 LMS 自适应滤波器 ===
def lms_filter(desired, input_signal, mu=0.01, filter_order=8):
    n = len(input_signal)
    y = np.zeros(n)
    e = np.zeros(n)
    w = np.zeros(filter_order)

    for i in range(filter_order, n):
        x = input_signal[i - filter_order:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w = w + 2 * mu * e[i] * x

    return y, e

# === 示例 ICG 模拟信号 ===
np.random.seed(0)
t = np.linspace(0, 1, 1000)
clean_icg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
baseline_drift = 0.5 * np.sin(2 * np.pi * 0.5 * t)
noise = np.random.normal(0, 0.4, t.shape)
noisy_icg = clean_icg + baseline_drift + noise

# === 应用小波去噪（db4 + sym8） ===
wavelet_denoised = cascade_wavelet_denoise(noisy_icg)

# === 串联 LMS 自适应滤波器（参考信号设为 clean_icg）===
lms_output, lms_error = lms_filter(desired=clean_icg, input_signal=wavelet_denoised, mu=0.01, filter_order=8)

# === 绘图对比 ===
plt.figure(figsize=(14, 6))
plt.plot(t, noisy_icg, label='Noisy ICG', alpha=0.5)
plt.plot(t, wavelet_denoised[:len(t)], label='After Wavelet (db4+sym8)', linewidth=2)
plt.plot(t, lms_output[:len(t)], label='Final Output (Wavelet + LMS)', linewidth=2)
plt.plot(t, clean_icg, '--', label='Clean ICG (Reference)', linewidth=1)
plt.title('ICG Signal Denoising using Wavelet (db4 + sym8) + LMS Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
