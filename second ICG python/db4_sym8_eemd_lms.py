import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec, waverec
from PyEMD import EEMD
from scipy.signal import lfilter


def generate_noisy_icg_signal(seed=0):
    np.random.seed(seed)
    t = np.linspace(0, 1, 1000)
    clean = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    baseline = 0.3 * np.sin(2 * np.pi * 0.5 * t)  # 模拟基线漂移
    noise = np.random.normal(0, 0.5, t.shape)
    noisy = clean + baseline + noise
    return t, clean, noisy


def adaptive_soft_threshold(coeffs, sigma=None):
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[-1])))
    return [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]


def wavelet_denoise(signal, wavelet_name='db4', level=3):
    coeffs = wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_thresh = adaptive_soft_threshold(coeffs)
    return waverec(coeffs_thresh, wavelet=wavelet_name)


def eemd_denoise(signal, max_imfs=10):
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    # 丢弃第一个 IMF（通常为高频噪声）
    denoised = np.sum(imfs[1:min(max_imfs, len(imfs))], axis=0)
    return denoised


def lms_filter(signal, desired, mu=0.01, order=5):
    N = len(signal)
    w = np.zeros(order)
    y = np.zeros(N)
    for n in range(order, N):
        x = signal[n-order:n][::-1]
        y[n] = np.dot(w, x)
        e = desired[n] - y[n]
        w += 2 * mu * e * x
    return y


# ==== 数据生成 ====
t, clean_icg, noisy_icg = generate_noisy_icg_signal()

# ==== 多级去噪 ====
# 1. db4
denoised_db4 = wavelet_denoise(noisy_icg, wavelet_name='db4')

# 2. sym8
denoised_sym8 = wavelet_denoise(denoised_db4, wavelet_name='sym8')

# 3. EEMD
denoised_eemd = eemd_denoise(denoised_sym8)

# 4. LMS
denoised_final = lms_filter(denoised_eemd, clean_icg)

# ==== 绘图 ====
plt.figure(figsize=(14, 6))
plt.plot(t, noisy_icg, label='Noisy ICG', alpha=0.4)
plt.plot(t, clean_icg, label='Clean ICG (Reference)', linestyle='--', linewidth=1)
plt.plot(t, denoised_final, label='Denoised ICG (db4 + sym8 + EEMD + LMS)', linewidth=2)
plt.legend()
plt.title("ICG Signal Denoising using db4 + sym8 + EEMD + LMS")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
