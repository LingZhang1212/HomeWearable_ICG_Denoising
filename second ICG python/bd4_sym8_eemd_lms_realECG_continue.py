import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import pywt
from PyEMD import EEMD
import pandas as pd
import os

from InitializeHRVparams import InitializeHRVparams
from ConvertRawDataToRRIntervals import ConvertRawDataToRRIntervals

# ==== 读取 Excel ECG/ICG 数据 ====
def load_ecg_icg_from_excel(filepath):
    df = pd.read_excel(filepath, header=None)
    ecg = df.iloc[:, 0].values.astype(float)
    icg = df.iloc[:, 1].values.astype(float)
    return ecg, icg

# ==== 自适应软阈值小波去噪 ====
def adaptive_soft_threshold(coeffs, sigma=None):
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[-1])))
    return [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]

def wavelet_denoise(signal, wavelet_name='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_thresh = adaptive_soft_threshold(coeffs)
    return pywt.waverec(coeffs_thresh, wavelet=wavelet_name)

# ==== EEMD 去噪 ====
def eemd_denoise(signal, max_imfs=10):
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    return np.sum(imfs[1:min(max_imfs, len(imfs))], axis=0)

# ==== LMS 滤波 ====
def lms_filter(signal, desired, mu=0.01, order=5):
    N = len(signal)
    w = np.zeros(order)
    y = np.zeros(N)
    for n in range(order, N):
        x = signal[n - order:n][::-1]
        y[n] = np.dot(w, x)
        e = desired[n] - y[n]
        w += 2 * mu * e * x
    return y

# ==== 主处理流程 ====
def process_with_ecg_toolbox(ecg, clean_icg, fs=1000):
    HRVparams = InitializeHRVparams('Excel_ECG_ICG')
    HRVparams['Fs'] = fs

    # ICG 带通滤波
    b, a = butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered_icg = filtfilt(b, a, clean_icg)

    # R 波检测
    print("Running ConvertRawDataToRRIntervals to get R peaks...")
    _, rr, R_pk, SQIvalue, SQIidx = ConvertRawDataToRRIntervals(ecg, HRVparams, subjectID="real_data")

    # 计算心搏长度
    RR_intervals = np.diff(R_pk)
    median_RR = int(np.ceil(np.median(RR_intervals)))
    llim_beat = int(0.15 * fs)
    ulim_beat = median_RR - llim_beat
    beat_len = llim_beat + ulim_beat

    beat_segments_clean = []
    beat_segments_denoised = []

    for r in R_pk:
        start = r - llim_beat
        end = r + ulim_beat
        if start < 0 or end > len(clean_icg):
            continue

        icg_seg = filtered_icg[start:end]

        # 多段串联去噪
        db4_out = wavelet_denoise(icg_seg, wavelet_name='db4')
        sym8_out = wavelet_denoise(db4_out, wavelet_name='sym8')
        eemd_out = eemd_denoise(sym8_out)
        lms_out = lms_filter(eemd_out, icg_seg)

        beat_segments_clean.append(icg_seg)
        beat_segments_denoised.append(lms_out)

    # ==== 构造完整时序波形 ====
    denoised_icg_full = np.zeros_like(clean_icg)
    counts = np.zeros_like(clean_icg)
    valid_R_peaks = [r for r in R_pk if r - llim_beat >= 0 and r + ulim_beat <= len(clean_icg)]

    for idx, r in enumerate(valid_R_peaks):
        start = r - llim_beat
        end = r + ulim_beat
        denoised_icg_full[start:end] += beat_segments_denoised[idx]
        counts[start:end] += 1

    counts[counts == 0] = 1
    denoised_icg_full /= counts

    return np.array(beat_segments_clean), np.array(beat_segments_denoised), beat_len, filtered_icg, denoised_icg_full

# ==== 主程序入口 ====
if __name__ == "__main__":
    filepath = r"C:\Users\LingZhang\Desktop\ECG ICG\ECG_ICG\second ICG python\RawData_Subject_1_task_BL_converted.xlsx"
    output_dir = r"C:\Users\LingZhang\Desktop\ECG ICG\ECG_ICG\second ICG python"
    os.makedirs(output_dir, exist_ok=True)

    ecg, icg = load_ecg_icg_from_excel(filepath)
    beats_clean, beats_denoised, beat_len, filtered_icg, denoised_icg_full = process_with_ecg_toolbox(ecg, icg, fs=1000)

    avg_clean = np.mean(beats_clean, axis=0)
    avg_denoised = np.mean(beats_denoised, axis=0)

    # ==== 单搏平均对比图 ====
    plt.figure(figsize=(14, 6))
    plt.plot(avg_clean, label="Avg Filtered ICG", linestyle='--')
    plt.plot(avg_denoised, label="Avg Denoised ICG", linewidth=2)
    plt.title("Denoising with Real ECG R-Peaks (db4 + sym8 + EEMD + LMS)")
    plt.xlabel(f"Sample (window size = {beat_len})")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path1 = os.path.join(output_dir, "avg_denoised_icg.png")
    plt.savefig(save_path1, dpi=300)
    print(f"Saved: {save_path1}")
    plt.show()

    # ==== 连续信号对比图 ====
    plt.figure(figsize=(16, 6))
    plt.plot(icg, label='Raw ICG', alpha=0.4)
    plt.plot(filtered_icg, label='Filtered ICG (bandpass)', alpha=0.6)
    plt.plot(denoised_icg_full, label='Denoised ICG (full signal)', linewidth=1.5)
    plt.title("Full ICG Signal Before and After Denoising")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "full_denoised_icg.png")
    plt.savefig(save_path2, dpi=300)
    print(f"Saved: {save_path2}")
    plt.show()
