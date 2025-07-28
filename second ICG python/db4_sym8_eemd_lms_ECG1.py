import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import pywt
from PyEMD import EEMD

from InitializeHRVparams import InitializeHRVparams
from ConvertRawDataToRRIntervals import ConvertRawDataToRRIntervals

# ==== 读取真实 CSV ECG/ICG 数据 ====
def load_real_ecg_icg_from_csv(filepath, ecg_col=0, icg_col=1):
    data = pd.read_csv(filepath)
    ecg = data.iloc[:, ecg_col].values
    icg = data.iloc[:, icg_col].values
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

def eemd_denoise(signal, max_imfs=10):
    eemd = EEMD()
    imfs = eemd.eemd(signal)
    return np.sum(imfs[1:min(max_imfs, len(imfs))], axis=0)

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

# ==== 主流程 ====
def process_with_ecg_toolbox(ecg, clean_icg, fs=1000):
    HRVparams = InitializeHRVparams('Real_ECG_ICG')
    HRVparams['Fs'] = fs

    # ICG 带通滤波
    b, a = butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered_icg = filtfilt(b, a, clean_icg)

    # R 波检测
    print("Running ConvertRawDataToRRIntervals to get R peaks...")
    _, rr, R_pk, SQIvalue, SQIidx = ConvertRawDataToRRIntervals(ecg, HRVparams, subjectID="real_data")

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

        # 去噪流程
        db4_out = wavelet_denoise(icg_seg, wavelet_name='db4')
        sym8_out = wavelet_denoise(db4_out, wavelet_name='sym8')
        eemd_out = eemd_denoise(sym8_out)
        lms_out = lms_filter(eemd_out, icg_seg)

        beat_segments_clean.append(icg_seg)
        beat_segments_denoised.append(lms_out)

    return np.array(beat_segments_clean), np.array(beat_segments_denoised), beat_len

# ==== 执行 ====
if __name__ == "__main__":
    filepath = r"C:\Users\LingZhang\Desktop\ECG ICG\ECG_ICG\second ICG python\Ecg_ICG.csv"
    ecg, icg = load_real_ecg_icg_from_csv(filepath)

    beats_clean, beats_denoised, beat_len = process_with_ecg_toolbox(ecg, icg, fs=1000)

    avg_clean = np.mean(beats_clean, axis=0)
    avg_denoised = np.mean(beats_denoised, axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(avg_clean, label="Avg Filtered ICG", linestyle='--')
    plt.plot(avg_denoised, label="Avg Denoised ICG", linewidth=2)
    plt.title("Denoising with Real ECG R-Peaks (db4 + sym8 + EEMD + LMS)")
    plt.xlabel(f"Sample (window size = {beat_len})")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
