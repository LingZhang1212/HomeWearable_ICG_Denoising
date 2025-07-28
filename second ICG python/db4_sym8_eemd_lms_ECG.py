# Final integrated version: ECG-assisted beat segmentation using your provided HRV modules

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import butter, filtfilt
from InitializeHRVparams import InitializeHRVparams
from ConvertRawDataToRRIntervals import ConvertRawDataToRRIntervals

# ==== Parameters ====
fs = 1000
lag_thres = 50
window_size_sec = 60

# ==== Load simulated synchronized ECG + ICG data ====
def generate_simulated_ecg_icg(seed=0):
    np.random.seed(seed)
    t = np.linspace(0, 10, 10000)
    ecg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(len(t))
    clean_icg = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    baseline = 0.3 * np.sin(2 * np.pi * 0.3 * t)
    noise = np.random.normal(0, 0.5, t.shape)
    noisy_icg = clean_icg + baseline + noise
    return ecg, clean_icg, noisy_icg

# ==== Denoising steps ====
def adaptive_soft_threshold(coeffs, sigma=None):
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(coeffs[-1])))
    return [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]

def wavelet_denoise(signal, wavelet_name='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet=wavelet_name, level=level)
    coeffs_thresh = adaptive_soft_threshold(coeffs)
    return pywt.waverec(coeffs_thresh, wavelet=wavelet_name)

from PyEMD import EEMD
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

# ==== Main workflow using your ECG module ====
def process_with_ecg_toolbox(ecg, clean_icg, noisy_icg):
    # === 1. Initialize ECG HRV detection params
    HRVparams = InitializeHRVparams('Noise_removal_ICG')
    HRVparams['Fs'] = fs

    # === 2. Bandpass filter ICG
    b, a = butter(4, [0.5 / (fs / 2), 40 / (fs / 2)], btype='band')
    filtered_icg = filtfilt(b, a, noisy_icg)

    # === 3. R peak detection using your own toolbox
    print("Running ConvertRawDataToRRIntervals to get R peaks...")
    _, rr, R_pk, SQIvalue, SQIidx = ConvertRawDataToRRIntervals(ecg, HRVparams, "Simulated")

    # === 4. Compute median RR interval
    RR_intervals = np.diff(R_pk)
    median_RR = int(np.ceil(np.median(RR_intervals)))

    # === 5. Extract beat segments centered around R peaks
    llim_beat = int(0.15 * fs)
    ulim_beat = median_RR - llim_beat
    beat_len = llim_beat + ulim_beat

    beat_segments_clean = []
    beat_segments_denoised = []

    for r in R_pk:
        start = r - llim_beat
        end = r + ulim_beat
        if start < 0 or end > len(ecg):
            continue

        clean_icg_seg = clean_icg[start:end]
        noisy_icg_seg = filtered_icg[start:end]

        # Denoising
        db4_out = wavelet_denoise(noisy_icg_seg, wavelet_name='db4')
        sym8_out = wavelet_denoise(db4_out, wavelet_name='sym8')
        eemd_out = eemd_denoise(sym8_out)
        lms_out = lms_filter(eemd_out, clean_icg_seg)  # clean_icg used as desired

        beat_segments_clean.append(clean_icg_seg)
        beat_segments_denoised.append(lms_out)

    return np.array(beat_segments_clean), np.array(beat_segments_denoised), beat_len

# ==== Run and Plot ====
import pywt

ecg, clean_icg, noisy_icg = generate_simulated_ecg_icg()
beats_clean, beats_denoised, beat_len = process_with_ecg_toolbox(ecg, clean_icg, noisy_icg)

avg_clean = np.mean(beats_clean, axis=0)
avg_denoised = np.mean(beats_denoised, axis=0)

plt.figure(figsize=(14, 6))
plt.plot(avg_clean, label="Avg Clean ICG", linestyle='--')
plt.plot(avg_denoised, label="Avg Denoised ICG", linewidth=2)
plt.title("Denoising with Your ECG Toolbox R-Peaks (db4 + sym8 + EEMD + LMS)")
plt.xlabel(f"Sample (window size = {beat_len})")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
