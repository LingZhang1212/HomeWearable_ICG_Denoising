# HomeWearable_ICG_Denoising
Signal processing pipeline for denoising ICG and ECG signals in home and wearable environments. Combines wavelet transforms (db4/sym8), EEMD, and LMS filtering. ECG R-peaks guide ICG segmentation for accurate SV/CO estimation outside clinical settings.
# HomeWearable_ICG_Denoising

A signal processing project focused on improving the quality of impedance cardiography (ICG) signals in home and wearable environments.

## 🔍 Overview

This repository implements a multi-stage denoising pipeline combining:

- Discrete wavelet transforms (db4 and sym8)
- Ensemble empirical mode decomposition (EEMD)
- Least-mean-square (LMS) adaptive filtering

ECG signals are used for robust R-peak detection, providing temporal reference points for accurate ICG beat segmentation and feature extraction.

## 🎯 Goal

To enable reliable, real-time, and high-fidelity cardiac monitoring outside clinical settings, supporting stroke volume (SV) and cardiac output (CO) estimation for wearable and home healthcare applications.

## 📁 Structure

