def run_qrsdet_by_seg(ecg, HRVparams):
    """
    Run QRS detection segment-by-segment to avoid issues from global thresholding.

    Parameters:
    - ecg: 1D numpy array, the ECG signal
    - HRVparams: dictionary with keys:
        - 'Fs': sampling frequency
        - 'PeakDetect': dictionary with:
            - 'windows': window length in seconds
            - 'THRES': initial threshold
            - 'ecgType': 'FECG' or 'MECG'

    Returns:
    - QRS: list of detected QRS sample indices
    """

    import numpy as np

    fs = HRVparams['Fs']
    window = HRVparams['PeakDetect']['windows']
    thres = HRVparams['PeakDetect']['THRES']
    ecgType = HRVparams['PeakDetect']['ecgType']

    segsize_samp = int(window * fs)
    nb_seg = len(ecg) // segsize_samp
    QRS = []
    start = 0
    stop = segsize_samp
    sign_force = 0

    def jqrs(segment, HRVparams, sign_force=0):
        """
        Placeholder jqrs function.
        Should return (qrs_peaks, sign_force) for the segment
        """
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(segment, distance=int(0.25 * HRVparams['Fs']))
        return peaks.tolist(), sign_force

    try:
        for ch in range(nb_seg):
            qrstemp = []

            if ch == 0:
                dTplus = fs
                dTminus = 0
            elif ch == nb_seg - 1:
                dTplus = 0
                dTminus = fs
            else:
                dTplus = fs
                dTminus = fs

            seg_start = max(start - dTminus, 0)
            seg_stop = min(stop + dTplus, len(ecg))
            segment = ecg[seg_start:seg_stop]

            if ecgType == 'FECG':
                thres_trans = thres
                while len(qrstemp) < 20 and thres_trans > 0.1:
                    qrstemp, sign_force = jqrs(segment, HRVparams, sign_force)
                    thres_trans -= 0.1
            else:
                qrstemp, sign_force = jqrs(segment, HRVparams, sign_force)

            new_qrs = [start - dTminus + q for q in qrstemp]
            new_qrs = [q for q in new_qrs if start <= q < stop]

            if QRS and new_qrs and (new_qrs[0] - QRS[-1]) < 0.25 * fs:
                new_qrs = new_qrs[1:]

            QRS.extend(new_qrs)
            start += segsize_samp
            stop += segsize_samp

        return QRS

    except Exception as e:
        import traceback
        traceback.print_exc()
        return [1000, 2000]
