import numpy as np
from scipy.signal import filtfilt, medfilt, resample, find_peaks
import matplotlib.pyplot as plt

def jqrs(ecg, HRVparams):
    fs = HRVparams['Fs']
    REF_PERIOD = HRVparams['PeakDetect']['REF_PERIOD']
    THRES = HRVparams['PeakDetect']['THRES']
    fid_vec = HRVparams['PeakDetect'].get('fid_vec', None)
    SIGN_FORCE = HRVparams['PeakDetect'].get('SIGN_FORCE', None)
    debug = HRVparams['PeakDetect'].get('debug', False)

    ecg = np.asarray(ecg).flatten()
    NB_SAMP = len(ecg)
    tm = np.arange(1, NB_SAMP + 1) / fs

    MED_SMOOTH_NB_COEFF = round(fs / 100)
    INT_NB_COEFF = round(7 * fs / 256)
    SEARCH_BACK = True
    MAX_FORCE = None
    MIN_AMP = 0.1

    try:
        b1 = np.array([
            -7.757327341237223e-05, -2.357742589814283e-04, -6.689305101192819e-04, -0.001770119249103,
            -0.004364327211358, -0.010013251577232, -0.021344241245400, -0.042182820580118, -0.077080889653194,
            -0.129740392318591, -0.200064921294891, -0.280328573340852, -0.352139052257134, -0.386867664739069,
            -0.351974030208595, -0.223363323458050, 0, 0.286427448595213, 0.574058766243311,
            0.788100265785590, 0.867325070584078, 0.788100265785590, 0.574058766243311, 0.286427448595213, 0,
            -0.223363323458050, -0.351974030208595, -0.386867664739069, -0.352139052257134,
            -0.280328573340852, -0.200064921294891, -0.129740392318591, -0.077080889653194, -0.042182820580118,
            -0.021344241245400, -0.010013251577232, -0.004364327211358, -0.001770119249103, -6.689305101192819e-04,
            -2.357742589814283e-04, -7.757327341237223e-05
        ])

        b1 = resample(b1, int(len(b1) * fs / 250))
        bpfecg = filtfilt(b1, [1], ecg)

        if np.mean(np.abs(bpfecg) > MIN_AMP) > 0.20:
            dffecg = np.diff(bpfecg)
            sqrecg = dffecg ** 2
            intecg = np.convolve(sqrecg, np.ones(INT_NB_COEFF), mode='same')
            mdfint = medfilt(intecg, MED_SMOOTH_NB_COEFF)
            delay = int(np.ceil(INT_NB_COEFF / 2))
            mdfint = np.roll(mdfint, -delay)

            if fid_vec is not None:
                mdfintFidel = np.copy(mdfint)
                mdfintFidel[np.array(fid_vec) > 2] = 0
            else:
                mdfintFidel = mdfint

            if NB_SAMP / fs > 90:
                xs = np.sort(mdfintFidel[fs:int(fs * 90)])
            else:
                xs = np.sort(mdfintFidel[fs:])

            if MAX_FORCE is None:
                ind_xs = int(np.ceil(0.98 * len(xs))) if NB_SAMP / fs > 10 else int(np.ceil(0.99 * len(xs)))
                en_thres = xs[ind_xs]
            else:
                en_thres = MAX_FORCE

            poss_reg = mdfint > (THRES * en_thres)
            if not np.any(poss_reg):
                poss_reg[10] = True

            if SEARCH_BACK:
                indAboveThreshold = np.where(poss_reg)[0]
                RRv = np.diff(tm[indAboveThreshold])
                medRRv = np.median(RRv[RRv > 0.01])
                indMissed = np.where(RRv > 1.5 * medRRv)[0]

                for i in indMissed:
                    poss_reg[indAboveThreshold[i]:indAboveThreshold[i + 1]] = (
                        mdfint[indAboveThreshold[i]:indAboveThreshold[i + 1]] > (0.5 * THRES * en_thres)
                    )

            left = np.where(np.diff(np.concatenate(([0], poss_reg.astype(int)))) == 1)[0]
            right = np.where(np.diff(np.concatenate((poss_reg.astype(int), [0]))) == -1)[0]

            if SIGN_FORCE is not None:
                sign = SIGN_FORCE
            else:
                loc = [np.argmax(np.abs(bpfecg[l:r])) + l for l, r in zip(left, right)]
                sign = np.mean(ecg[loc])

            maxloc, maxval = [], []
            for l, r in zip(left, right):
                segment = ecg[l:r]
                if sign > 0:
                    idx = np.argmax(segment)
                else:
                    idx = np.argmin(segment)
                loc = l + idx
                if maxloc and (loc - maxloc[-1]) < fs * REF_PERIOD:
                    if abs(segment[idx]) < abs(maxval[-1]):
                        continue
                    else:
                        maxloc.pop()
                        maxval.pop()
                maxloc.append(loc)
                maxval.append(segment[idx])

            qrs_pos = np.array(maxloc)
            R_t = tm[qrs_pos]
            R_amp = np.array(maxval)
            hrv = 60 / np.diff(R_t)
        else:
            qrs_pos, sign, en_thres = [], [], []
    except Exception as e:
        print("Error:", str(e))
        qrs_pos, sign, en_thres = [1, 10, 20], 1, 0.5

    return qrs_pos, sign, en_thres
