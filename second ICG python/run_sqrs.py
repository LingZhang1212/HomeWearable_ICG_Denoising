import numpy as np
from scipy.signal import resample


def run_sqrs(ecg, HRVparams, rs=1):
    """
    Python implementation of SQRS QRS detector
    """
    if ecg is None or HRVparams is None:
        raise ValueError("Must provide ECG signal and HRVparams")

    debug = 0
    fs = HRVparams['Fs']
    out = []

    if rs == 0:
        freq = fs
        ecg_data = ecg
    else:
        freq = 256
        ecg_data = resample(ecg, int(len(ecg) * freq / fs))

    ms160 = int(np.ceil(0.16 * freq))
    ms200 = int(np.ceil(0.2 * freq))
    s2 = int(np.ceil(2 * freq))
    scmin = 500
    scmax = 10 * scmin
    slopecrit = 10 * scmin
    maxslope = 0
    nslope = 0

    time = 0
    now = 10
    maxtime = 0
    sign = 0
    qtime = 0

    while now < len(ecg_data):
        filt = np.dot([1, 4, 6, 4, 1, -1, -4, -6, -4, -1], ecg_data[now - 9:now + 1])

        if time % s2 == 0:
            if nslope == 0:
                slopecrit = max(slopecrit - slopecrit / 16, scmin)
            elif nslope >= 5:
                slopecrit = min(slopecrit + slopecrit / 16, scmax)

        if nslope == 0 and abs(filt) > slopecrit:
            nslope += 1
            maxtime = ms160
            sign = 1 if filt > 0 else -1
            qtime = time

        if nslope != 0:
            if filt * sign < -slopecrit:
                sign = -sign
                nslope += 1
                maxtime = ms200 if nslope > 4 else ms160
            elif filt * sign > slopecrit and abs(filt) > maxslope:
                maxslope = abs(filt)

            if maxtime < 0:
                if 2 <= nslope <= 4:
                    slopecrit += ((maxslope / 4) - slopecrit) / 8
                    slopecrit = max(min(slopecrit, scmax), scmin)
                    out.append(now - (time - qtime) - 4)
                    time = 0
                elif nslope >= 5:
                    out.append(now - (time - qtime) - 4)
                nslope = 0
            maxtime -= 1

        time += 1
        now += 1

    out = [x - 1 for x in out]  # Adjust for 1-sample offset

    if debug > 0:
        import matplotlib.pyplot as plt
        plt.plot(ecg_data, 'b')
        plt.plot(out, ecg_data[out], 'm*')
        plt.title("QRS Detections (SQRS)")
        plt.show()

    return np.array(out)

