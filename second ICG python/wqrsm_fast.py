import numpy as np

def wqrsm_fast(data, Fs=125, PWfreq=60, TmDEF=100, jflag=0):
    BUFLN = 16384
    EYE_CLS = 0.25
    MaxQRSw = 0.13
    NDP = 2.5
    WFDB_DEFGAIN = 200.0

    # === Gain scaling ===
    datatest = data[:(len(data) // Fs) * Fs]
    if len(datatest) > Fs:
        datatest = datatest.reshape((-1, Fs))
    # 兼容 1D 和 2D 数据
    if datatest.ndim == 1:
        test_ap = np.max(datatest) - np.min(datatest)
    else:
        test_ap = np.median(np.max(datatest, axis=1) - np.min(datatest, axis=1))

    if test_ap < 10:
        data = data * WFDB_DEFGAIN

    # === 初始化 swqrsm 状态 ===
    lfsc = int(1.25 * WFDB_DEFGAIN**2 / Fs)
    LPn = min(int(Fs / PWfreq), 8)
    LP2n = 2 * LPn
    EyeClosing = int(Fs * EYE_CLS)
    ExpectPeriod = int(Fs * NDP)
    LTwindow = int(Fs * MaxQRSw)
    Tm = int(TmDEF / 5.0)

    swqrsm = {
        'data': data,
        'lfsc': lfsc,
        'LPn': LPn,
        'LP2n': LP2n,
        'LTwindow': LTwindow,
        'BUFLN': BUFLN,
        'lbuf': np.zeros(BUFLN),
        'ebuf': np.full(BUFLN, np.sqrt(lfsc), dtype=int),
        'lt_tt': 0,
        'aet': 0,
        'Yn': 0,
        'Yn1': 0,
        'Yn2': 0
    }

    def ltsamp(t, swqrsm):
        while t > swqrsm['lt_tt']:
            swqrsm['Yn2'] = swqrsm['Yn1']
            swqrsm['Yn1'] = swqrsm['Yn']
            tt = swqrsm['lt_tt']
            v0 = swqrsm['data'][tt] if 0 < tt < len(swqrsm['data']) else swqrsm['data'][0]
            v1 = swqrsm['data'][tt - swqrsm['LPn']] if 0 < tt - swqrsm['LPn'] < len(swqrsm['data']) else swqrsm['data'][0]
            v2 = swqrsm['data'][tt - swqrsm['LP2n']] if 0 < tt - swqrsm['LP2n'] < len(swqrsm['data']) else swqrsm['data'][0]

            swqrsm['Yn'] = 2 * swqrsm['Yn1'] - swqrsm['Yn2'] + v0 - 2 * v1 + v2
            dy = int((swqrsm['Yn'] - swqrsm['Yn1']) / swqrsm['LP2n'])
            swqrsm['lt_tt'] += 1
            et = int(np.sqrt(swqrsm['lfsc'] + dy * dy))
            id = swqrsm['lt_tt'] % swqrsm['BUFLN']
            swqrsm['ebuf'][id] = et
            id2 = (swqrsm['lt_tt'] - swqrsm['LTwindow']) % swqrsm['BUFLN']
            swqrsm['aet'] += et - swqrsm['ebuf'][id2]
            swqrsm['lbuf'][id] = swqrsm['aet']

        id3 = t % swqrsm['BUFLN']
        swqrsm['lt_data'] = swqrsm['lbuf'][id3]
        return swqrsm

    # === 初始化滤波器 ===
    t1 = min(Fs * 8, int(BUFLN * 0.5))
    T0 = 0
    for t in range(1, t1 + 1):
        swqrsm = ltsamp(t, swqrsm)
        T0 += swqrsm['lt_data']
    T0 /= t1
    Ta = 3 * T0

    qrs = []
    jpoints = []
    learning = True
    t = 1
    T1 = 2 * T0
    timer_d = 0

    # === 主循环 ===
    while t < len(data):
        if learning and t > t1:
            learning = False
            T1 = T0
            t = 1
            continue

        swqrsm = ltsamp(t, swqrsm)
        if swqrsm['lt_data'] > T1:
            timer_d = 0
            maxd = swqrsm['lt_data']
            mind = maxd
            for tt in range(t + 1, t + EyeClosing // 2):
                swqrsm = ltsamp(tt, swqrsm)
                maxd = max(maxd, swqrsm['lt_data'])
            for tt in range(t - 1, t - EyeClosing // 2, -1):
                swqrsm = ltsamp(tt, swqrsm)
                mind = min(mind, swqrsm['lt_data'])

            if maxd > mind + 10:
                onset = int(maxd / 100) + 2
                tpq = t - 5
                for tt in range(t, t - EyeClosing // 2, -1):
                    d = []
                    for i in range(5):
                        swqrsm = ltsamp(tt - i, swqrsm)
                        d.append(swqrsm['lt_data'])
                    if all(d[i] - d[i + 1] < onset for i in range(4)):
                        tpq = tt - swqrsm['LP2n']
                        break

                if not learning and tpq < len(data):
                    qrs.append(tpq)
                    if jflag:
                        tj = t + 5
                        for tt in range(t, t + EyeClosing // 2):
                            swqrsm = ltsamp(tt, swqrsm)
                            if swqrsm['lt_data'] > maxd - int(maxd / 10):
                                tj = tt
                                break
                        if tj < len(data):
                            jpoints.append(tj)

                Ta += (maxd - Ta) / 10
                T1 = Ta / 3
                t += EyeClosing
        elif not learning:
            timer_d += 1
            if timer_d > ExpectPeriod and Ta > Tm:
                Ta -= 1
                T1 = Ta / 3
        t += 1

    return qrs, jpoints if jflag else []
