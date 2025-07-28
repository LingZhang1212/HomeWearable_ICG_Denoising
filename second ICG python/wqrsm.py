import numpy as np


class WQRSDetector:
    def __init__(self, fs=125, PWfreq=60, TmDEF=100, jflag=0):
        self.fs = fs
        self.PWfreq = PWfreq
        self.TmDEF = TmDEF
        self.jflag = jflag

        self.EYE_CLS = 0.25
        self.MaxQRSw = 0.13
        self.NDP = 2.5
        self.WFDB_DEFGAIN = 200.0

        self.BUFLN = 16384
        self.Yn = 0
        self.Yn1 = 0
        self.Yn2 = 0
        self.lt_tt = 0
        self.lbuf = np.zeros(self.BUFLN)
        self.ebuf = np.full(self.BUFLN, int(np.sqrt(1)), dtype=int)
        self.aet = 0
        self.et = 0

    def initialize(self, data):
        self.data = np.array(data)
        self.gain = self.WFDB_DEFGAIN
        self.lfsc = int(1.25 * self.gain * self.gain / self.fs)
        self.LPn = min(int(self.fs / self.PWfreq), 8)
        self.LP2n = 2 * self.LPn
        self.EyeClosing = int(self.fs * self.EYE_CLS)
        self.ExpectPeriod = int(self.fs * self.NDP)
        self.LTwindow = int(self.fs * self.MaxQRSw)

        datatest = self.data[: (len(self.data) // self.fs) * self.fs]
        if len(datatest) > self.fs:
            datatest = datatest.reshape((-1, self.fs))
        test_ap = np.median(np.max(datatest, axis=1) - np.min(datatest, axis=1))
        if test_ap < 10:
            self.data *= self.gain

        self.lbuf = np.zeros(self.BUFLN)
        self.ebuf = np.full(self.BUFLN, int(np.sqrt(self.lfsc)), dtype=int)
        self.lt_tt = 0
        self.aet = 0
        self.Yn = 0
        self.Yn1 = 0
        self.Yn2 = 0

    def ltsamp(self, t):
        while t > self.lt_tt:
            self.Yn2 = self.Yn1
            self.Yn1 = self.Yn

            v0 = self.data[self.lt_tt] if 0 <= self.lt_tt < len(self.data) else self.data[0]
            v1 = self.data[self.lt_tt - self.LPn] if 0 <= self.lt_tt - self.LPn < len(self.data) else self.data[0]
            v2 = self.data[self.lt_tt - self.LP2n] if 0 <= self.lt_tt - self.LP2n < len(self.data) else self.data[0]

            if v0 != -32768 and v1 != -32768 and v2 != -32768:
                self.Yn = 2 * self.Yn1 - self.Yn2 + v0 - 2 * v1 + v2

            dy = int((self.Yn - self.Yn1) / self.LP2n)
            self.lt_tt += 1
            self.et = int(np.sqrt(self.lfsc + dy * dy))

            id = self.lt_tt % self.BUFLN
            self.ebuf[id] = self.et
            id2 = (self.lt_tt - self.LTwindow) % self.BUFLN
            self.aet += self.et - self.ebuf[id2]
            self.lbuf[id] = self.aet

        id3 = t % self.BUFLN
        return self.lbuf[id3]

    def detect(self, data):
        self.initialize(data)
        qrs = []
        jpoints = []
        Tm = int(self.TmDEF / 5.0)

        t1 = min(self.fs * 8, int(self.BUFLN * 0.5))
        T0 = sum(self.ltsamp(t) for t in range(1, t1 + 1)) / t1
        Ta = 3 * T0
        T1 = 2 * T0

        t = 1
        learning = True
        timer_d = 0

        while t < len(self.data):
            if learning and t > t1:
                learning = False
                T1 = T0
                t = 1
                continue
            elif learning:
                T1 = 2 * T0

            if self.ltsamp(t) > T1:
                timer_d = 0
                maxd = self.ltsamp(t)
                mind = maxd

                for tt in range(t + 1, t + self.EyeClosing // 2):
                    maxd = max(maxd, self.ltsamp(tt))
                for tt in range(t - 1, t - self.EyeClosing // 2, -1):
                    mind = min(mind, self.ltsamp(tt))

                if maxd > mind + 10:
                    onset = int(maxd / 100) + 2
                    tpq = t - 5
                    for tt in range(t, t - self.EyeClosing // 2, -1):
                        diffs = [self.ltsamp(tt - i) for i in range(5)]
                        if all(diffs[i] - diffs[i + 1] < onset for i in range(4)):
                            tpq = tt - self.LP2n
                            break

                    if not learning and tpq < len(self.data):
                        qrs.append(tpq)
                        if self.jflag:
                            tj = t + 5
                            for tt in range(t, t + self.EyeClosing // 2):
                                if self.ltsamp(tt) > maxd - int(maxd / 10):
                                    tj = tt
                                    break
                            if tj < len(self.data):
                                jpoints.append(tj)

                    Ta += (maxd - Ta) / 10
                    T1 = Ta / 3
                    t += self.EyeClosing
            elif not learning:
                timer_d += 1
                if timer_d > self.ExpectPeriod and Ta > Tm:
                    Ta -= 1
                    T1 = Ta / 3
            t += 1

        return qrs, jpoints if self.jflag else []
