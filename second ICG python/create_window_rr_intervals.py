def create_window_rr_intervals(tNN, NN, HRVparams, option='normal'):
    """
    Create window start times for RR interval analysis.

    Parameters:
        tNN (list or np.ndarray): Time of RR intervals (in seconds)
        NN (list or np.ndarray): RR intervals (in seconds)
        HRVparams (dict): Configuration parameters
        option (str): One of 'normal', 'af', 'sqi', 'mse', 'dfa', 'HRT'

    Returns:
        list: Start times of analysis windows (in seconds)
    """
    import numpy as np

    if tNN is None:
        raise ValueError("Need to supply time to create windows")
    
    if not HRVparams:
        option = 'normal'

    increment = HRVparams.get('increment')
    windowlength = HRVparams.get('windowlength')
    win_tol = HRVparams.get('MissingDataThreshold')

    if option == 'af':
        increment = HRVparams['af']['increment']
        windowlength = HRVparams['af']['windowlength']
    elif option == 'mse':
        increment = HRVparams['MSE']['increment'] * 3600
        windowlength = HRVparams['MSE']['windowlength'] * 3600
        if not increment:
            return [0]  # use entire signal
    elif option == 'dfa':
        increment = HRVparams['DFA']['increment'] * 3600
        windowlength = HRVparams['DFA']['windowlength'] * 3600
        if not increment:
            return [0]  # use entire signal
    elif option == 'sqi':
        increment = HRVparams['sqi']['increment']
        windowlength = HRVparams['sqi']['windowlength']
    elif option == 'HRT':
        increment = HRVparams['HRT']['increment']
        windowlength = HRVparams['HRT']['windowlength'] * 3600
        if windowlength > tNN[-1]:
            return [0]

    tNN = np.array(tNN)
    NN = np.array(NN) if NN is not None else []

    nx = int(np.floor(tNN[-1]))
    overlap = windowlength - increment
    Nwinds = int((nx - overlap) // (windowlength - overlap))

    window_rr_intervals = list((np.arange(Nwinds) * (windowlength - overlap)).astype(float))

    if option not in ['af', 'sqi']:
        t_window_start = 0.0
        i = 0
        result_intervals = []

        while t_window_start <= tNN[-1] - windowlength + increment:
            t_win = tNN[(tNN >= t_window_start) & (tNN < t_window_start + windowlength)]

            if NN.size > 0:
                nn_win = NN[(tNN >= t_window_start) & (tNN < t_window_start + windowlength)]
            else:
                nn_win = (windowlength / len(t_win)) * np.ones(len(t_win))

            win_start = t_window_start

            t_window_start += increment

            upper_lim = HRVparams['preprocess']['upperphysiolim']
            lower_lim = HRVparams['preprocess']['lowerphysiolim']
            nn_win = nn_win[(nn_win <= upper_lim) & (nn_win >= lower_lim)]

            truelength = np.sum(nn_win)
            if truelength < (windowlength * (1 - win_tol)):
                result_intervals.append(np.nan)
            else:
                result_intervals.append(win_start)

            i += 1
        return result_intervals

    return window_rr_intervals
