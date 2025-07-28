import numpy as np
from create_window_rr_intervals import create_window_rr_intervals
from run_sqi import run_sqi


def bsqi(ann1, ann2, HRVparams):
    """
    Measure SQI by comparing two QRS detection annotations.

    Parameters:
        ann1 : np.ndarray - First annotation array (sample indices)
        ann2 : np.ndarray - Second annotation array (sample indices)
        HRVparams : dict or object with keys:
            - Fs : int - Sampling frequency
            - sqi : dict with keys 'windowlength', 'TimeThreshold', 'margin'

    Returns:
        F1 : np.ndarray - SQI score for each window
        StartIdxSQIwindows : np.ndarray - Start times of the SQI windows
    """
    if HRVparams is None:
        raise ValueError("HRVparams must be provided")

    windowlength = HRVparams['sqi']['windowlength']
    threshold = HRVparams['sqi']['TimeThreshold']
    margin = HRVparams['sqi']['margin']
    fs = HRVparams['Fs']

    
    ann1 = np.array(ann1[0]).flatten() / fs
    ann2 = np.array(ann2[0]).flatten() / fs


    endtime = max(ann1[-1], ann2[-1])
    time = np.arange(1/fs, endtime + 1/fs, 1/fs)

    # Create windows
    StartIdxSQIwindows = create_window_rr_intervals(time, None, HRVparams, 'sqi')

    # Initialize SQI results
    F1 = np.full(len(StartIdxSQIwindows), np.nan)

    for seg in range(len(StartIdxSQIwindows)):
        if not np.isnan(StartIdxSQIwindows[seg]):
            try:
                # Extract RR intervals within window
                idx_ann1_in_win = np.where((ann1 >= StartIdxSQIwindows[seg]) &
                                           (ann1 < StartIdxSQIwindows[seg] + windowlength))[0]
                a1 = ann1[idx_ann1_in_win] - StartIdxSQIwindows[seg]
                a2 = ann2 - StartIdxSQIwindows[seg]

                F1[seg] = run_sqi(a1, a2, threshold, margin, windowlength, fs)
            except Exception:
                continue

    return F1, StartIdxSQIwindows
