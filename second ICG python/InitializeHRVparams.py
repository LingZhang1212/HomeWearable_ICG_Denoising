import os
from datetime import datetime

def InitializeHRVparams(project_name='none'):
    HRVparams = {}

    # 1. Project settings
    if project_name == 'demo_NSR':
        HRVparams['Fs'] = 125
        HRVparams['readdata'] = os.path.join('TestData', 'Physionet_nsr2db')
        HRVparams['writedata'] = os.path.join('OutputData', 'ResultsNSR')
        HRVparams['ext'] = 'ecg'
    elif project_name == 'demoICU':
        HRVparams['Fs'] = 128
        HRVparams['readdata'] = 'TestData'
        HRVparams['writedata'] = os.path.join('OutputData', 'ResultsICU')
        HRVparams['ext'] = 'mat'
    elif project_name == 'demoAF':
        HRVparams['Fs'] = 128
        HRVparams['readdata'] = 'TestData'
        HRVparams['writedata'] = os.path.join('OutputData', 'ResultsAFData')
        HRVparams['ext'] = 'mat'
    else:
        HRVparams['Fs'] = None
        HRVparams['readdata'] = ''
        HRVparams['writedata'] = f'{project_name}_Results'
        HRVparams['ext'] = ''

    os.makedirs(HRVparams['writedata'], exist_ok=True)

    # 2. Confidence level
    HRVparams['data_confidence_level'] = 1

    # 3. Window settings
    HRVparams.update({
        'windowlength': 300,
        'increment': 30,
        'numsegs': 5,
        'RejectionThreshold': 0.20,
        'MissingDataThreshold': 0.15
    })

    # 5. Debug
    HRVparams.update({
        'rawsig': 0,
        'debug': 0
    })

    # 6. SQI settings
    HRVparams['sqi'] = {
        'LowQualityThreshold': 0.9,
        'windowlength': 10,
        'increment': 1,
        'TimeThreshold': 0.1,
        'margin': 2
    }

    # 7. Preprocess settings
    HRVparams['preprocess'] = {
        'figures': 0,
        'gaplimit': 2,
        'per_limit': 0.2,
        'forward_gap': 3,
        'method_outliers': 'rem',
        'lowerphysiolim': 60/160,
        'upperphysiolim': 60/30,
        'method_unphysio': 'rem',
        'threshold1': 0.9,
        'minlength': 30
    }

    # 8. AF detection
    HRVparams['af'] = {
        'on': 1,
        'windowlength': 30,
        'increment': 30
    }

    HRVparams['PVC'] = {
        'qrsth': 0.1
    }

    # 9. Time domain
    HRVparams['timedomain'] = {
        'on': 1,
        'dataoutput': 0,
        'alpha': 50,
        'win_tol': 0.15
    }

    # 10. Frequency domain
    HRVparams['freq'] = {
        'on': 1,
        'limits': [[0, 0.0033], [0.0033, 0.04], [0.04, 0.15], [0.15, 0.4]],
        'zero_mean': 1,
        'method': 'lomb',
        'plot_on': 0,
        'debug_sine': 0,
        'debug_freq': 0.15,
        'debug_weight': 0.03,
        'normalize_lomb': 0,
        'burg_poles': 15,
        'resampling_freq': 7,
        'resample_interp_method': 'cub',
        'resampled_burg_poles': 100
    }

    # 11. SDANN and SDNNI
    HRVparams['sd'] = {
        'on': 1,
        'segmentlength': 300
    }

    # 12. PRSA
    HRVparams['prsa'] = {
        'on': 1,
        'win_length': 30,
        'thresh_per': 20,
        'plot_results': 0,
        'scale': 2,
        'min_anch': 20
    }

    # 13. Peak Detection
    HRVparams['PeakDetect'] = {
        'REF_PERIOD': 0.250,
        'THRES': 0.6,
        'fid_vec': [],
        'SIGN_FORCE': [],
        'debug': 0,
        'ecgType': 'MECG',
        'windows': 15
    }

    # 14. Entropy
    HRVparams['MSE'] = {
        'on': 1,
        'windowlength': None,
        'increment': None,
        'RadiusOfSimilarity': 0.15,
        'patternLength': 2,
        'maxCoarseGrainings': 20,
        'method': 'fir',
        'moment': 'mean',
        'constant_r': 1
    }

    HRVparams['Entropy'] = {
        'on': 1,
        'RadiusOfSimilarity': 0.15,
        'patternLength': 2
    }

    # 15. DFA
    HRVparams['DFA'] = {
        'on': 1,
        'windowlength': None,
        'increment': None,
        'minBoxSize': 4,
        'maxBoxSize': None,
        'midBoxSize': 16
    }

    # 16. Poincare
    HRVparams['poincare'] = {
        'on': 1
    }

    # 17. HRT
    HRVparams['HRT'] = {
        'on': 1,
        'BeatsBefore': 2,
        'BeatsAfter': 16,
        'GraphOn': 0,
        'windowlength': 24,
        'increment': 24,
        'filterMethod': 'mean5before'
    }

    # 18. Output
    HRVparams['gen_figs'] = 0
    HRVparams['save_figs'] = 0
    HRVparams['output'] = {
        'format': 'csv',
        'separate': 1,
        'num_win': None,
        'ann_format': 'binary'
    }

    # 19. Time/Filename
    HRVparams['time'] = datetime.now().strftime('%Y%m%d')
    HRVparams['filename'] = f"{HRVparams['time']}_{project_name}"

    return HRVparams
