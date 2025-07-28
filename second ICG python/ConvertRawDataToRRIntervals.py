import os
import numpy as np
from run_qrsdet_by_seg import run_qrsdet_by_seg
from run_sqrs import run_sqrs
from wqrsm_fast import wqrsm_fast
from bsqi import bsqi
from write_hea import write_hea
from write_ann import write_ann

def ConvertRawDataToRRIntervals(ECG_RawData, HRVparams, subjectID):
    """
    Convert raw ECG data to RR intervals and perform QRS detection and SQI.

    Parameters:
        ECG_RawData: np.ndarray - raw ECG signal in mV (1D array or Nx1 array)
        HRVparams: dict or object - HRV analysis settings
        subjectID: str - identifier for the record

    Returns:
        t: np.ndarray - RR interval time points (s)
        rr: np.ndarray - RR intervals (s)
        jqrs_ann: np.ndarray - detected QRS locations (samples)
        SQIjw: np.ndarray - SQI comparing jqrs and wqrs
        StartSQIwindows_jw: np.ndarray - start time of SQI windows
    """
    if ECG_RawData.ndim > 1 and ECG_RawData.shape[0] < ECG_RawData.shape[1]:
        ECG_RawData = ECG_RawData.T

    ECG_RawData = ECG_RawData[:, 0] if ECG_RawData.ndim > 1 else ECG_RawData
    GainQrsDetect = 2000

    # QRS detection
    jqrs_ann = run_qrsdet_by_seg(ECG_RawData, HRVparams)
    sqrs_ann = run_sqrs(ECG_RawData * GainQrsDetect, HRVparams, 0)
    wqrs_ann = wqrsm_fast(ECG_RawData * GainQrsDetect, HRVparams['Fs'])

    # SQI comparison
    SQIjs, StartSQIwindows_js = bsqi(jqrs_ann, sqrs_ann, HRVparams)
    SQIjw, StartSQIwindows_jw = bsqi(jqrs_ann, wqrs_ann, HRVparams)

    # RR interval and timing
    rr = np.diff(jqrs_ann) / HRVparams['Fs']
    t = np.array(jqrs_ann[1:]) / HRVparams['Fs']

    # Create Annotation Folder
    WriteAnnotationFolder = os.path.join(HRVparams['writedata'], 'Annotation')
    os.makedirs(WriteAnnotationFolder, exist_ok=True)
    print(f'Creating a new folder: "Annotation", folder is located in {WriteAnnotationFolder}')

    AnnFile = os.path.join(WriteAnnotationFolder, subjectID)
    write_hea(AnnFile, HRVparams['Fs'], len(ECG_RawData), 'jqrs', 1, 0, 'mV')

    # Save annotations
    write_ann(AnnFile, HRVparams, 'jqrs', jqrs_ann)
    write_ann(AnnFile, HRVparams, 'sqrs', sqrs_ann)
    write_ann(AnnFile, HRVparams, 'wqrs', wqrs_ann)

    fakeAnnType = ['S'] * len(SQIjs)
    SQIjw_array = np.array(SQIjw)

    # 将 list 转换为 numpy array
    SQIjw_array = np.array(SQIjw)
    StartSQIwindows_array = np.array(StartSQIwindows_jw)

    # 处理 NaN
    if np.isnan(SQIjw_array).any():
        print("Warning: SQIjw contains NaN. Replacing with 0.")
        SQIjw_array = np.nan_to_num(SQIjw_array)

    # 修正类型不匹配
    if not isinstance(fakeAnnType, list):
        fakeAnnType = ['S'] * len(SQIjw_array)
    elif len(fakeAnnType) != len(SQIjw_array):
        print("Warning: fakeAnnType length mismatch. Regenerating...")
        fakeAnnType = ['S'] * len(SQIjw_array)

    # 输出长度确认
    print("Lengths:", len(StartSQIwindows_array), len(fakeAnnType), len(SQIjw_array))

    # ✅ 正确调用 write_ann：一定要用转换后的 numpy array
    write_ann(
        AnnFile,
        HRVparams,
        'sqijs',
        (StartSQIwindows_array * HRVparams['Fs']).astype(int),
        fakeAnnType,
        (SQIjw_array * 100).round().astype(int)
    )


    
    fakeAnnType = ['S'] * len(SQIjw)

    # 转成 numpy array（确保在 write_ann 前执行）
    StartSQIwindows_array = np.array(StartSQIwindows_jw)
    SQIjw_array = np.array(SQIjw)

    # 替换 NaN 为 0
    if np.isnan(SQIjw_array).any():
        print("Warning: SQIjw contains NaN. Replacing with 0.")
        SQIjw_array = np.nan_to_num(SQIjw_array)

    # 检查类型匹配
    if not isinstance(fakeAnnType, list):
        fakeAnnType = ['S'] * len(SQIjw_array)
    elif len(fakeAnnType) != len(SQIjw_array):
        print("Warning: fakeAnnType length mismatch. Regenerating...")
        fakeAnnType = ['S'] * len(SQIjw_array)

    # 检查长度
    print("Lengths:", len(StartSQIwindows_array), len(fakeAnnType), len(SQIjw_array))

    # ✅ 正确调用
    write_ann(
        AnnFile,
        HRVparams,
        'sqijs',
        (StartSQIwindows_array * HRVparams['Fs']).astype(int),
        fakeAnnType,
        (SQIjw_array * 100).round().astype(int)
    )


    return t, rr, jqrs_ann, SQIjw, StartSQIwindows_jw
