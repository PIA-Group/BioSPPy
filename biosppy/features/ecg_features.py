import numpy as np
import pyhrv
from .. import utils
from .. import ecg
from .. import tools as st
import json

def pNN(nni, TH):
    """Compute the percentage of the number of times the HRV differed by more than 50ms.

    Parameters
    ----------
    nni : array
        Input signal.
    TH : float
        Threshold.
    Returns
    -------
    p : float
        Percentage of the number of times the HRV differed by more than TH*ms over the total number of NN intervals.
    """

    try:
        rint = np.abs(np.diff(nni))
        p = len(np.where(rint > TH)[0]) / len(rint) * 100
    except:
        p = None
    args = (p,)
    names = ('pNN',)
    return utils.ReturnTuple(args, names)


def ecg_features(signal=None, sampling_rate=1000.):
    """Compute ECG characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    rpeaks : list
        R-peak location indices.

    nni: list
        NN intervals in ms or s.

    hr : list
        Instantaneous heart rate in bpm.

    pnn20 : float
        Ratio between NN20 and total number of NN intervals in percentage.

    pnn50 : float
        Ratio between NN50 and total number of NN intervals in percentage.

    sdnn_index : float
        Mean of the standard deviations of all the NN intervals for each 5 min segment.

    sdann : float
        Standard deviation of the average NN intervals for each 5 min segment.

    fft_peak_VLF : float
        Peak frequencies of the very low frequency bands [0.00Hz - 0.04Hz] in Hz.

    fft_peak_LF : float
        Peak frequencies of the low frequency bands [0.04Hz - 0.15Hz] in Hz.

    fft_peak_HF : float
        Peak frequencies of the high low frequency bands [0.15Hz - 0.40Hz] in Hz.

    fft_abs_VLF : float
        Absolute powers of the very low frequency bands [0.00Hz - 0.04Hz] in ms^2.

    fft_abs_LF : float
        Absolute powers of the low frequency bands [0.04Hz - 0.15Hz] in ms^2.

    fft_abs_HF : float
        Absolute powers of the high frequency bands [0.15Hz - 0.40Hz] in ms^2.

    fft_rel_VLF : float
        Relative powers of the very low frequency bands [0.00Hz - 0.04Hz] in %.

    fft_rel_LF : float
        Relative powers of the low frequency bands [0.04Hz - 0.15Hz] in %.

    fft_rel_HF : float
        Relative powers of the high frequency bands [0.15Hz - 0.40Hz] in %.

    fft_log_VLF : float
        Logarithmic powers of the very low frequency bands [0.00Hz - 0.04Hz].

    fft_log_LF : float
        Logarithmic powers of the low frequency bands [0.04Hz - 0.15Hz].

    fft_log_HF : float
        Logarithmic powers of the high frequency bands [0.15Hz - 0.40Hz].

    fft_total : float
        Total power over all frequency bands in ms^2.

    fft_ratio : list
        LF/HF ratio.

    References
    ----------
    Gomes, Pedro & Margaritoff, Petra & Plácido da Silva, Hugo. (2019). pyHRV: Development and Evaluation of an Open-Source Python Toolbox for Heart Rate Variability (HRV).
    """
    signal = np.array(signal)
    dict = json.load(open('ecg_features_log.json'))
    args, names = [], []

    try:
        rpeaks = np.array(ecg.get_rpks(signal, sampling_rate))
    except:
        rpeaks = None

    if dict['rpeaks']['use'] == 'yes':
        args += [rpeaks]
        names += ['rpeaks']

    try:
        nni = pyhrv.tools.nn_intervals(signal[rpeaks].astype(float))
    except:
        nni = None

    if dict['nni']['use'] == 'yes':
        args += [nni]
        names += ['nni']

    if dict['hr']['use'] == 'yes':
        try:
            _, hr = st.get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
        except:
            hr = None
        args += [hr]
        names += ['hr']

    if dict['pnn20']['use'] == 'yes':
        pnn20 = pNN(nni, 20)[0]
        args += [pnn20]
        names += ['pnn20']

    if dict['pnn50']['use'] == 'yes':
        pnn50 = pNN(nni, 50)[0]
        args += [pnn50]
        names += ['pnn50']

    if dict['sdnn_index']['use'] == 'yes':
        try:
            sdnn_index = pyhrv.time_domain.sdnn_index(nni)[0]
        except:
            sdnn_index = None
        args += [sdnn_index]
        names += ['sdnn_index']

    if dict['sdann']['use'] == 'yes':
        try:
            sdann = pyhrv.time_domain.sdann(nni)[0]
        except:
            sdann = None
        args += [sdann]
        names += ['sdann']

    # Spectral features
    freq_param = pyhrv.frequency_domain.welch_psd(nni, mode='dev', show=False, show_param=False, legend=False)[0]
    try:
        freq_param = pyhrv.frequency_domain.welch_psd(nni, mode='dev', show=False, show_param=False, legend=False)[0]
    except:
        print("Error in line 173")

    try:
        if dict['fft_peak_VLF']['use'] == 'yes':
            fft_peak_VLF = freq_param['fft_peak'][0]
    except:
        fft_peak_VLF = None
    args += [fft_peak_VLF]
    names += ['fft_peak_VLF']
    try:
        if dict['fft_peak_LF']['use'] == 'yes':
            fft_peak_LF = freq_param['fft_peak'][1]
    except:
        fft_peak_LF = None
    args += [fft_peak_LF]
    names += ['fft_peak_LF']
    try:
        if dict['fft_peak_HF']['use'] == 'yes':
            fft_peak_HF = freq_param['fft_peak'][2]
    except:
        fft_peak_HF = None
    args += [fft_peak_HF]
    names += ['fft_peak_HF']
    try:
        if dict['fft_abs_VLF']['use'] == 'yes':
            fft_abs_VLF = freq_param['fft_abs'][0]
    except:
        fft_abs_VLF = None
    args += [fft_abs_VLF]
    names += ['fft_abs_VLF']
    try:
        if dict['fft_abs_LF']['use'] == 'yes':
            fft_abs_LF = freq_param['fft_abs'][1]
    except:
        fft_abs_LF = None
    args += [fft_abs_LF]
    names += ['fft_abs_LF']
    try:
        if dict['fft_abs_HF']['use'] == 'yes':
            fft_abs_HF = freq_param['fft_abs'][2]
    except:
        fft_abs_HF = None
    args += [fft_abs_HF]
    names += ['fft_abs_HF']
    try:
        if dict['fft_rel_VLF']['use'] == 'yes':
            fft_rel_VLF = freq_param['fft_rel'][0]
    except:
        fft_rel_VLF = None
    args += [fft_rel_VLF]
    names += ['fft_rel_VLF']
    try:
        if dict['fft_rel_LF']['use'] == 'yes':
            fft_rel_LF = freq_param['fft_rel'][1]
    except:
        fft_rel_LF = None
    args += [fft_rel_LF]
    names += ['fft_rel_LF']
    try:
        if dict['fft_rel_HF']['use'] == 'yes':
            fft_rel_HF = freq_param['fft_rel'][2]
    except:
        fft_rel_HF = None
    args += [fft_rel_HF]
    names += ['fft_rel_HF']
    try:
        if dict['fft_log_VLF']['use'] == 'yes':
            fft_log_VLF = freq_param['fft_log'][0]
    except:
        fft_log_VLF = None
    args += [fft_log_VLF]
    names += ['fft_log_VLF']
    try:
        if dict['fft_log_LF']['use'] == 'yes':
            fft_log_LF = freq_param['fft_log'][1]
    except:
        fft_log_LF = None
    args += [fft_log_LF]
    names += ['fft_log_LF']
    try:
        if dict['fft_log_HF']['use'] == 'yes':
            fft_log_HF = freq_param['fft_log'][2]
    except:
        fft_log_HF = None
    args += [fft_log_HF]
    names += ['fft_log_HF']
    try:
        if dict['fft_total']['use'] == 'yes':
            fft_total = freq_param['fft_total']
    except:
        fft_total = None
    args += [fft_total]
    names += ['fft_total']
    try:
        if dict['fft_ratio']['use'] == 'yes':
            fft_ratio = freq_param['fft_ratio']
    except:
        fft_ratio = None
    args += [fft_ratio]
    names += ['fft_ratio']



    return utils.ReturnTuple(tuple(args), tuple(names))
