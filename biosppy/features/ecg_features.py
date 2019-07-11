import numpy as np
import pyhrv
from .. import utils
from .. import ecg
from .. import tools as st


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

    try:
        rpeaks = np.array(ecg.get_rpks(signal, sampling_rate))
    except:
        rpeaks = None

    try:
        nni = pyhrv.tools.nn_intervals(signal[rpeaks].astype(float))
    except:
        nni = None

    try:
        _, hr = st.get_heart_rate(beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
    except:
        hr = None

    pnn20 = pNN(nni, 20)[0]

    pnn50 = pNN(nni, 50)[0]

    try:
        sdnn_index = pyhrv.time_domain.sdnn_index(nni)[0]
    except:
        sdnn_index = None

    try:
        sdann = pyhrv.time_domain.sdann(nni)[0]
    except:
        sdann = None

    # Spectral features
    try:
        freq_param = pyhrv.frequency_domain.welch_psd(nni, mode='dev', show=False, show_param=False, legend=False)[0]

        fft_peak_VLF = freq_param['fft_peak'][0]
        fft_peak_LF = freq_param['fft_peak'][1]
        fft_peak_HF = freq_param['fft_peak'][2]

        fft_abs_VLF = freq_param['fft_abs'][0]
        fft_abs_LF = freq_param['fft_abs'][1]
        fft_abs_HF = freq_param['fft_abs'][2]

        fft_rel_VLF = freq_param['fft_rel'][0]
        fft_rel_LF = freq_param['fft_rel'][1]
        fft_rel_HF = freq_param['fft_rel'][2]

        fft_log_VLF = freq_param['fft_log'][0]
        fft_log_LF = freq_param['fft_log'][1]
        fft_log_HF = freq_param['fft_log'][2]

        fft_total = freq_param['fft_total']
        fft_ratio = freq_param['fft_ratio']

    except:
        fft_peak_VLF, fft_peak_LF, fft_peak_HF, fft_abs_VLF, fft_abs_LF, fft_abs_HF, fft_rel_VLF, fft_rel_LF, fft_rel_HF, fft_log_VLF, fft_log_LF, fft_log_HF, fft_total, fft_ratio = [None] * 14

    # output
    args = (rpeaks, nni, hr , pnn20, pnn50, sdnn_index, sdann, fft_peak_VLF, fft_peak_LF, fft_peak_HF, fft_abs_VLF, fft_abs_LF, fft_abs_HF,
            fft_rel_VLF, fft_rel_LF, fft_rel_HF, fft_log_VLF, fft_log_LF, fft_log_HF, fft_total, fft_ratio)

    names = ('rpeaks', 'nni', 'hr' , 'pnn20', 'pnn50', 'sdnn_index', 'sdann', 'fft_peak_VLF', 'fft_peak_LF', 'fft_peak_HF', 'fft_abs_VLF', 'fft_abs_LF', 'fft_abs_HF',
            'fft_rel_VLF', 'fft_rel_LF', 'fft_rel_HF', 'fft_log_VLF', 'fft_log_LF', 'fft_log_HF', 'fft_total', 'fft_ratio')

    return utils.ReturnTuple(args, names)
