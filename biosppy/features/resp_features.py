import numpy as np
from .. import utils
from .. import tools as st
import json


def get_restrate(zeros, sampling_rate):
    """Compute Respiration rate.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    rate : float
        Respiration rate.
    """
    beats = zeros[::2]

    if len(beats) < 2:
        rate_idx = []
        rate = []
    else:
        # compute respiration rate
        rate_idx = beats[1:]
        rate = sampling_rate * (1. / np.diff(beats))

        # physiological limits
        indx = np.nonzero(rate <= 0.35)
        rate_idx = rate_idx[indx]
        rate = rate[indx]

        # smooth with moving average
        size = 3
        rate, _ = st.smoother(signal=rate,
                              kernel='boxcar',
                              size=size,
                              mirror=True)
    args = (rate,)
    names = ('respiration_rate',)
    return utils.ReturnTuple(args, names)


def calc_inhale(sig, zeros):
    """Compute Respiration Inhalation volumes.

    Parameters
    ----------
    signal : array
        Input signal.
    zeros : list
        Indexes of signal zero-crossings.
    Returns
    -------
    inh : list
        Inhalation volumes.
    """
    inh = None
    ext_idx = []
    if sig != []:
        ext_idx, v = st.find_extrema(signal=sig, mode='max')
    if zeros is not None:
        if ext_idx != []:
            if np.min(ext_idx) < np.min(zeros):
                while zeros[0] > ext_idx[0]:
                    if len(ext_idx) > 1:
                        ext_idx = ext_idx[1:]
                    else:
                        ext_idx = []
                        break
        if ext_idx != [] and zeros is not None:
            zeros = zeros[::2]
            c = min(len(zeros), len(ext_idx))
            inh = []
            for i in range(c):
                inh.append(np.trapz(sig[zeros[i]:ext_idx[i]]))
    args = (inh,)
    names = ('inhalation_vol',)
    return utils.ReturnTuple(args, names)


def calc_exhale(sig, zeros):
    """Compute Respiration Exhalation volumes.

    Parameters
    ----------
    signal : array
        Input signal.
    zeros : list
        Indexes of signal zero-crossings.
    Returns
    -------
    exh : list
        Exhalation volumes.
    """
    ext_idx = None
    exh = None
    if sig != []:
        ext_idx, v = st.find_extrema(signal=sig, mode='max')
    if zeros is not None and ext_idx != []:
        if np.min(zeros) < np.min(ext_idx):
            while zeros[0] < ext_idx[0]:
                if len(zeros) > 1:
                    zeros = zeros[1:]
                else:
                    zeros = []
                    break
        if ext_idx != []  and zeros != [] :
            zeros = zeros[::2]
            c = min(len(zeros), len(ext_idx))
            exh = []
            for i in range(c):
                exh.append(np.trapz(sig[ext_idx[i]:zeros[i]]))

    args = (exh,)
    names = ('exhalation_vol',)
    return utils.ReturnTuple(args, names)


def calc_inhExhRatio(inh, exh):
    """Compute Respiration inhalation/exhalation ratio.

    Parameters
    ----------
    inh : list
        Inhalation Volumes
    exh : list
        Exhalation Volumes
    Returns
    -------
    ratio : list
        Inhalation/exhalation ratio.
    """
    ratio = None
    if inh is not None and exh is not None:
        c = min(len(inh), len(exh))
        ratio = np.array(inh[:c])/np.array(exh[:c])
        ratio = np.abs(ratio.tolist())
    args = (ratio,)
    names = ('inh_exh_ratio',)
    return utils.ReturnTuple(args, names)


def calc_inhdur(sig, zeros):
    """Compute Respiration Inhalation time duration.

    Parameters
    ----------
    signal : array
        Input signal.
    zeros : list
        Indexes of signal zero-crossings.
    Returns
    -------
    inh : float
        Inhalation time duration.
    """
    inh = None
    ext_idx = None
    if sig != []:
        ext_idx, v = st.find_extrema(signal=sig, mode='max')
    if zeros is not None and ext_idx != []:
        if np.min(ext_idx) < np.min(zeros):
            while zeros[0] > ext_idx[0]:
                if len(ext_idx) > 1:
                    ext_idx = ext_idx[1:]
                else:
                    ext_idx = []
                    break

        if ext_idx != []  and zeros is not None:
            zeros = zeros[::2]
            c = min(len(zeros), len(ext_idx))
            inh = []
            for i in range(c):
                inh.append(ext_idx[i]-zeros[i])

    args = (inh,)
    names = ('inhdur',)
    return utils.ReturnTuple(args, names)


def calc_exhdur(sig, zeros):
    """Compute Respiration exhalation time duration.

    Parameters
    ----------
    signal : array
        Input signal.
    zeros : list
        Indexes of signal zero-crossings.
    Returns
    -------
    exh : float
        Exhalation time duration.
    """
    exh = None
    ext_idx = None
    if sig != []:
        ext_idx, v = st.find_extrema(signal=sig, mode='max')
    if zeros is not None and ext_idx != []:
        if np.min(zeros) < np.min(ext_idx):
            while zeros[0] < ext_idx[0]:
                if len(zeros) > 1:
                    zeros = zeros[1:]
                else:
                    zeros = []
                    break

        if ext_idx != [] and zeros != []:
            exh = []
            zeros = zeros[::2]
            c = min(len(zeros), len(ext_idx))

            for i in range(c):
                exh.append(sig(zeros[i])-sig(ext_idx[i]))
    args = (exh,)
    names = ('exhdur',)
    return utils.ReturnTuple(args, names)


def resp_features(signal=None, sampling_rate=1000.):
    """Compute Respiration characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    zeros : list
        Signal zero crossing indexes.
    hr : list
        Respiration rate.
    inhale : list
        Inhalation volume.
    exhale : list
        Exhalation volume.
    inhExhRatio : list
        Ratio between Inhalation and Exhalation.
    inhale_dur : list
        Inhalation time duration.
    exhale_dur : list
        Exhalation time duration.
    """
    signal = np.array(signal)
    dict = json.load(open('resp_features_log.json'))
    args, names = [], []

    # Zero crossings
    try:
        zeros, = st.zero_cross(signal=signal, detrend=True)
    except:
        zeros = None

    if dict['zeros']['use'] == 'yes':
        args += [zeros]
        names += ['zeros']

    min = st.find_extrema(signal, 'min')
    if dict['hr']['use'] == 'yes':
        # Respiration rate
        try:
            hr = get_restrate(zeros, sampling_rate)[0]
        except:
            hr = None
        args += [hr]
        names += ['hr']

    try:
        inhale = calc_inhale(signal, min)[0]
    except:
        inhale = None

    if dict['inhale']['use'] == 'yes':
        args += [inhale]
        names += ['inhale']
    try:
        exhale = calc_exhale(signal, min)[0]
    except:
        exhale = None
    if dict['exhale']['use'] == 'yes':
        args += [exhale]
        names += ['exhale']

    if dict['inhExhRatio']['use'] == 'yes':
        try:
            inhExhRatio = calc_inhExhRatio(inhale, exhale)[0]
        except:
            inhExhRatio = None
        args += [inhExhRatio]
        names += ['inhExhRatio']

    if dict['inhale_dur']['use'] == 'yes':
        try:
            inhale_dur = calc_inhdur(signal, min)[0]
        except:
            inhale_dur = None
        args += [inhale_dur]
        names += ['inhale_dur']

    if dict['exhale_dur']['use'] == 'yes':
        try:
            exhale_dur = calc_exhdur(signal, min)[0]
        except:
            exhale_dur = None
        exhale_dur += [exhale_dur]
        exhale_dur += ['exhale_dur']

    return utils.ReturnTuple(tuple(args), tuple(names))
