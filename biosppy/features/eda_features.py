import numpy as np
import pyhrv
from .. import utils
from .. import eda


def eda_features(signal=None, sampling_rate=1000.):
    """Compute EDA characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    onsets : list
        Signal EDR events onsets.

    pks : list
        Signal EDR events peaks.

    amps : list
        Signal EDR events Amplitudes.

    phasic_rate : list
        Signal EDR events rate in 60s.

    rise_ts : list
        Rise times, i.e. onset-peak time difference.

    half_rise : list
        Half Rise times, i.e. time between onset and 50% amplitude.

    half_rec : list
        Half Recovery times, i.e. time between peak and 63% amplitude.

    six_rise : list
        63 % rise times, i.e. time between onset and 63% amplitude.

    six_rec : list
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """
    # ensure numpy
    signal = np.array(signal)

    # get EDR signal
    try:
        scr = eda.get_scr(signal, sampling_rate)
    except:
        scr = []

    # onsets, pks, amps
    onsets, pks, amps, _ = eda.get_eda_param(scr, sampling_rate)

    # phasic_rate
    try:
        phasic_rate = sampling_rate * (60. / np.diff(pks))
    except:
        phasic_rate = None

    # rise_ts
    try:
        rise_ts = list(np.array(pks - onsets))
    except:
        rise_ts = None

    # half, six, half_rise, half_rec, six_rec
    _, _, half_rise, half_rec, six_rise, six_rec = eda.edr_times(scr, onsets, pks)

    # output
    args = (onsets, pks, amps, phasic_rate, rise_ts, half_rise, half_rec, six_rise, six_rec)
    names = ('onsets', 'peaks', 'amplitudes', 'phasic_rate', 'rise_ts', 'half_rise', 'half_rec', 'six_rise', 'six_rec')

    return utils.ReturnTuple(args, names)
