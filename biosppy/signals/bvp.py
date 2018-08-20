# -*- coding: utf-8 -*-
"""
biosppy.signals.bvp
-------------------

This module provides methods to process Blood Volume Pulse (BVP) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np

# local
from . import tools as st
from .. import plotting, utils


def bvp(signal=None, sampling_rate=1000., show=True):
    """Process a raw BVP signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw BVP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered BVP signal.
    onsets : array
        Indices of BVP pulse onsets.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=4,
                                      frequency=[1, 8],
                                      sampling_rate=sampling_rate)

    # find onsets
    onsets, = find_onsets(signal=filtered, sampling_rate=sampling_rate)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=onsets,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    ts_hr = ts[hr_idx]

    # plot
    if show:
        plotting.plot_bvp(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          onsets=onsets,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, onsets, ts_hr, hr)
    names = ('ts', 'filtered', 'onsets', 'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def find_onsets(signal=None, sampling_rate=1000., sm_size=None, size=None,
                alpha=2., wrange=None, d1_th=0, d2_th=None):
    """Determine onsets of BVP pulses.

    Skips corrupted signal parts.
    Based on the approach by Zong *et al.* [Zong03]_.

    Parameters
    ----------
    signal : array
        Input filtered BVP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    sm_size : int, optional
        Size of smoother kernel (seconds).
        Defaults to 0.25
    size : int, optional
        Window to search for maxima (seconds).
        Defaults to 5
    alpha : float, optional
        Normalization parameter.
        Defaults to 2.0
    wrange : int, optional
        The window in which to search for a peak (seconds).
        Defaults to 0.1
    d1_th : int, optional
        Smallest allowed difference between maxima and minima.
        Defaults to 0
    d2_th : int, optional
        Smallest allowed time between maxima and minima (seconds),
        Defaults to 0.15

    Returns
    -------
    onsets : array
        Indices of BVP pulse onsets.
    
    References
    ----------
    .. [Zong03] W Zong, T Heldt, GB Moody and RG Mark, "An Open-source
       Algorithm to Detect Onset of Arterial Blood Pressure Pulses",
       IEEE Comp. in Cardiology, vol. 30, pp. 259-262, 2003

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # parameters
    sm_size = 0.25 if not sm_size else sm_size
    sm_size = int(sm_size * sampling_rate)
    size = 5 if not size else size
    size = int(size * sampling_rate)
    wrange = 0.1 if not wrange else wrange
    wrange = int(wrange * sampling_rate)
    d2_th = 0.15 if not d2_th else d2_th
    d2_th = int(d2_th * sampling_rate)

    length = len(signal)

    # slope sum function
    dy = np.diff(signal)
    dy[dy < 0] = 0

    ssf, _ = st.smoother(signal=dy, kernel='boxcar', size=sm_size, mirror=True)

    # main loop
    start = 0
    stop = size
    if stop > length:
        stop = length

    idx = []

    while True:
        sq = np.copy(signal[start:stop])
        sq -= sq.mean()
        # sq = sq[1:]
        ss = 25 * ssf[start:stop]
        sss = 100 * np.diff(ss)
        sss[sss < 0] = 0
        sss = sss - alpha * np.mean(sss)

        # find maxima
        pk, pv = st.find_extrema(signal=sss, mode='max')
        pk = pk[np.nonzero(pv > 0)]
        pk += wrange
        dpidx = pk

        # analyze between maxima of 2nd derivative of ss
        detected = False
        for i in range(1, len(dpidx) + 1):
            try:
                v, u = dpidx[i - 1], dpidx[i]
            except IndexError:
                v, u = dpidx[-1], -1

            s = sq[v:u]
            Mk, Mv = st.find_extrema(signal=s, mode='max')
            mk, mv = st.find_extrema(signal=s, mode='min')

            try:
                M = Mk[np.argmax(Mv)]
                m = mk[np.argmax(mv)]
            except ValueError:
                continue

            if (s[M] - s[m] > d1_th) and (m - M > d2_th):
                idx += [v + start]
                detected = True

        # next round continues from previous detected beat
        if detected:
            start = idx[-1] + wrange
        else:
            start += size

        # stop condition
        if start > length:
            break

        # update stop
        stop += size
        if stop > length:
            stop = length

    idx = np.array(idx, dtype='int')

    return utils.ReturnTuple((idx,), ('onsets',))
