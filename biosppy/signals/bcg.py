# -*- coding: utf-8 -*-
"""
biosppy.signals.bcg
-------------------

This module provides methods to process Ballistocardiographic (BCG) signals.
Implemented code assumes a single-channel head-to-foot like BCG signal.

:author: Guillaume Cathelain

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip

# 3rd party
import numpy as np
import scipy.signal as ss

# local
from . import tools as st
from .. import plotting, utils
from ./ecg import correct_rpeaks as correct_peaks
from ./ecg import extract_heartbeats

def bcg(signal=None, sampling_rate=1000., show=True):
    """Process a raw BCG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered BCG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
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
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[2, 20],
                                      sampling_rate=sampling_rate)

    # segment
    hpeaks,ipeaks,jpeaks = bsegpp_segmenter(signal=signal, sampling_rate=sampling_rate)

    # extract templates
    templates, ipeaks = extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=ipeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        plotting.plot_bcg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          ipeaks=ipeaks,
                          templates_ts=ts_tmpl,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = ('ts', 'filtered', 'ipeaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def bsegpp_segmenter(signal=None, sampling_rate=1000., thresholds= [0.5,5],R=0.3, t1=0.6, M=0.2, N=0.2):
    """BSEG++ BCG cycle extraction algorithm.

    Follows the approach by Akhbardeh et al. [Akhb02]_.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    thresholds : array
        Lower and upper amplitude threshold values for local maxima of the absolute coarse signal.
    R : float
        Range of local minima search for final synchronization points (seconds). Empirically 0.1<R<0.5.
    t1 : float
        Minimum delay between final synchronization points (seconds). Empirically 0.4<t1<0.6
    M : float
        Maximum delay between H and I waves (seconds). Empirically 0.1<M<0.5.
    N : float
        Maximum delay between I and J waves (seconds). Empirically 0.1<N<0.5.

    Returns
    -------
    jpeaks : array
        J-peak location indices.

    References
    ----------
    .. [Akhb02] A. Akhbardeh, B. Kaminska, K. Tavakolian, "BSeg++: A modified
    Blind Segmentation Method for Ballistocardiogram Cycle Extraction",
    Proceedings of the 29th Annual International Conference of the IEEE EMBS, 2007

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # 1) normalization
    signal -= mean(signal)
    signal /= max(abs(signal))
    signal *= 5

    # 2) filtering
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[2, 20],
                                      sampling_rate=sampling_rate)

    # 3) extract coarse bcg
    order = int(0.3 * sampling_rate)
    coarse, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[1, 2],
                                      sampling_rate=sampling_rate)

    # synchronization points
    # a) local maxima of absolute coarse BCG with distance constraint
    cntr,properties = ss.find_peaks(abs(coarse),
                                        height=thresholds,
                                        threshold=None,
                                        distance=int(t1*sampling_rate),
                                        prominence=None,
                                        width=None,
                                        wlen=None,
                                        rel_height=0.5,
                                        plateau_size=None)
    # b) final synchronization points
    p, = correct_peaks(signal=-filtered,
                             rpeaks=cntr,
                             sampling_rate=sampling_rate,
                             tol=R)

    # 6) define I waves
    ipeaks = p

    # 7) search for H waves
    hpeaks, = correct_peaks(signal=filtered,
                             rpeaks=Iindx - int(M*sampling_rate/2),
                             sampling_rate=sampling_rate,
                             tol = M/2)
    # 8) search for J waves
    jpeaks, = correct_peaks(signal=filtered,
                             rpeaks=Iindx + int(N*sampling_rate/2),
                             sampling_rate=sampling_rate,
                             tol= N/2)

    return utils.ReturnTuple((hpeaks,ipeaks,jpeaks), ('hpeaks','ipeaks','jpeaks'))
