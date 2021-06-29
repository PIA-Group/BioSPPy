# -*- coding: utf-8 -*-
"""
biosppy.signals.bvp
-------------------

This module provides methods to process Blood Volume Pulse (BVP) signals.

-------- DEPRECATED --------
PLEASE, USE THE PPG MODULE
This module was left for compatibility
----------------------------

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
from . import ppg
from .. import plotting, utils


def bvp(signal=None, sampling_rate=1000., path=None, show=True):
    """Process a raw BVP signal and extract relevant signal features using
    default parameters.
    Parameters
    ----------
    signal : array
        Raw BVP signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
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
    onsets,_ = ppg.find_onsets_elgendi2013(signal=filtered, sampling_rate=sampling_rate)

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
                          path=path,
                          show=True)

    # output
    args = (ts, filtered, onsets, ts_hr, hr)
    names = ('ts', 'filtered', 'onsets', 'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


