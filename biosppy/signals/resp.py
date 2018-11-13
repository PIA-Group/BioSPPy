# -*- coding: utf-8 -*-
"""
biosppy.signals.resp
--------------------

This module provides methods to process Respiration (Resp) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np

# local
from . import tools as st
from .. import plotting, utils


def resp(signal=None, sampling_rate=1000., show=True):
    """Process a raw Respiration signal and extract relevant signal features
    using default parameters.

    Parameters
    ----------
    signal : array
        Raw Respiration signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered Respiration signal.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Respiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).

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
                                      order=2,
                                      frequency=[0.1, 0.35],
                                      sampling_rate=sampling_rate)

    # compute zero crossings
    zeros, = st.zero_cross(signal=filtered, detrend=True)
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

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_rate = ts[rate_idx]

    # plot
    if show:
        plotting.plot_resp(ts=ts,
                           raw=signal,
                           filtered=filtered,
                           zeros=zeros,
                           resp_rate_ts=ts_rate,
                           resp_rate=rate,
                           path=None,
                           show=True)

    # output
    args = (ts, filtered, zeros, ts_rate, rate)
    names = ('ts', 'filtered', 'zeros', 'resp_rate_ts', 'resp_rate')

    return utils.ReturnTuple(args, names)
