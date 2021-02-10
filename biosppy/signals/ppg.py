# -*- coding: utf-8 -*-
"""
biosppy.signals.ppg
-------------------

This module provides methods to process Photoplethysmogram (PPG) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np
import scipy.signal as ss

# local
from . import tools as st
from .. import plotting, utils


def ppg(signal=None, sampling_rate=1000., show=True):
    """Process a raw PPG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered PPG signal.
    onsets : array
        Indices of PPG pulse onsets.
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
    onsets, _ = find_onsets_elgendi2013(signal=filtered, sampling_rate=sampling_rate)

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
        plotting.plot_ppg(ts=ts,
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

def find_onsets_elgendi2013(signal=None, sampling_rate=1000., peakwindow=0.111, beatwindow=0.667, beatoffset=0.02, mindelay=0.3):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peakwindow : float
        Parameter W1 on referenced article
        Optimized at 0.111
    beatwindow : float
        Parameter W2 on referenced article
        Optimized at 0.667
    beatoffset : float
        Parameter beta on referenced article
        Optimized at 0.2
    mindelay : float
        Minimum delay between peaks.
        Avoids false positives

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    params : dict
        Input parameters of the function


    References
    ----------
    - Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions.
    PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585.
    
    Notes
    ---------------------
    Optimal ranges for signal filtering (from Elgendi et al. 2013):
    "Optimization of the beat detector’s spectral window for the lower frequency resulted in a 
    value within 0.5– 1 Hz with the higher frequency within 7–15 Hz"
    
    All the number references below between curly brackets {...} by the code refer to the line numbers of
    code in "Table 2 Algorithm IV: DETECTOR (PPG signal, F1, F2, W1, W2, b)" from Elgendi et al. 2013 for a
    better comparison of the algorithm
    
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # Create copy of signal (not to modify the original object)
    signal_copy = np.copy(signal)

    # Truncate to zero and square
    # {3, 4}
    signal_copy[signal_copy < 0] = 0
    squared_signal = signal_copy ** 2

    # Calculate peak detection threshold
    # {5}
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_peak_kernel)

    # {6}
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_beat_kernel)

    # Calculate threshold value
    # {7, 8, 9}
    thr1 = ma_beat + beatoffset * np.mean(squared_signal)

    # Identify start and end of PPG waves.
    # {10-16}
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    # {18}
    min_len = int(np.rint(peakwindow * sampling_rate))
    min_delay = int(np.rint(mindelay * sampling_rate))
    onsets = [0]

    # {19}
    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        # {20, 22, 23}
        if len_wave < min_len:
            continue

        # Find local maxima and their prominence within wave span.
        # {21}
        data = signal_copy[beg:end]
        locmax, props = ss.find_peaks(data, prominence=(None, None))

        # If more than one peak
        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between onsets.
            if peak - onsets[-1] > min_delay:
                onsets.append(peak)

    onsets.pop(0)
    onsets = np.array(onsets, dtype='int')

    # output
    params = {'signal': signal, 'sampling_rate': sampling_rate, 'peakwindow': peakwindow, 'beatwindow': beatwindow, 'beatoffset': beatoffset, 'mindelay': mindelay}

    args = (onsets, params)
    names = ('onsets', 'params')

    return utils.ReturnTuple(args, names)
