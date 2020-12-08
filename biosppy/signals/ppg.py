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


def find_onsets_kavsaoglu2016(
    signal=None,
    sampling_rate=1000.0,
    alpha=0.2,
    k=4,
    init_bpm=90,
    min_delay=0.6,
    max_BPM=150,
):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    alpha : float, optional
        Low-pass filter factor.
        Avoids abrupt changes of BPM.
    k : int, float, optional
        Number of segments by pulse.
        Width of each segment = Period of pulse according to current BPM / k
    init_bpm : int, float, optional
        Initial BPM.
        Higher value results in a smaller segment width.
    min_delay : float
        Minimum delay between peaks as percentage of current BPM pulse period.
        Avoids false positives
    max_bpm : int, float, optional
        Maximum BPM.
        Maximum value accepted as valid BPM.

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    window_marks : array
        Indices of segments window boundaries.
    params : dict
        Input parameters of the function


    References
    ----------
    - Kavsaoğlu, Ahmet & Polat, Kemal & Bozkurt, Mehmet. (2016). An innovative peak detection algorithm for
    photoplethysmography signals: An adaptive segmentation method. TURKISH JOURNAL OF ELECTRICAL ENGINEERING
    & COMPUTER SCIENCES. 24. 1782-1796. 10.3906/elk-1310-177.

    Notes
    ---------------------
    This algorithm is an adaption of the one described on Kavsaoğlu et al. (2016).
    This version takes into account a minimum delay between peaks and builds upon the adaptive segmentation
    by using a low-pass filter for BPM changes. This way, even if the algorithm wrongly detects a peak, the
    BPM value will stay relatively constant so the next pulse can be correctly segmented.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if alpha <= 0 or alpha > 1:
        raise TypeError("The value of alpha must be in the range: ]0, 1].")

    if k <= 0:
        raise TypeError("The number of divisions by pulse should be greater than 0.")

    if init_bpm <= 0:
        raise TypeError("Provide a valid BPM value for initial estimation.")

    if min_delay < 0 or min_delay > 1:
        raise TypeError(
            "The minimum delay percentage between peaks must be between 0 and 1"
        )

    if max_BPM >= 248:
        raise TypeError("The maximum BPM must assure the person is alive")

    # current bpm
    bpm = init_bpm

    # current segment window width
    window = int(sampling_rate * (60 / bpm) / k)

    # onsets array
    onsets = []

    # window marks array - stores the boundaries of each segment
    window_marks = []

    # buffer for peak indices
    idx_buffer = [-1, -1, -1]

    # buffer to store the previous 3 values for onset detection
    min_buffer = [0, 0, 0]

    # signal pointer
    i = 0
    while i + window < len(signal):
        # remove oldest values
        idx_buffer.pop(0)
        min_buffer.pop(0)

        # add the index of the minimum value of the current segment to buffer
        idx_buffer.append(int(i + np.argmin(signal[i : i + window])))

        # add the minimum value of the current segment to buffer
        min_buffer.append(signal[idx_buffer[-1]])

        if (
            # the buffer has to be filled with valid values
            idx_buffer[0] > -1
            # the center value of the buffer must be smaller than its neighbours
            and (min_buffer[1] < min_buffer[0] and min_buffer[1] <= min_buffer[2])
            # if an onset was previously detected, guarantee that the new onset respects the minimum delay, minimum BPM and maximum BPM
            and (
                len(onsets) == 0
                or (
                    (idx_buffer[1] - onsets[-1]) / sampling_rate >= min_delay * 60 / bpm
                    and (idx_buffer[1] - onsets[-1]) / sampling_rate > 60 / max_BPM
                )
            )
        ):
            # store the onset
            onsets.append(idx_buffer[1])

            # if more than one onset was detected, update the bpm and the segment width
            if len(onsets) > 1:
                # calculate new bpm from the latest two onsets
                new_bpm = int(60 * sampling_rate / (onsets[-1] - onsets[-2]))

                # update the bpm value
                bpm = alpha * new_bpm + (1 - alpha) * bpm

                # update the segment window width
                window = int(sampling_rate * (60 / bpm) / k)

        # update the signal pointer
        i += window

        # store window segment boundaries index
        window_marks.append(i)

    onsets = np.array(onsets, dtype="int")
    window_marks = np.array(window_marks, dtype="int")

    # output
    params = {
        "signal": signal,
        "sampling_rate": sampling_rate,
        "alpha": alpha,
        "k": k,
        "init_bpm": init_bpm,
        "min_delay": min_delay,
        "max_bpm": max_BPM,
    }

    args = (onsets, window_marks, params)
    names = ("onsets", "window_marks", "params")

    return utils.ReturnTuple(args, names)
