# -*- coding: utf-8 -*-
"""
biosppy.signals.emg
-------------------

This module provides methods to process Electromyographic (EMG) signals.

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


def emg(signal=None, sampling_rate=1000., show=True):
    """Process a raw EMG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered EMG signal.
    onsets : array
        Indices of EMG pulse onsets.

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
                                      band='highpass',
                                      order=4,
                                      frequency=100,
                                      sampling_rate=sampling_rate)

    # find onsets
    onsets, = find_onsets(signal=filtered, sampling_rate=sampling_rate)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_emg(ts=ts,
                          sampling_rate=1000.,
                          raw=signal,
                          filtered=filtered,
                          processed=None,
                          onsets=onsets,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, onsets)
    names = ('ts', 'filtered', 'onsets')

    return utils.ReturnTuple(args, names)


def find_onsets(signal=None, sampling_rate=1000., size=0.05, threshold=None):
    """Determine onsets of EMG pulses.

    Skips corrupted signal parts.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Detection window size (seconds).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # full-wave rectification
    fwlo = np.abs(signal)

    # smooth
    size = int(sampling_rate * size)
    mvgav, _ = st.smoother(signal=fwlo,
                           kernel='boxzen',
                           size=size,
                           mirror=True)

    # threshold
    if threshold is None:
        aux = np.abs(mvgav)
        threshold = 1.2 * np.mean(aux) + 2.0 * np.std(aux, ddof=1)

    # find onsets
    length = len(signal)
    start = np.nonzero(mvgav > threshold)[0]
    stop = np.nonzero(mvgav <= threshold)[0]

    onsets = np.union1d(np.intersect1d(start - 1, stop),
                        np.intersect1d(start + 1, stop))

    if np.any(onsets):
        if onsets[-1] >= length:
            onsets[-1] = length - 1

    return utils.ReturnTuple((onsets,), ('onsets',))


def hodges_bui_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                              size=None, threshold=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Hodges and Bui [HoBu96]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : int
        Detection window size (seconds).
    threshold : int, float
        Detection threshold.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [HoBu96] Hodges PW, Bui BH, "A comparison of computer-based methods for
       the determination of onset of muscle contraction using
       electromyography", Electroencephalography and Clinical Neurophysiology 
       - Electromyography and Motor Control, vol. 101:6, pp. 511-519, 1996

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if size is None:
        raise TypeError("Please specify the detection window size.")

    if threshold is None:
        raise TypeError("Please specify the detection threshold.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            mean_rest = statistics['mean']
            std_dev_rest = statistics['std_dev']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        mean_rest = rest['mean']
        std_dev_rest = rest['std_dev']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # moving average
    mvgav = np.convolve(fwlo, np.ones((size,))/size, mode='valid')

    # calculate the test function
    tf = (1 / std_dev_rest) * (mvgav - mean_rest)

    # find onsets
    length = len(signal)
    start = np.nonzero(tf >= threshold)[0]
    stop = np.nonzero(tf < threshold)[0]

    onsets = np.union1d(np.intersect1d(start - 1, stop),
                        np.intersect1d(start + 1, stop))

    # adjust indices because of moving average
    onsets += int(size / 2)

    if np.any(onsets):
        if onsets[-1] >= length:
            onsets[-1] = length - 1

    return utils.ReturnTuple((onsets, tf), ('onsets', 'processed'))


def bonato_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                          threshold=None, active_state_duration=None,
                          samples_above_fail=None, fail_size=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Bonato et al. [Bo98]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : int, float
        Detection threshold.
    active_state_duration: int
        Minimum duration of the active state.
    samples_above_fail : int
        Number of samples above the threshold level in a group of successive
        samples.
    fail_size : int
        Number of successive samples.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Bo98] Bonato P, D’Alessio T, Knaflitz M, "A statistical method for the
       measurement of muscle activation intervals from surface myoelectric
       signal during gait", IEEE Transactions on Biomedical Engineering,
       vol. 45:3, pp. 287–299, 1998

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if threshold is None:
        raise TypeError("Please specify the detection threshold.")

    if active_state_duration is None:
        raise TypeError("Please specify the mininum duration of the "
                        "active state.")

    if samples_above_fail is None:
        raise TypeError("Please specify the number of samples above the "
                        "threshold level in a group of successive samples.")

    if fail_size is None:
        raise TypeError("Please specify the number of successive samples.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            var_rest = statistics['var']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        var_rest = rest['var']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    tf_list = []
    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    state_duration = 0
    j = 0
    n = 0
    onset = False
    alarm = False
    for k in range(1, len(signal_zero_mean), 2):  # odd values only
        # calculate the test function
        tf = (1 / var_rest) * (signal_zero_mean[k-1]**2 + signal_zero_mean[k]**2)
        tf_list.append(tf)
        if onset is True:
            if alarm is False:
                if tf < threshold:
                    alarm_time = k // 2
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of inactive state
                if tf < threshold:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples above the threshold level but now one is bellow it
                        # the test function may go above the threshold , but each time not longer than j samples
                        n += 1
                        if n == samples_above_fail:
                            n = 0
                            j = 0
                    if state_duration == active_state_duration:
                        offset_time_list.append(alarm_time)
                        onset = False
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the inactive state is above the threshold for longer than the predefined number of samples
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:  # if the alarm time has not yet been identified
                if tf >= threshold:  # alarm time
                    alarm_time = k // 2
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of active state
                if tf >= threshold:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples below the threshold level but now one is above it.
                        # a total of n samples must be above it
                        n += 1
                        if n == samples_above_fail:
                            n = 0
                            j = 0
                    if state_duration == active_state_duration:
                        onset_time_list.append(alarm_time)
                        onset = True
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the active state has fallen below the threshold for longer than the predefined number of samples
                        alarm = False
                        n = 0
                        j = 0
                        state_duration = 0

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    # adjust indices because of odd numbers
    onsets *= 2

    return utils.ReturnTuple((onsets, tf_list), ('onsets', 'processed'))


def lidierth_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                            size=None, threshold=None,
                            active_state_duration=None, fail_size=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Lidierth. [Li86]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : int
        Detection window size (seconds).
    threshold : int, float
        Detection threshold.
    active_state_duration: int
        Minimum duration of the active state.
    fail_size : int
        Number of successive samples.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Li86] Lidierth M, "A computer based method for automated measurement
       of the periods of muscular activity from an EMG and its application to
       locomotor EMGs", ElectroencephClin Neurophysiol, vol. 64:4,
       pp. 378–380, 1986

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if size is None:
        raise TypeError("Please specify the detection window size.")

    if threshold is None:
        raise TypeError("Please specify the detection threshold.")

    if active_state_duration is None:
        raise TypeError("Please specify the mininum duration of the "
                        "active state.")

    if fail_size is None:
        raise TypeError("Please specify the number of successive samples.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            mean_rest = statistics['mean']
            std_dev_rest = statistics['std_dev']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        mean_rest = rest['mean']
        std_dev_rest = rest['std_dev']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # moving average
    mvgav = np.convolve(fwlo, np.ones((size,)) / size, mode='valid')

    # calculate the test function
    tf = (1 / std_dev_rest) * (mvgav - mean_rest)

    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    state_duration = 0
    j = 0
    onset = False
    alarm = False
    for k in range(0, len(tf)):
        if onset is True:
            # an onset was previously detected and we are looking for the offset time applying the same criteria
            if alarm is False:  # if the alarm time has not yet been identified
                if tf[k] < threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of inactive state
                if tf[k] < threshold:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples above the threshold level but now one is bellow it
                        # the test function may go above the threshold , but each time not longer than j samples
                        j = 0
                    if state_duration == active_state_duration:
                        offset_time_list.append(alarm_time)
                        onset = False
                        alarm = False
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the inactive state is above the threshold for longer than the predefined number of samples
                        alarm = False
                        j = 0
                        state_duration = 0
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:  # if the alarm time has not yet been identified
                if tf[k] >= threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of active state
                if tf[k] >= threshold:
                    state_duration += 1
                    if j > 0:  # there was one (or more) samples below the threshold level but now one is above it
                        # the test function may repeatedly fall below the threshold, but each time not longer than j samples
                        j = 0
                    if state_duration == active_state_duration:
                        onset_time_list.append(alarm_time)
                        onset = True
                        alarm = False
                        j = 0
                        state_duration = 0
                else:  # sample falls below the threshold level
                    j += 1
                    if j > fail_size:
                        # the active state has fallen below the threshold for longer than the predefined number of samples
                        alarm = False
                        j = 0
                        state_duration = 0

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    # adjust indices because of moving average
    onsets += int(size / 2)

    return utils.ReturnTuple((onsets, tf), ('onsets', 'processed'))


def abbink_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                          size=None, alarm_size=None, threshold=None,
                          transition_threshold=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Abbink et al.. [Abb98]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : int
        Detection window size (seconds).
    alarm_size : int
        Number of amplitudes searched in the calculation of the transition
        index.
    threshold : int, float
        Detection threshold.
    transition_threshold: int, float
        Threshold used in the calculation of the transition index.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Abb98] Abbink JH, van der Bilt A, van der Glas HW, "Detection of onset
       and termination of muscle activity in surface electromyograms",
       Journal of Oral Rehabilitation, vol. 25, pp. 365–369, 1998

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if size is None:
        raise TypeError("Please specify the detection window size.")

    if alarm_size is None:
        raise TypeError("Please specify the number of amplitudes searched in "
                        "the calculation of the transition index.")

    if threshold is None:
        raise TypeError("Please specify the detection threshold.")

    if transition_threshold is None:
        raise TypeError("Please specify the second threshold.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            mean_rest = statistics['mean']
            std_dev_rest = statistics['std_dev']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        mean_rest = rest['mean']
        std_dev_rest = rest['std_dev']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # moving average
    mvgav = np.convolve(fwlo, np.ones((size,)) / size, mode='valid')

    # calculate the test function
    tf = (1 / std_dev_rest) * (mvgav - mean_rest)

    # additional filter
    filtered_tf, _, _ = st.filter_signal(signal=tf,
                                         ftype='butter',
                                         band='lowpass',
                                         order=10,
                                         frequency=30,
                                         sampling_rate=sampling_rate)
    # convert from numpy array to list to use list comprehensions
    filtered_tf = filtered_tf.tolist()

    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    onset = False
    alarm = False
    for k in range(0, len(tf)):
        if onset is True:
            # an onset was previously detected and we are looking for the offset time, applying the same criteria
            if alarm is False:
                if filtered_tf[k] < threshold:
                    # the first index of the sliding window is used as an estimate for the onset time (simple post-processor)
                    alarm_time = k
                    alarm = True
            else:
                # if alarm_time > alarm_window_size and len(emg_conditioned_list) == (alarm_time + alarm_window_size + 1):
                if alarm_time > alarm_size and k == (alarm_time + alarm_size + 1):
                    transition_indices = []
                    for j in range(alarm_size, alarm_time):
                        low_list = [filtered_tf[j-alarm_size+a] for a in range(1, alarm_size+1)]
                        low = sum(i < transition_threshold for i in low_list)
                        high_list = [filtered_tf[j+b] for b in range(1, alarm_size+1)]
                        high = sum(i > transition_threshold for i in high_list)
                        transition_indices.append(low + high)
                    offset_time_list = np.where(transition_indices == np.amin(transition_indices))[0].tolist()
                    onset = False
                    alarm = False
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:
                if filtered_tf[k] >= threshold:
                    # the first index of the sliding window is used as an estimate for the onset time (simple post-processor)
                    alarm_time = k
                    alarm = True
            else:
                # if alarm_time > alarm_window_size and len(emg_conditioned_list) == (alarm_time + alarm_window_size + 1):
                if alarm_time > alarm_size and k == (alarm_time + alarm_size + 1):
                    transition_indices = []
                    for j in range(alarm_size, alarm_time):
                        low_list = [filtered_tf[j-alarm_size+a] for a in range(1, alarm_size+1)]
                        low = sum(i < transition_threshold for i in low_list)
                        high_list = [filtered_tf[j+b] for b in range(1, alarm_size+1)]
                        high = sum(i > transition_threshold for i in high_list)
                        transition_indices.append(low + high)
                    onset_time_list = np.where(transition_indices == np.amax(transition_indices))[0].tolist()
                    onset = True
                    alarm = False

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    # adjust indices because of moving average
    onsets += int(size / 2)

    return utils.ReturnTuple((onsets, filtered_tf), ('onsets', 'processed'))


def solnik_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                          threshold=None, active_state_duration=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Solnik et al. [Sol10]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : int, float
        Scale factor for calculating the detection threshold.
    active_state_duration: int
        Minimum duration of the active state.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Sol10] Solnik S, Rider P, Steinweg K, DeVita P, Hortobágyi T,
       "Teager-Kaiser energy operator signal conditioning improves EMG onset
       detection", European Journal of Applied Physiology, vol 110:3,
       pp. 489-498, 2010

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if threshold is None:
        raise TypeError("Please specify the scale factor for calculating the "
                        "detection threshold.")

    if active_state_duration is None:
        raise TypeError("Please specify the mininum duration of the "
                        "active state.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            mean_rest = statistics['mean']
            std_dev_rest = statistics['std_dev']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        mean_rest = rest['mean']
        std_dev_rest = rest['std_dev']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # calculate threshold
    threshold = mean_rest + threshold * std_dev_rest

    tf_list = []
    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    state_duration = 0
    onset = False
    alarm = False
    for k in range(1, len(signal_zero_mean)-1):
        # calculate the test function
        # Teager-Kaiser energy operator
        tf = signal_zero_mean[k]**2 - signal_zero_mean[k+1] * signal_zero_mean[k-1]
        # full-wave rectification
        tf = np.abs(tf)
        tf_list.append(tf)
        if onset is True:
            # an onset was previously detected and we are looking for the offset time, applying the same criteria
            if alarm is False:  # if the alarm time has not yet been identified
                if tf < threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of inactive state
                if tf < threshold:
                    state_duration += 1
                    if state_duration == active_state_duration:
                        offset_time_list.append(alarm_time)
                        onset = False
                        alarm = False
                        state_duration = 0
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:  # if the alarm time has not yet been identified
                if tf >= threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of active state
                if tf >= threshold:
                    state_duration += 1
                    if state_duration == active_state_duration:
                        onset_time_list.append(alarm_time)
                        onset = True
                        alarm = False
                        state_duration = 0

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    return utils.ReturnTuple((onsets, tf_list), ('onsets', 'processed'))


def silva_onset_detector(signal=None, sampling_rate=1000.,
                         size=None, threshold_size=None, threshold=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Silva et al. [Sil12]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : int
        Detection window size (seconds).
    threshold_size : int
        Window size for calculation of the adaptive threshold; must be bigger
        than the detection window size.
    threshold : int, float
        Fixed threshold for the double criteria.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Sil12] Silva H, Scherer R, Sousa J, Londral A , "Towards improving the
       usability of electromyographic interfacess", Journal of Oral
       Rehabilitation, pp. 1–2, 2012

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if size is None:
        raise TypeError("Please specify the detection window size.")

    if threshold_size is None:
        raise TypeError("Please specify the window size for calculation of "
                        "the adaptive threshold.")

    if threshold_size <= size:
        raise TypeError("The window size for calculation of the adaptive "
                        "threshold must be bigger than the detection "
                        "window size")

    if threshold is None:
        raise TypeError("Please specify the fixed threshold for the "
                        "double criteria.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # full-wave rectification
    fwlo = np.abs(signal_zero_mean)

    # moving average for calculating the test function
    tf_mvgav = np.convolve(fwlo, np.ones((size,)) / size, mode='valid')

    # moving average for calculating the adaptive threshold
    threshold_mvgav = np.convolve(fwlo, np.ones((threshold_size,)) / threshold_size, mode='valid')

    onset_time_list = []
    offset_time_list = []
    onset = False
    for k in range(0, len(threshold_mvgav)):
        if onset is True:
            # an onset was previously detected and we are looking for the offset time, applying the same criteria
            if tf_mvgav[k] < threshold_mvgav[k] and tf_mvgav[k] < threshold:
                offset_time_list.append(k)
                onset = False  # the offset has been detected, and we can look for another activation
        else:  # we only look for another onset if a previous offset was detected
            if tf_mvgav[k] >= threshold_mvgav[k] and tf_mvgav[k] >= threshold:
                # the first index of the sliding window is used as an estimate for the onset time (simple post-processor)
                onset_time_list.append(k)
                onset = True

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    # adjust indices because of moving average
    onsets += int(size / 2)

    return utils.ReturnTuple((onsets, tf_mvgav), ('onsets', 'processed'))


def londral_onset_detector(signal=None, rest=None, sampling_rate=1000.,
                           size=None, threshold=None,
                           active_state_duration=None):
    """Determine onsets of EMG pulses.

    Follows the approach by Londral et al. [Lon13]_.

    Parameters
    ----------
    signal : array
        Input filtered EMG signal.
    rest : array, list, dict
        One of the following 3 options:
        * N-dimensional array with filtered samples corresponding to a
        rest period;
        * 2D array or list with the beginning and end indices of a segment of
        the signal corresponding to a rest period;
        * Dictionary with {'mean': mean value, 'std_dev': standard variation}.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : int
        Detection window size (seconds).
    threshold : int, float
        Scale factor for calculating the detection threshold.
    active_state_duration: int
        Minimum duration of the active state.

    Returns
    -------
    onsets : array
        Indices of EMG pulse onsets.
    processed : array
        Processed EMG signal.

    References
    ----------
    .. [Lon13] Londral A, Silva H, Nunes N, Carvalho M, Azevedo L, "A wireless
       user-computer interface to explore various sources of biosignals and
       visual biofeedback for severe motor impairment",
       Journal of Accessibility and Design for All, vol. 3:2, pp. 118–134, 2013
    
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rest is None:
        raise TypeError("Please specidy rest parameters.")

    if size is None:
        raise TypeError("Please specify the detection window size.")

    if threshold is None:
        raise TypeError("Please specify the scale factor for calculating the "
                        "detection threshold.")

    if active_state_duration is None:
        raise TypeError("Please specify the mininum duration of the "
                        "active state.")

    # gather statistics on rest signal
    if isinstance(rest, np.ndarray) or isinstance(rest, list):
        # if the input parameter is a numpy array or a list
        if len(rest) >= 2:
            # first ensure numpy
            rest = np.array(rest)
            if len(rest) == 2:
                # the rest signal is a segment of the signal
                rest_signal = signal[rest[0]:rest[1]]
            else:
                # the rest signal is provided as is
                rest_signal = rest
            rest_zero_mean = rest_signal - np.mean(rest_signal)
            statistics = st.signal_stats(signal=rest_zero_mean)
            mean_rest = statistics['mean']
            std_dev_rest = statistics['std_dev']
        else:
            raise TypeError("Please specify the rest analysis.")
    elif isinstance(rest, dict):
        # if the input is a dictionary
        mean_rest = rest['mean']
        std_dev_rest = rest['std_dev']
    else:
        raise TypeError("Please specify the rest analysis.")

    # subtract baseline offset
    signal_zero_mean = signal - np.mean(signal)

    # calculate threshold
    threshold = mean_rest + threshold * std_dev_rest

    # helper function for calculating the test function for each window
    def _londral_test_function(signal=None):
        tf = (1 / size) * (sum(j ** 2 for j in signal) - (1 / size) * (sum(signal) ** 2))
        return tf

    # calculate the test function
    _, tf = st.windower(
        signal=signal_zero_mean,
        size=size, step=1,
        fcn=_londral_test_function,
        kernel='rectangular',
    )

    onset_time_list = []
    offset_time_list = []
    alarm_time = 0
    state_duration = 0
    onset = False
    alarm = False
    for k in range(0, len(tf)):
        if onset is True:
            # an onset was previously detected and we are looking for the offset time, applying the same criteria
            if alarm is False:  # if the alarm time has not yet been identified
                if tf[k] < threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of inactive state
                if tf[k] < threshold:
                    state_duration += 1
                    if state_duration == active_state_duration:
                        offset_time_list.append(alarm_time)
                        onset = False
                        alarm = False
                        state_duration = 0
        else:  # we only look for another onset if a previous offset was detected
            if alarm is False:  # if the alarm time has not yet been identified
                if tf[k] >= threshold:  # alarm time
                    alarm_time = k
                    alarm = True
            else:  # now we have to check for the remaining rule to me bet - duration of active state
                if tf[k] >= threshold:
                    state_duration += 1
                    if state_duration == active_state_duration:
                        onset_time_list.append(alarm_time)
                        onset = True
                        alarm = False
                        state_duration = 0

    onsets = np.union1d(onset_time_list,
                        offset_time_list)

    # adjust indices because of moving average
    onsets += int(size / 2)

    return utils.ReturnTuple((onsets, tf), ('onsets', 'processed'))
