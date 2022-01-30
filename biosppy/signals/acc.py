# -*- coding: utf-8 -*-
"""
biosppy.signals.acc
-------------------

This module provides methods to process Acceleration (ACC) signals.
Implemented code assumes ACC acquisition from a 3 orthogonal axis reference system.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

Authors
-------
Afonso Ferreira
Diogo Vieira

"""

# Imports
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np

# local
from .. import plotting, utils
from biosppy.inter_plotting import acc as inter_plotting


def acc(signal=None, sampling_rate=100.0, path=None, show=True, interactive=True):
    """Process a raw ACC signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.
    interactive : bool, optional
        If True, shows an interactive plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    signal : array
        Raw (unfiltered) ACC signal.
    vm : array
        Vector Magnitude feature of the signal.
    sm : array
        Signal Magnitude feature of the signal.
    freq_features : dict
        Positive Frequency domains (Hz) of the signal.
    amp_features : dict
        Normalized Absolute Amplitudes of the signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # extract features
    vm_features, sm_features = time_domain_feature_extractor(signal=signal)
    freq_features, abs_amp_features = frequency_domain_feature_extractor(
        signal=signal, sampling_rate=sampling_rate
    )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        if interactive:
            inter_plotting.plot_acc(
                ts=ts,  # plotting.plot_acc
                raw=signal,
                vm=vm_features,
                sm=sm_features,
                spectrum={"freq": freq_features, "abs_amp": abs_amp_features},
                path=path,
            )
        else:
            plotting.plot_acc(
                ts=ts,  # plotting.plot_acc
                raw=signal,
                vm=vm_features,
                sm=sm_features,
                path=path,
                show=True,
            )

    # output
    args = (ts, signal, vm_features, sm_features, freq_features, abs_amp_features)
    names = ("ts", "signal", "vm", "sm", "freq", "abs_amp")

    return utils.ReturnTuple(args, names)


def time_domain_feature_extractor(signal=None):
    """Extracts the vector magnitude and signal magnitude features from an input ACC signal, given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.

    Returns
    -------
    vm_features : array
        Extracted Vector Magnitude (VM) feature.
    sm_features : array
        Extracted Signal Magnitude (SM) feature.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # get acceleration features
    vm_features = np.zeros(signal.shape[0])
    sm_features = np.zeros(signal.shape[0])

    for i in range(signal.shape[0]):
        vm_features[i] = np.linalg.norm(
            np.array([signal[i][0], signal[i][1], signal[i][2]])
        )
        sm_features[i] = (abs(signal[i][0]) + abs(signal[i][1]) + abs(signal[i][2])) / 3

    return utils.ReturnTuple((vm_features, sm_features), ("vm", "sm"))


def frequency_domain_feature_extractor(signal=None, sampling_rate=100.0):
    """Extracts the FFT from each ACC sub-signal (x, y, z), given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    freq_features : dict
        Dictionary of positive frequencies (Hz) for all sub-signals.
    amp_features : dict
        Dictionary of Normalized Absolute Amplitudes for all sub-signals.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    freq_features = {}
    amp_features = {}

    # get Normalized FFT for each sub-signal
    for ind, axis in zip(range(signal.shape[1]), ["x", "y", "z"]):
        sub_signal = signal[:, ind]

        n = len(sub_signal)
        k = np.arange(n)
        T = n / sampling_rate
        frq = k / T
        freq_features[axis] = frq[range(n // 2)]

        amp = np.fft.fft(sub_signal) / n
        amp_features[axis] = abs(amp[range(n // 2)])

    return utils.ReturnTuple((freq_features, amp_features), ("freq", "abs_amp"))
