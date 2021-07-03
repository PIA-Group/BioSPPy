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


def acc(signal=None, sampling_rate=100., path=None, show=True):
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

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    signal : array
        Raw (unfiltered) ACC signal.
    vm : array
        Vector Magnitude feature of the signal
    sm : array
        Signal Magnitude feature of the signal

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # extract features
    vm_features, sm_features = extract_acc_features(signal=signal,
                                                    sampling_rate=sampling_rate)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_acc(ts=ts,
                          raw=signal,
                          vm=vm_features,
                          sm=sm_features,
                          path=path,
                          show=True)

    # output
    args = (ts, signal, vm_features, sm_features)
    names = ('ts', 'signal', 'vm', 'sm')

    return utils.ReturnTuple(args, names)


def extract_acc_features(signal=None, sampling_rate=100.0):
    """Extracts the vector magnitude and signal magnitude features from an input ACC signal, given the signal itself.

    Parameters
    ----------
    signal : array
        Input ACC signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

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
        vm_features[i] = np.linalg.norm(np.array([signal[i][0], signal[i][1], signal[i][2]]))
        sm_features[i] = (abs(signal[i][0]) + abs(signal[i][1]) + abs(signal[i][2])) / 3

    return utils.ReturnTuple((vm_features, sm_features), ('vm', 'sm'))

