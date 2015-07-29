# -*- coding: utf-8 -*-
"""
    biosppy.signals.emg
    -------------------
    
    This module provides methods to process Electromyographic (EMG) signals.
    
    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in

# 3rd party
import numpy as np

# local
from . import tools as st
from .. import plotting, utils

# Globals


def emg(signal=None, sampling_rate=1000., show=True):
    """Process a raw EMG signal and extract relevant signal features using default parameters.
    
    Args:
        signal (array): Raw EMG signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
        
        show (bool): If True, show a summary plot (optional).
    
    Returns:
        ts (array): Signal time axis reference (seconds).
        
        filtered (array): Filtered EMG signal.
        
        onsets (array): Indices of EMG pulse onsets.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # ensure numpy
    signal = np.array(signal)
    
    sampling_rate = float(sampling_rate)
    
    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal, ftype='butter', band='highpass',
                                      order=4, frequency=100, sampling_rate=sampling_rate)
    
    # find onsets
    onsets, = find_onsets(signal=filtered, sampling_rate=sampling_rate)
    
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    
    # plot
    if show:
        plotting.plot_emg(ts=ts, raw=signal, filtered=filtered, onsets=onsets,
                          path=None, show=True)
    
    # output
    args = (ts, filtered, onsets)
    names = ('ts', 'filtered', 'onsets')
    
    return utils.ReturnTuple(args, names)


def find_onsets(signal=None, sampling_rate=1000., size=0.05, threshold=None):
    """Determine onsets of EMG pulses.
    
    Skips corrupted signal parts.
    
    Args:
        signal (array): Input filtered BVP signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
        
        size (float): Detection window size (seconds) (optional).
        
        threshold (float): Detection threshold (optional).
    
    Returns:
        onsets (array): Indices of BVP pulse onsets.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # full-wave rectification
    fwlo = np.abs(signal)
    
    # smooth
    size = int(sampling_rate * size)
    mvgav, _ = st.smoother(signal=fwlo, kernel='boxzen', size=size, mirror=True)
    
    # threshold
    if threshold is None:
        aux = np.abs(mvgav)
        threshold = 1.2 * np.mean(aux) + 2.0 * np.std(aux, ddof=1)
    
    # find onsets
    length = len(signal)
    start = np.nonzero(mvgav > threshold)[0]
    stop = np.nonzero(mvgav <= threshold)[0]
    
    onsets = np.union1d(np.intersect1d(start - 1, stop), np.intersect1d(start + 1, stop))
    
    if np.any(onsets):
        if onsets[-1] >= length:
            onsets[-1] = length - 1
    
    return utils.ReturnTuple((onsets, ), ('onsets', ))

