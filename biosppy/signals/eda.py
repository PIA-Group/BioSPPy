# -*- coding: utf-8 -*-
"""
    biosppy.signals.eda
    -------------------
    
    This module provides methods to process Electrodermal Activity (EDA) signals.
    
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


def eda(signal=None, sampling_rate=1000., show=True):
    """Process a raw EDA signal and extract relevant signal features using default parameters.
    
    Args:
        signal (array): Raw EDA signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
        
        show (bool): If True, show a summary plot (optional).
    
    Returns:
        ts (array): Signal time axis reference (seconds).
        
        filtered (array): Filtered EDA signal.
        
        onsets (array): Indices of SCR pulse onsets.
        
        peaks (array): Indices of the SCR peaks.
        
        amplitudes (array): SCR pulse amplitudes.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # ensure numpy
    signal = np.array(signal)
    
    sampling_rate = float(sampling_rate)
    
    # filter signal
    aux, _, _ = st.filter_signal(signal=signal, ftype='butter', band='lowpass',
                                 order=4, frequency=20, sampling_rate=sampling_rate)
    
    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux, kernel='boxzen', size=sm_size, mirror=True)
    
    # get SCR info
    onsets, peaks, amps = kbk_scr(signal=filtered, sampling_rate=sampling_rate)
    
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    
    # plot
    if show:
        plotting.plot_eda(ts=ts, raw=signal, filtered=filtered, onsets=onsets,
                          peaks=peaks, amplitudes=amps, path=None, show=True)
    
    # output
    args = (ts, filtered, onsets, peaks, amps)
    names = ('ts', 'filtered', 'onsets', 'peaks', 'amplitudes')
    
    return utils.ReturnTuple(args, names)


def basic_scr(signal=None, sampling_rate=1000.):
    """Basic method to extract Skin Conductivity Responses (SCR) from an EDA signal.
    
    Args:
        signal (array): Input filterd EDA signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
    
    Returns:
        onsets (array): Indices of the SCR onsets.
        
        peaks (array): Indices of the SRC peaks.
        
        amplitudes (array): SCR pulse amplitudes.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # find extrema
    pi, _ = st.find_extrema(signal=signal, mode='max')
    ni, _ = st.find_extrema(signal=signal, mode='min')
    
    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError("Could not find SCR pulses.")
    
    # pair vectors
    if ni[0] < pi[0]:
        ni = ni[1:]
    if pi[-1] > ni[-1]:
        pi = pi[:-1]
    if len(pi) > len(ni):
        pi = pi[:-1]
    
    li = min(len(pi), len(ni))
    i1 = pi[:li]
    i3 = ni[:li]
    
    # indices
    i0 = i1 - (i3 - i1) / 2.
    if i0[0] < 0:
        i0[0] = 0
    
    # amplitude
    a = np.array(map(lambda i: np.max(signal[i1[i]:i3[i]]), range(li)))
    
    # output
    args = (i3, i1, a)
    names = ('onsets', 'peaks', 'amplitudes')
    
    return utils.ReturnTuple(args, names)


def kbk_scr(signal=None, sampling_rate=1000.):
    """KBK method to extract Skin Conductivity Responses (SCR) from an EDA signal.
    
    Args:
        signal (array): Input filterd EDA signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
    
    Returns:
        onsets (array): Indices of the SCR onsets.
        
        peaks (array): Indices of the SRC peaks.
        
        amplitudes (array): SCR pulse amplitudes.
    
    References:
        [1] K.H. Kim, S.W. Bang, and S.R. Kim, "Emotion recognition system
            using short-term monitoring of physiological signals",
            Med. Biol. Eng. Comput., 2004, 42, 419-427
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # differentiation
    df = np.diff(signal)
    
    # smooth
    size = int(1. * sampling_rate)
    df, _ = st.smoother(signal=df, kernel='bartlett', size=size, mirror=True)
    
    # zero crosses
    zeros, = st.zero_cross(signal=df, detrend=True)
    if np.all(df[:zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1]:] > 0):
        zeros = zeros[:-1]
    
    # exclude SCRs with small amplitude
    thr = 0.1 * np.max(df)
    
    scrs, amps, ZC, pks = [], [], [], []
    for i in xrange(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i]:zeros[i+1]]]
        aux = scrs[-1].max()
        if aux > thr:
            amps += [aux]
            ZC += [zeros[i]]
            ZC += [zeros[i+1]]
            pks += [zeros[i] + np.argmax(df[zeros[i]:zeros[i+1]])]
    
    scrs, amps, ZC, pks = np.array(scrs), np.array(amps), np.array(ZC), np.array(pks)
    onsets = ZC[::2]
    
    # output
    args = (onsets, pks, amps)
    names = ('onsets', 'peaks', 'amplitudes')
    
    return utils.ReturnTuple(args, names)

