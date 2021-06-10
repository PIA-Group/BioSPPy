# -*- coding: utf-8 -*-
"""
biosppy.signals.eda
-------------------

This module provides methods to process Electrodermal Activity (EDA)
signals, also known as Galvanic Skin Response (GSR).

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
from .. import plotting, utils
import matplotlib.pyplot as plt


def get_filt_eda(signal, sampling_rate=1000., sm_size=0.75):
    """ Filter for the EDA data.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    filtered : list
        Filtered signal.
    """
    sampling_rate = float(sampling_rate)

    # filter signal
    try:
        if sampling_rate > 1:
            signal, _, _ = st.filter_signal(signal=signal,
                                     ftype='butter',
                                     band='lowpass',
                                     order=4,
                                     frequency=1,
                                     sampling_rate=sampling_rate)
    except:
        print("Error filtering EDA")

    # smooth
    try:
        sm_size = int(sm_size * sampling_rate)
        signal, _ = st.smoother(signal=signal,
                                  kernel='boxzen',
                                  size=sm_size,
                                  mirror=True)
    except:
        print("Error smoothing EDA")
        
    return signal


def get_scr(signal, sampling_rate, size=1.):
    """ Returns Electrodermal Response.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    edr : list
        Electrodermal events data.
    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # differentiation
    try:
        signal = np.diff(signal)
        # smooth
        size = int(size * sampling_rate)
        signal, _ = st.smoother(signal=signal, kernel='bartlett', size=size, mirror=True)
    except Exception as e: print(e)
    return signal


def eda(signal=None, sampling_rate=1000., show=True, min_amplitude=0.1):
    """Process a raw EDA signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered EDA signal.
    onsets : array
        Indices of SCR pulse onsets.
    peaks : array
        Indices of the SCR peaks.
    amplitudes : array
        SCR pulse amplitudes.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # Get filtered signal
    filtered = get_filt_eda(signal)

    # get SCR info
    onsets, peaks, amps = kbk_scr(signal=filtered,
                                  sampling_rate=sampling_rate,
                                  min_amplitude=min_amplitude)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_eda(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          onsets=onsets,
                          peaks=peaks,
                          amplitudes=amps,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, onsets, peaks, amps)
    names = ('ts', 'filtered', 'onsets', 'peaks', 'amplitudes')

    return utils.ReturnTuple(args, names)


def basic_scr(signal=None, sampling_rate=1000.):
    """Basic method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.
    Follows the approach in [Gamb08]_.
    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.
    References
    ----------
    .. [Gamb08] Hugo Gamboa, "Multi-modal Behavioral Biometrics Based on HCI
       and Electrophysiology", PhD thesis, Instituto Superior T{\'e}cnico, 2008
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
    a = np.array([np.max(signal[i1[i]:i3[i]]) for i in range(li)])

    # output
    args = (i3, i1, a)
    names = ('onsets', 'peaks', 'amplitudes')

    return utils.ReturnTuple(args, names)


def kbk_scr(signal=None, sampling_rate=1000., min_amplitude=0.1):
    """KBK method to extract Skin Conductivity Responses (SCR) from an
    EDA signal.

    Follows the approach by Kim *et al.* [KiBK04]_.

    Parameters
    ----------
    signal : array
        Input filterd EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.

    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.

    References
    ----------
    .. [KiBK04] K.H. Kim, S.W. Bang, and S.R. Kim, "Emotion recognition
       system using short-term monitoring of physiological signals",
       Med. Biol. Eng. Comput., vol. 42, pp. 419-427, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # Get SCR signal
    df = get_scr(signal, sampling_rate)

    # zero crosses
    zeros, = st.zero_cross(signal=df, detrend=False)
    if np.all(df[:zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1]:] > 0):
        zeros = zeros[:-1]

    # exclude SCRs with small amplitude
    thr = min_amplitude * np.max(df)

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i]:zeros[i + 1]]]
        aux = scrs[-1].max()
        if aux > thr:
            amps += [aux]
            ZC += [zeros[i]]
            ZC += [zeros[i + 1]]
            pks += [zeros[i] + np.argmax(df[zeros[i]:zeros[i + 1]])]

    amps = np.array(amps)
    ZC = np.array(ZC)
    pks = np.array(pks)
    onsets = ZC[::2]

    # output
    args = (onsets, pks, amps)
    names = ('onsets', 'peaks', 'amplitudes')

    return utils.ReturnTuple(args, names)


def get_eda_param(signal, min_amplitude=0.08, filt=False, size=1., sampling_rate= 1000.):
    """ Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    min_amplitude : float, optional
        Minimum treshold by which to exclude SCRs.
    Returns
    -------
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.
    amplitudes : array
        SCR pulse amplitudes.
    end : array
        Indices of the SCR end.
    """
    
    signal = np.array(signal).astype(np.float)
    # smooth
    if filt:
        size = int(size * sampling_rate)
        signal, _ = st.smoother(signal=signal, kernel='bartlett', size=size, mirror=True)
    

    amps, onsets, pks, end = [], [], [], []
    zeros = st.find_extrema(signal=signal, mode='min')[0] # get zeros
    for z in range(len(zeros)):
        if z == len(zeros) -1:  # last zero
            s = signal[zeros[z]:]  # signal amplitude between event
        else:
            s = signal[zeros[z]:zeros[z + 1]]  # signal amplitude between event
            
        pk = st.find_extrema(signal=s, mode='max')[0]  # get pk between events
        #print("_pks", pks)
        for p in pk:
            if (s[p] - s[0]) > (min_amplitude*np.max(signal)):  # only count events with high amplitude
                pks += [zeros[z] + p]
                onsets += [zeros[z]]
                amps += [s[p] - s[0]]
                if z == len(zeros) -1:  # last zero
                    end += [len(signal)]                    
                else:
                    end += [zeros[z + 1]]
                
    return onsets, pks, amps, end


def edr_times(signal, onsets, pks):
    """ Returns characteristic EDA events.

    Parameters
    ----------
    signal : array
        Input signal.
    onsets : array
        Indices of the SCR onsets.
    peaks : array
        Indices of the SRC peaks.

    Returns
    -------
    half : list
        Indices where edr data drops to 0.5 of the edr events peak amplitudes.
    six : list
        Indices where edr data drops to 0.63 of the edr events peak amplitudes.
    half_rise : list
        Half Rise times, i.e. time between onset and 50% amplitude.
    half_rec : list
        Half Recovery times, i.e. time between peak and 63% amplitude.
    six_rise : list
        63 % rise times, i.e. time between onset and 63% amplitude.
    six_rec : list
        63 % recovery times, i.e. time between peak and 50% amplitude.

    """
    a = np.array(signal[pks[:]] - signal[onsets[:]])
    n_p_1 = 0
    n_p_2 = 0
    half = []
    six = []
    li = min(len(onsets), len(pks))

    half_rise = []
    half_rec = []
    six_rec = []
    six_rise = []
    for i in range(li):
        n_p_1 += 1
        n_p_2 += 1
        half_rec_amp = 0.5 * (a[i] + signal[onsets[i]])
        six_rec_amp = 0.37 * (a[i] + signal[onsets[i]])
        try:
            wind = np.array(signal[pks[i]:onsets[i + 1]])
        except:
            wind = np.array(signal[pks[i]:])
        for ts_idx in range(len(wind)):
            if wind[ts_idx] <= half_rec_amp:
                half += [pks[i] + ts_idx for n in range(n_p_2)]
                half_rise += [half[-n] - onsets[i] for n in range(n_p_2, 0, -1)]
                half_rec += [half[i-n] - pks[i] for n in range(n_p_2, 0, -1)]
                n_p_1 = 0
                break
        for ts_idx in range(len(wind)):
            if wind[ts_idx] <= six_rec_amp:
                six += [pks[i] + ts_idx for n in range(n_p_2)]
                six_rise += [six[-n] - onsets[i] for n in range(n_p_2, 0, -1)]
                six_rec += [six[-n] - pks[i] for n in range(n_p_2, 0, -1)]
                n_p_2 = 0
                break
    return half, six, half_rise, half_rec, six_rise, six_rec
