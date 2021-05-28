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


def eda(signal=None, sampling_rate=1000.0, path=None, show=True, min_amplitude=0.1):
    """Process a raw EDA signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EDA signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
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

    sampling_rate = float(sampling_rate)

    # filter signal
    aux, _, _ = st.filter_signal(
        signal=signal,
        ftype="butter",
        band="lowpass",
        order=4,
        frequency=5,
        sampling_rate=sampling_rate,
    )

    # smooth
    sm_size = int(0.75 * sampling_rate)
    filtered, _ = st.smoother(signal=aux, kernel="boxzen", size=sm_size, mirror=True)

    # get SCR info
    onsets, peaks, amps = kbk_scr(
        signal=filtered, sampling_rate=sampling_rate, min_amplitude=min_amplitude
    )

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_eda(
            ts=ts,
            raw=signal,
            filtered=filtered,
            onsets=onsets,
            peaks=peaks,
            amplitudes=amps,
            path=path,
            show=True,
        )

    # output
    args = (ts, filtered, onsets, peaks, amps)
    names = ("ts", "filtered", "onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def basic_scr(signal=None, sampling_rate=1000.0):
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
    pi, _ = st.find_extrema(signal=signal, mode="max")
    ni, _ = st.find_extrema(signal=signal, mode="min")

    # sanity check
    if len(pi) == 0 or len(ni) == 0:
        raise ValueError("Could not find SCR pulses.")

    # pair vectors
    if ni[0] > pi[0]:
        ni = ni[1:]
    if pi[-1] < ni[-1]:
        pi = pi[:-1]
    if len(pi) > len(ni):
        pi = pi[:-1]

    li = min(len(pi), len(ni))
    i1 = pi[:li]
    i3 = ni[:li]

    # indices
    i0 = np.array((i1 + i3) / 2.0, dtype=int)
    if i0[0] < 0:
        i0[0] = 0

    # amplitude
    a = signal[i0] - signal[i3]

    # output
    args = (i3, i0, a)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)


def kbk_scr(signal=None, sampling_rate=1000.0, min_amplitude=0.1):
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

    # differentiation
    df = np.diff(signal)

    # smooth
    size = int(1.0 * sampling_rate)
    df, _ = st.smoother(signal=df, kernel="bartlett", size=size, mirror=True)

    # zero crosses
    (zeros,) = st.zero_cross(signal=df, detrend=False)
    if np.all(df[: zeros[0]] > 0):
        zeros = zeros[1:]
    if np.all(df[zeros[-1] :] > 0):
        zeros = zeros[:-1]

    scrs, amps, ZC, pks = [], [], [], []
    for i in range(0, len(zeros) - 1, 2):
        scrs += [df[zeros[i] : zeros[i + 1]]]
        ZC += [zeros[i]]
        ZC += [zeros[i + 1]]
        pks += [zeros[i] + np.argmax(df[zeros[i] : zeros[i + 1]])]
        amps += [signal[pks[-1]] - signal[ZC[-2]]]

    # exclude SCRs with small amplitude
    thr = min_amplitude * np.max(amps)
    idx = np.where(amps > thr)

    scrs = np.array(scrs, dtype=np.object)[idx]
    amps = np.array(amps)[idx]
    ZC = np.array(ZC)[np.array(idx) * 2]
    pks = np.array(pks, dtype=int)[idx]

    onsets = ZC[0].astype(int)

    # output
    args = (onsets, pks, amps)
    names = ("onsets", "peaks", "amplitudes")

    return utils.ReturnTuple(args, names)
