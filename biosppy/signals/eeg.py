# -*- coding: utf-8 -*-
"""
biosppy.signals.eeg
-------------------

This module provides methods to process Electroencephalographic (EEG)
signals.

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


def eeg(signal=None, sampling_rate=1000., labels=None, show=True):
    """Process raw EEG signals and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    labels : list, optional
        Channel labels.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered BVP signal.
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    plf_pairs : list
        PLF pair indices.
    plf : array
        PLF matrix; each column is a channel pair.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)
    nch = signal.shape[1]

    if labels is None:
        labels = ['Ch. %d' % i for i in range(nch)]
    else:
        if len(labels) != nch:
            raise ValueError(
                "Number of channels mismatch between signal matrix and labels.")

    # high pass filter
    b, a = st.get_filter(ftype='butter',
                         band='highpass',
                         order=8,
                         frequency=4,
                         sampling_rate=sampling_rate)

    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

    # low pass filter
    b, a = st.get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=40,
                         sampling_rate=sampling_rate)

    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)

    # band power features
    out = get_power_features(signal=filtered,
                             sampling_rate=sampling_rate,
                             size=0.25,
                             overlap=0.5)
    ts_feat = out['ts']
    theta = out['theta']
    alpha_low = out['alpha_low']
    alpha_high = out['alpha_high']
    beta = out['beta']
    gamma = out['gamma']

    # PLF features
    _, plf_pairs, plf = get_plf_features(signal=filtered,
                                         sampling_rate=sampling_rate,
                                         size=0.25,
                                         overlap=0.5)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)

    # plot
    if show:
        plotting.plot_eeg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          labels=labels,
                          features_ts=ts_feat,
                          theta=theta,
                          alpha_low=alpha_low,
                          alpha_high=alpha_high,
                          beta=beta,
                          gamma=gamma,
                          plf_pairs=plf_pairs,
                          plf=plf,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, ts_feat, theta, alpha_low, alpha_high, beta, gamma,
            plf_pairs, plf)
    names = ('ts', 'filtered', 'features_ts', 'theta', 'alpha_low',
             'alpha_high', 'beta', 'gamma', 'plf_pairs', 'plf')

    return utils.ReturnTuple(args, names)


def car_reference(signal=None):
    """Change signal reference to the Common Average Reference (CAR).

    Parameters
    ----------
    signal : array
        Input EEG signal matrix; each column is one EEG channel.

    Returns
    -------
    signal : array
        Re-referenced EEG signal matrix; each column is one EEG channel.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    length, nch = signal.shape
    avg = np.mean(signal, axis=1)

    out = signal - np.tile(avg.reshape((length, 1)), nch)

    return utils.ReturnTuple((out,), ('signal',))


def get_power_features(signal=None,
                       sampling_rate=1000.,
                       size=0.25,
                       overlap=0.5):
    """Extract band power features from EEG signals.

    Computes the average signal power, with overlapping windows, in typical
    EEG frequency bands:
    * Theta: from 4 to 8 Hz,
    * Lower Alpha: from 8 to 10 Hz,
    * Higher Alpha: from 10 to 13 Hz,
    * Beta: from 13 to 25 Hz,
    * Gamma: from 25 to 40 Hz.

    Parameters
    ----------
    signal  array
        Filtered EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).

    Returns
    -------
    ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    nch = signal.shape[1]

    sampling_rate = float(sampling_rate)

    # convert sizes to samples
    size = int(size * sampling_rate)
    step = size - int(overlap * size)

    # padding
    min_pad = 1024
    pad = None
    if size < min_pad:
        pad = min_pad - size

    # frequency bands
    bands = [[4, 8], [8, 10], [10, 13], [13, 25], [25, 40]]
    nb = len(bands)

    # windower
    fcn_kwargs = {'sampling_rate': sampling_rate, 'bands': bands, 'pad': pad}
    index, values = st.windower(signal=signal,
                                size=size,
                                step=step,
                                kernel='hann',
                                fcn=_power_features,
                                fcn_kwargs=fcn_kwargs)

    # median filter
    md_size = int(0.625 * sampling_rate / float(step))
    if md_size % 2 == 0:
        # must be odd
        md_size += 1

    for i in range(nb):
        for j in range(nch):
            values[:, i, j], _ = st.smoother(signal=values[:, i, j],
                                             kernel='median',
                                             size=md_size)

    # extract individual bands
    theta = values[:, 0, :]
    alpha_low = values[:, 1, :]
    alpha_high = values[:, 2, :]
    beta = values[:, 3, :]
    gamma = values[:, 4, :]

    # convert indices to seconds
    ts = index.astype('float') / sampling_rate

    # output
    args = (ts, theta, alpha_low, alpha_high, beta, gamma)
    names = ('ts', 'theta', 'alpha_low', 'alpha_high', 'beta', 'gamma')

    return utils.ReturnTuple(args, names)


def get_plf_features(signal=None, sampling_rate=1000., size=0.25, overlap=0.5):
    """Extract Phase-Locking Factor (PLF) features from EEG signals between all
    channel pairs.

    Parameters
    ----------
    signal : array
        Filtered EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).

    Returns
    -------
    ts : array
        Features time axis reference (seconds).
    plf_pairs : list
        PLF pair indices.
    plf : array
        PLF matrix; each column is a channel pair.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    nch = signal.shape[1]

    sampling_rate = float(sampling_rate)

    # convert sizes to samples
    size = int(size * sampling_rate)
    step = size - int(overlap * size)

    # padding
    min_pad = 1024
    N = None
    if size < min_pad:
        N = min_pad

    # PLF pairs
    pairs = [(i, j) for i in range(nch) for j in range(i + 1, nch)]
    nb = len(pairs)

    # windower
    fcn_kwargs = {'pairs': pairs, 'N': N}
    index, values = st.windower(signal=signal,
                                size=size,
                                step=step,
                                kernel='hann',
                                fcn=_plf_features,
                                fcn_kwargs=fcn_kwargs)

    # median filter
    md_size = int(0.625 * sampling_rate / float(step))
    if md_size % 2 == 0:
        # must be odd
        md_size += 1

    for i in range(nb):
        values[:, i], _ = st.smoother(signal=values[:, i],
                                      kernel='median',
                                      size=md_size)

    # convert indices to seconds
    ts = index.astype('float') / sampling_rate

    # output
    args = (ts, pairs, values)
    names = ('ts', 'plf_pairs', 'plf')

    return utils.ReturnTuple(args, names)


def _power_features(signal=None, sampling_rate=1000., bands=None, pad=0):
    """Helper function to compute band power features for each window.

    Parameters
    ----------
    signal : array
        Filtered EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    bands : list
        List of frequency pairs defining the bands.
    pad : int, optional
        Padding for the Fourier Transform (number of zeros added).

    Returns
    -------
    out : array
        Average power for each band and EEG channel; shape is
        (bands, channels).

    """

    nch = signal.shape[1]

    out = np.zeros((len(bands), nch), dtype='float')
    for i in range(nch):
        # compute power spectrum
        freqs, power = st.power_spectrum(signal=signal[:, i],
                                         sampling_rate=sampling_rate,
                                         pad=pad,
                                         pow2=False,
                                         decibel=False)

        # compute average band power
        for j, b in enumerate(bands):
            avg, = st.band_power(freqs=freqs,
                                 power=power,
                                 frequency=b,
                                 decibel=False)
            out[j, i] = avg

    return out


def _plf_features(signal=None, pairs=None, N=None):
    """Helper function to compute PLF features for each window.

    Parameters
    ----------
    signal : array
        Filtered EEG signal matrix; each column is one EEG channel.
    pairs : iterable
        List of signal channel pairs.
    N : int, optional
        Number of Fourier components.

    Returns
    -------
    out : array
        PLF for each channel pair.

    """

    out = np.zeros(len(pairs), dtype='float')
    for i, p in enumerate(pairs):
        # compute PLF
        s1 = signal[:, p[0]]
        s2 = signal[:, p[1]]
        out[i], = st.phase_locking(signal1=s1, signal2=s2, N=N)

    return out
