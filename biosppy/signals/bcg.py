# -*- coding: utf-8 -*-
"""
biosppy.signals.bcg
-------------------

This module provides methods to process Ballistocardiographic (BCG) signals.
Implemented code assumes a single-channel head-to-foot like BCG signal.

:author: Guillaume Cathelain

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip

# 3rd party
import numpy as np
import scipy.signal as ss
from cv2 import matchTemplate,TM_CCORR_NORMED

# local
from . import tools as st
from .. import plotting, utils
from . import ecg

def bcg(signal=None, sampling_rate=1000., show=True):
    """Process a raw BCG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered BCG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
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

    # segment
    gpeaks,hpeaks,ipeaks,jpeaks,filtered = bsegpp_segmenter(signal=signal,
                                            sampling_rate=sampling_rate)

    # extract templates
    templates, jpeaks = extract_heartbeats(signal=filtered,
                                           peaks=jpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.4,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=jpeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.4, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        plotting.plot_bcg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          jpeaks=jpeaks,
                          templates_ts=ts_tmpl,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, gpeaks,hpeaks,ipeaks,jpeaks, ts_tmpl, templates,
            ts_hr, hr)
    names = ('ts', 'filtered', 'gpeaks', 'hpeaks', 'ipeaks', 'jpeaks',
                'templates_ts', 'templates','heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def bsegpp_segmenter(signal=None, sampling_rate=1000., thresholds= [0.05,5],
                        R=0.1, t1=0.6, H=0.2, I=0.3, J=0.4):
    """BSEG++ BCG cycle extraction algorithm.

    Follows the approach by Akhbardeh et al. [Akhb02]_.
    It was adapted to our BCG device, which measures higher G-peaks in the [1,2]
     Hz frequency band. Thus G-peaks are here synchronization points and H, I, J
     peaks are searched time ranges depending on G-peaks positions.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    thresholds : array
        Lower and upper amplitude threshold values for local maxima of the
        absolute coarse signal.
    R : float
        Range of local extrema search for final synchronization points (seconds).
        Empirically 0.1<R<0.5.
    t1 : float
        Minimum delay between final synchronization points (seconds).
        Empirically 0.4<t1<0.6
    H : float
        Maximum delay between G and H waves (seconds).
    I : float
        Maximum delay between G and I waves (seconds).
    J : float
        Maximum delay between G and J waves (seconds).
    Returns
    -------
    gpeaks : array
        G-peak location indices.
    hpeaks : array
        H-peak location indices.
    ipeaks : array
        I-peak location indices.
    jpeaks : array
        J-peak location indices.
    filtered : array
        Bandpassed signal in the [2,20] Hz frequency range.

    References
    ----------
    .. [Akhb02] A. Akhbardeh, B. Kaminska, K. Tavakolian, "BSeg++: A modified
    Blind Segmentation Method for Ballistocardiogram Cycle Extraction",
    Proceedings of the 29th Annual International Conference of the IEEE EMBS,
    2007

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # 1) normalization
    signal_normed = signal - np.mean(signal)
    signal_normed /= max(abs(signal_normed))
    signal_normed *= 5

    # 2) filtering
    order = 4
    filtered, _, _ = st.filter_signal(signal=signal_normed,
                                      ftype='butter',
                                      band='bandpass',
                                      order=order,
                                      frequency=[2, 20],
                                      sampling_rate=sampling_rate)

    # 3) extract coarse bcg
    order = 6
    coarse, _, _ = st.filter_signal(signal=signal_normed,
                                      ftype='butter',
                                      band='highpass',
                                      order=order,
                                      frequency=1,
                                      sampling_rate=sampling_rate)
    order = 6
    coarse, _, _ = st.filter_signal(signal=coarse,
                                    ftype='butter',
                                    band='lowpass',
                                    order=order,
                                    frequency=2,
                                    sampling_rate=sampling_rate)
    coarse = abs(coarse)

    # synchronization points
    # a) local maxima of absolute coarse BCG with distance constraint
    cntr,properties = ss.find_peaks(coarse,
                                        height=thresholds,
                                        threshold=None,
                                        distance=int(t1*sampling_rate))

    # b) final synchronization points
    p, = correct_peaks(signal=-filtered,
                             peaks=cntr,
                             sampling_rate=sampling_rate,
                             tol=R)

    # define G waves
    gpeaks = p
    # search for H waves
    hpeaks, = search_peaks(signal=filtered,
                             peaks=gpeaks,
                             sampling_rate=sampling_rate,
                             before = 0,
                             after = H)
    # search for I waves
    ipeaks, = search_peaks(signal=-filtered,
                             peaks=gpeaks,
                             sampling_rate=sampling_rate,
                             before = -H,
                             after = I)
    # search for J waves
    jpeaks, = search_peaks(signal=filtered,
                             peaks=gpeaks,
                             sampling_rate=sampling_rate,
                             before = 0,
                             after = J)

    return utils.ReturnTuple((gpeaks,hpeaks,ipeaks,jpeaks,filtered),
                                ('gpeaks','hpeaks','ipeaks','jpeaks','filtered'))


def template_matching(signal=None,filtered=None,peaks=None,sampling_rate=1000.,
                            threshold = 0.5,R=0.1,show=True):
    """Manual template matching algorithm.

    Follows the approach by Shin et al. [Shin03]_.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    filtered : array
        Input filtered BCG signal, bandpassed to the [2,20] Hz frequency range.
    peaks : array
        J-peaks labels.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float
        Minimal correlation value for local maxima.
    R : float
        Range of local extrema search for final synchronization points (seconds).
        Empirically 0.1<R<0.5.

    Returns
    -------
    template : array
        Template model.
    peaks : array
        J-peaks location indices.

    References
    ----------
    .. [Shin03] J. H. Shin, B. H. Choi, Y. G. Lim, D. U. Jeong, K. S. Park,
    "Automatic Ballistocardiogram (BCG) Beat Detection Using a Template Matching
    Approach", Proceedings of the 30th Annual International Conference of the
    IEEE EMBS, 2008

    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input unfiltered signal.")
    if filtered is None:
        raise TypeError("Please specify an input filtered signal.")
    if peaks is None:
        raise TypeError("Please specify peaks indices in the input signal.")

    # ensure numpy
    signal = np.array(signal)
    filtered = np.array(filtered)
    sampling_rate = float(sampling_rate)

    #template modelling
    templates, peaks = extract_heartbeats(signal=filtered, peaks=peaks,
                                sampling_rate=1000., before=0.4, after=0.4)
    for n,tmpl in enumerate(templates):
        tmpl -= np.mean(tmpl)
        tmpl /= max(abs(tmpl))
        templates[n] = tmpl
    template = np.mean(templates,axis=0)

    #template_matching
    corr = matchTemplate(filtered.astype('float32'),template.astype('float32'),TM_CCORR_NORMED)
    corr = corr.flatten()
    cntr,properties = ss.find_peaks(corr,height=threshold)
    cntr += int(len(template)/2)
    peaks, = correct_peaks(signal=filtered,
                             peaks=cntr,
                             sampling_rate=sampling_rate,
                             tol=R)
    # plot
    if show:
        # extract templates
        templates, peaks = extract_heartbeats(signal=filtered,
                                               peaks=peaks,
                                               sampling_rate=sampling_rate,
                                               before=0.4,
                                               after=0.4)
        # compute heart rate
        hr_idx, hr = st.get_heart_rate(beats=peaks,
                                       sampling_rate=sampling_rate,
                                       smooth=True,
                                       size=3)
        # get time vectors
        length = len(signal)
        T = (length - 1) / sampling_rate
        ts = np.linspace(0, T, length, endpoint=True)
        ts_hr = ts[hr_idx]
        ts_tmpl = np.linspace(-0.4, 0.4, templates.shape[1], endpoint=False)

        plotting.plot_bcg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          jpeaks=peaks,
                          templates_ts=ts_tmpl,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    return utils.ReturnTuple((template,peaks),('template','peaks'))


def correct_peaks(signal=None, peaks=None, sampling_rate=1000., tol=0.3):
    return ecg.correct_rpeaks(signal=signal,
                                rpeaks=peaks,
                                sampling_rate=sampling_rate,
                                tol=tol)

def search_peaks(signal=None, peaks=None, sampling_rate=1000.,
                                                before=0.2, after=0.2):
    return ecg.correct_rpeaks(signal=signal,
                            rpeaks=peaks+int(sampling_rate*(after-before)/2),
                            sampling_rate=sampling_rate,
                            tol=(after+before)/2)

def extract_heartbeats(signal=None, peaks=None, sampling_rate=1000.,
                       before=0.4, after=0.4):
    return ecg.extract_heartbeats(signal=signal,
                                    rpeaks=peaks,
                                    sampling_rate=sampling_rate,
                                    before=before,
                                    after=after)
