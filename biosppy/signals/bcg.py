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
import scipy.ndimage as si
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.fftpack as sf
import scipy.optimize as so
from cv2 import matchTemplate,TM_CCORR_NORMED,TM_CCORR
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby
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

    Follows the approach by Akhbardeh et al. [Akhb07]_.
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
    .. [Akhb07] A. Akhbardeh, B. Kaminska, K. Tavakolian, "BSeg++: A modified
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
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=4,
                                      frequency=[2, 20],
                                      sampling_rate=sampling_rate)

    # 3) extract coarse bcg
    coarse, _, _ = st.filter_signal(signal=signal_normed,
                                      ftype='butter',
                                      band='highpass',
                                      order=6,
                                      frequency=1,
                                      sampling_rate=sampling_rate)
    coarse, _, _ = st.filter_signal(signal=coarse,
                                    ftype='butter',
                                    band='lowpass',
                                    order=6,
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

    Follows the approach by Shin et al. [Shin08]_.

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
    .. [Shin08] J. H. Shin, B. H. Choi, Y. G. Lim, D. U. Jeong, K. S. Park,
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

def adaptive_heartbeat_modelling(signal=None,sampling_rate=1000.,initial_length=0.6,residual_threshold = 0.35, show=True):

    """Adaptive Heartbeat Modelling.

    Follows the approach by Paalasmaa et al. [Paal14]_. Only suitable here for 15s-long BCG.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    initial_length : float, optional
        Initial length of the template.
    residual_threshold :
        Threshold for heartbeat intervals selection.

    Returns
    -------
    template : array
        Heartbeat model.
    peaks : array
        Heartbeats location indices.

    References
    ----------
    .. [Paal14] J. Paalasmaa, H. Toivonen, M. Partinen,
    "Adaptive heartbeat modeling for beat-to-beat heart rate measurement in
    ballistocardiograms", IEEE journal of biomedical and health informatics, 2015

    """
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)

    #preprocessing
    signal -= np.mean(signal)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='lowpass',
                                      order=2,
                                      frequency=10,
                                      sampling_rate=sampling_rate)
    gaussian_filter_std = 0.1
    filtered -= si.gaussian_filter(filtered,gaussian_filter_std*sampling_rate)

    #D. Initial estimation of the heartbeat model
    #clustering
    filtered_grad = np.gradient(filtered)
    windows_center_p,_ = ss.find_peaks(filtered_grad)
    windows_center_n,_ = ss.find_peaks(-filtered_grad)
    windows_center = np.sort(np.concatenate((windows_center_p,windows_center_n)))
    windows, windows_center = extract_heartbeats(signal=filtered,
                                           peaks=windows_center,
                                           sampling_rate=sampling_rate,
                                           before=initial_length/2,
                                           after=initial_length/2)

    #clustering
    dist_matrix = ssd.pdist(windows)
    n = len(windows)
    linkage_matrix = sch.linkage(dist_matrix,method='complete')
    densest_4_cluster_indices, = np.where(linkage_matrix[:,3]==4)
    densest_4_cluster_index = densest_4_cluster_indices[0]
    leader_node = densest_4_cluster_index + n
    max_inconsistent_value = linkage_matrix[densest_4_cluster_index,2]
    flat_clusters = sch.fcluster(linkage_matrix,max_inconsistent_value,criterion='distance')
    L,M = sch.leaders(linkage_matrix,flat_clusters)
    leaves, = np.where(flat_clusters == M[L == leader_node])

    windows, windows_center = extract_heartbeats(signal=filtered,
                                           peaks=windows_center[leaves],
                                           sampling_rate=sampling_rate,
                                           before=1.25,
                                           after=1.25)

    mu = np.mean(windows,axis=0)


    hvs_result = modified_heart_valve_signal(signal = signal, sampling_rate=sampling_rate)
    hvs = hvs_result['hvs']
    hvs_minima,_ = ss.find_peaks(-hvs)
    half_lengths = []
    for center in windows_center:
        half_lengths.append(min(center - hvs_minima[hvs_minima<center]))
        half_lengths.append(min(hvs_minima[hvs_minima>center] -  center))

    half_len = min(half_lengths)
    mu = mu[int(len(mu)/2)-half_len:int(len(mu)/2)+half_len]
    mu_center = int(len(mu)/2)

    #E/ Detecting heartbeat position candidates
    peaks=[]
    ta=[]
    tb=[]

    for iter in range(2):
        peaks=[]
        ta =[]
        tb=[]
        half_len = int(initial_length*sampling_rate/2)
        if (half_len > mu_center)|(half_len > len(mu)-mu_center):
            raise ValueError('Template is too short or badly centered')
        mu_corr = mu[mu_center-half_len:mu_center+half_len]
        corr = matchTemplate(filtered.astype('float32'),mu_corr.astype('float32'),TM_CCORR_NORMED)
        corr = corr.flatten()
        candidates_pos,_ = ss.find_peaks(corr)
        corr_delay = -mu_center+half_len

        #F/Detecting beat-to-beat intervals
        half_len = int(1*sampling_rate)
        if half_len > len(mu)-mu_center:
            mu2 = np.append(mu,np.zeros(2*half_len-len(mu)))
        else:
            mu2 = mu[:int(2*sampling_rate)]

        candidates_pos += corr_delay
        candidates_pos = candidates_pos[candidates_pos>=0]

        #1) Initialize ta to the first candidate position
        ta_cand = candidates_pos[0]
        while ta_cand < candidates_pos[-1]:
            try:
                if ta_cand+int(2*sampling_rate)>len(filtered):
                    raise Exception
                sa = filtered[ta_cand:ta_cand+int(2*sampling_rate)]
                za = so.least_squares(lambda z: np.mean(np.power(sa-z*mu2,2)),1).x[0]
                xa = za*mu2
                #2) Find candidates for tb
                tb_candidates = candidates_pos[np.logical_and(ta_cand+int(0.4*sampling_rate)<candidates_pos,candidates_pos<ta_cand+int(2*sampling_rate))]
                #3) find best tb or find another ta -> step 2)
                for tb_cand in tb_candidates:
                    if tb_cand+int(2*sampling_rate)>len(filtered):
                        raise Exception
                    sb = filtered[tb_cand:tb_cand+int(2*sampling_rate)]
                    zb = so.least_squares(lambda z: np.mean(np.power(sb-z*mu2,2)),1).x[0]
                    xb = zb*mu2
                    xa_tmp = np.concatenate((xa,np.zeros(max([0,2*(tb_cand-ta_cand)-int(2*sampling_rate)]))))
                    xb_tmp = np.concatenate((np.zeros(tb_cand-ta_cand),xb))
                    x = xa_tmp[:2*(tb_cand-ta_cand)]+xb_tmp[:2*(tb_cand-ta_cand)]
                    s = filtered[ta_cand:ta_cand+2*(tb_cand-ta_cand)]
                    eps = s - x

                    if (np.mean(np.power(eps,2)) < residual_threshold*np.mean(np.power(s,2))) & (max([za,zb])<2*min([za,zb])):
                        ta.append(ta_cand)
                        tb.append(tb_cand)
                        peak_a = ta_cand+mu_center
                        peak_b = tb_cand+mu_center
                        if peak_a not in peaks:
                            peaks.append(peak_a)
                        peaks.append(peak_b)
                        ta_cand = tb_cand
                        break
                    else:
                        continue

                if ta_cand != tb_cand:
                    ta_candidates = candidates_pos[np.logical_and(candidates_pos>ta_cand,candidates_pos<ta_cand+int(2*sampling_rate))]
                    ta_cand = ta_candidates[np.argmax(corr[ta_candidates-corr_delay])]
            except Exception:
                break
        beats = dict(peaks=np.array(peaks),ta =np.array(ta),tb=np.array(tb))

        #G. re-estimation of the model with detected beat to beat intervals
        template_extraction = long_template_extraction(signal = filtered, beats = beats, mu_center = mu_center,sampling_rate=1000.)
        try:
            mu = template_extraction['long_template']
            mu_center_new = template_extraction['long_template_center']
            mu = mu[mu_center_new-mu_center:]
        except KeyError:
            mu = template_extraction['short_template']
        peaks = beats['peaks']
        print('iteration no ',iter,': ',len(peaks),' beats detected')

    #H. Accounting for abrupt changes of the heartbeat shape
    # to complete, with four different instances of the beat-to-beat detection method

    #I. Post-preprocessing
    # slightly different in our case : we added a smoother rather than the non linear filter explained in the paper

    if show:
        # extract templates
        templates, peaks = extract_heartbeats(signal=filtered,
                                               peaks=peaks,
                                               sampling_rate=sampling_rate,
                                               before=0.6,
                                               after=0.2)
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

    return utils.ReturnTuple((mu,peaks),('template','peaks'))


def heart_valve_signal(signal = None, sampling_rate=1000.):
    """Heart valve signal filtering.

    Follows the approach by Friedrich et al. [Frie10]_.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    hvs : array
        Heart valve signal.

    References
    ----------
    .. [Frie10] Heart Rate Estimation on a Beat-to-Beat Basis via
    Ballistocardiography - A hybrid Approach.
    David Friedrich, Xavier L. Aubert, Hartmut Führ and Andreas Brauers

    """

    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=[20, 40],
                                      sampling_rate=sampling_rate)

    hvs = np.power(filtered,2)

    hvs, _, _ = st.filter_signal(signal=hvs,
                                      ftype='butter',
                                      band='lowpass',
                                      order=2,
                                      frequency=2,
                                      sampling_rate=sampling_rate)

    return utils.ReturnTuple((hvs,),('hvs',))


def modified_heart_valve_signal(signal = None, sampling_rate=1000.):
    """Heart valve signal filtering.

    Follows the approach by Bruser et al. [Brus11]_.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    hvs : array
        Heart valve signal.
    center_frequency : float
        Auto-tuned center frequency of the bandpass filter.

    References
    ----------
    .. [Brus11] C. Bruser and K. Stadlthanner and S. de Waele and S. Leonhardt,
    "Adaptive Beat-to-Beat Heart Rate Estimation in Ballistocardiograms",
    IEEE Transactions on Information Technology in Biomedicine, 2011

    """
    N = len(signal)
    signal_f = sf.fft(signal)
    signal_f = signal_f[:int(N/2)+1]
    freq_step = sampling_rate/N
    band_start_index = int(20/freq_step)
    band_stop_index = int(40/freq_step)
    center_frequency_index = np.argmax(signal_f[band_start_index:band_stop_index])
    center_frequency_index += band_start_index
    center_frequency = center_frequency_index*freq_step
    frequency = [center_frequency - 2,  center_frequency + 2]
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=frequency,
                                      sampling_rate=sampling_rate)

    hvs = np.power(filtered,2)

    hvs, _, _ = st.filter_signal(signal=hvs,
                                      ftype='butter',
                                      band='lowpass',
                                      order=2,
                                      frequency=2,
                                      sampling_rate=sampling_rate)

    return utils.ReturnTuple((hvs,center_frequency),('hvs','center_frequency'))

def long_template_extraction(signal = None, beats = None, mu_center=None, sampling_rate=1000.):
    """Long template extraction.

    Follows the approach by Inan et al. [Inan09]_.

    Parameters
    ----------
    signal : array
        Input unfiltered BCG signal.
    beats : dict
        'ta':start indices of heart beat interval
        'tb':stop indices of heart beat interval
        'peaks': peaks positions.
    mu_center : int
        Inner position reference of the template.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    short_template : array
        Template result when there is no detected heartbeat quadruplet
    amplitudes: array
        Relative amplitude of heartbeats
    long_template:
        Template result when there is at least one detected heartbeat quadruplet
    long_template_center:
        Template inner position reference.

    References
    ----------
    .. [Inan09] O. T. Inan, M. Etemadi, R. M. Wiard, G. T. A. Kovacs,
    "Novel Methods for Estimating the Ballistocardiogram Signal Using a Simultaneously Acquired Electrocardiogram",
    31st Annual International Conference of the IEEE EMBS Minneapolis, Minnesota, USA, September 2-6, 2009

    """

    ## Esimation of BCG Heartbeat Amplitudes
    ta = beats['ta']
    tb = beats['tb']
    t = np.append(ta,tb[-1])
    T = tb-ta
    Tmin = np.min(T)
    T = np.insert(T,0,0)
    x = np.array([signal[tn:tn+Tmin] for tn in t])
    s = np.mean(x,axis=0)

    ## Modified Ensemble Averaging for Extracting Long-Window Pulse Response
    # first find the longest segment with adjacent IBIs.
    is_adjacent = (tb[:-1] == ta[1:])
    are_adjacent_beats = [list([g[0] for g in group]) for key,group in groupby(enumerate(is_adjacent),lambda x:x[1]) if key == True]
    adjacent_beats_indices = np.array([seg.append(seg[-1]+1) or seg for seg in are_adjacent_beats])
    adjacent_beats_len = np.array([len(seg) for seg in adjacent_beats_indices])

    # if it's long enough find quadruplet templates
    if max(adjacent_beats_len)<4-1:
        return utils.ReturnTuple((s,),('short_template',))

    longest_segment_index = np.argmax(adjacent_beats_len)
    ta = ta[adjacent_beats_indices[longest_segment_index]]
    tb = tb[adjacent_beats_indices[longest_segment_index]]
    t = np.append(ta,tb[-1])
    T = tb-ta
    Tmin = min(T)
    Tmax = max(T)
    T = np.insert(T,0,0)
    x = np.array([signal[tn:tn+Tmin] for tn in t])
    s = np.mean(x,axis=0)
    Rss = matchTemplate(s.astype('float32'),s.astype('float32'),TM_CCORR)
    a = np.array([matchTemplate(xn.astype('float32'),s.astype('float32'),TM_CCORR)/Rss for xn in x]).flatten()
    K = 4*Tmax
    N = len(t)
    x_tilde = []
    for tn in t:
        if (tn>Tmax)&(tn+3*Tmax<len(signal)):
            x_tilde.append(signal[tn-Tmax:tn+3*Tmax])
    x_tilde = np.array(x_tilde)
    hcon = np.mean(x_tilde,axis=0)

    h = np.zeros(K)
    for n in range(1,N-2):
        h += x_tilde[n]
        delay = -T[n]+Tmax
        sn_1 = delay_and_pad(signal = s, delay = delay, desired_len = K)
        h -= a[n-1]*sn_1
        delay = T[n+1]+Tmax
        sn1 = delay_and_pad(signal = s, delay = delay, desired_len = K)
        h -= a[n+1]*sn1
        delay = T[n+2]+T[n+1]+Tmax
        sn2 = delay_and_pad(signal = s, delay = delay, desired_len = K)
        h -= a[n+2]*sn2

    h /= N-3
    mu_center += Tmax
    return utils.ReturnTuple((s,a,h,mu_center),('short_template','amplitudes','long_template','long_template_center'))

def delay_and_pad(signal = None, delay = None, desired_len = None):
    if delay < 0:
        raise TypeError("Please specify a positive delay.")
    elif delay > desired_len-len(signal):
        raise TypeError("Please specify a shorter delay.")
    else:
        return np.pad(signal,(delay,desired_len-len(signal)-delay),'constant',constant_values = (0,0))

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
