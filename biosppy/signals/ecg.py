# -*- coding: utf-8 -*-
"""
biosppy.signals.ecg
-------------------

This module provides methods to process Electrocardiographic (ECG) signals.
Implemented code assumes a single-channel Lead I like ECG signal.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range, zip

# 3rd party
import numpy as np
import scipy.signal as ss

# local
from . import tools as st
from .. import plotting, utils


def ecg(signal=None, sampling_rate=1000., show=True):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
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

    # filter signal
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='FIR',
                                      band='bandpass',
                                      order=order,
                                      frequency=[3, 45],
                                      sampling_rate=sampling_rate)

    # segment
    rpeaks, = hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)

    # extract templates
    templates, rpeaks = extract_heartbeats(signal=filtered,
                                           rpeaks=rpeaks,
                                           sampling_rate=sampling_rate,
                                           before=0.2,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=rpeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        plotting.plot_ecg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          rpeaks=rpeaks,
                          templates_ts=ts_tmpl,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          path=None,
                          show=True)

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def _extract_heartbeats(signal=None, rpeaks=None, before=200, after=400):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    before : int, optional
        Number of samples to include before the R peak.
    after : int, optional
        Number of samples to include after the R peak.

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    R = np.sort(rpeaks)
    length = len(signal)
    templates = []
    newR = []

    for r in R:
        a = r - before
        if a < 0:
            continue
        b = r + after
        if b > length:
            break
        templates.append(signal[a:b])
        newR.append(r)

    templates = np.array(templates)
    newR = np.array(newR, dtype='int')

    return templates, newR


def extract_heartbeats(signal=None, rpeaks=None, sampling_rate=1000.,
                       before=0.2, after=0.4):
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.

    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    before : float, optional
        Window size to include before the R peak (seconds).
    after : int, optional
        Window size to include after the R peak (seconds).

    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peak locations.")

    if before < 0:
        raise ValueError("Please specify a non-negative 'before' value.")
    if after < 0:
        raise ValueError("Please specify a non-negative 'after' value.")

    # convert delimiters to samples
    before = int(before * sampling_rate)
    after = int(after * sampling_rate)

    # get heartbeats
    templates, newR = _extract_heartbeats(signal=signal,
                                          rpeaks=rpeaks,
                                          before=before,
                                          after=after)

    return utils.ReturnTuple((templates, newR), ('templates', 'rpeaks'))


def compare_segmentation(reference=None, test=None, sampling_rate=1000.,
                         offset=0, minRR=None, tol=0.05):
    """Compare the segmentation performance of a list of R-peak positions
    against a reference list.

    Parameters
    ----------
    reference : array
        Reference R-peak location indices.
    test : array
        Test R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    offset : int, optional
        Constant a priori offset (number of samples) between reference and
        test R-peak locations.
    minRR : float, optional
        Minimum admissible RR interval (seconds).
    tol : float, optional
        Tolerance between corresponding reference and test R-peak
        locations (seconds).

    Returns
    -------
    TP : int
        Number of true positive R-peaks.
    FP : int
        Number of false positive R-peaks.
    performance : float
        Test performance; TP / len(reference).
    acc : float
        Accuracy rate; TP / (TP + FP).
    err : float
        Error rate; FP / (TP + FP).
    match : list
        Indices of the elements of 'test' that match to an R-peak
        from 'reference'.
    deviation : array
        Absolute errors of the matched R-peaks (seconds).
    mean_deviation : float
        Mean error (seconds).
    std_deviation : float
        Standard deviation of error (seconds).
    mean_ref_ibi : float
        Mean of the reference interbeat intervals (seconds).
    std_ref_ibi : float
        Standard deviation of the reference interbeat intervals (seconds).
    mean_test_ibi : float
        Mean of the test interbeat intervals (seconds).
    std_test_ibi : float
        Standard deviation of the test interbeat intervals (seconds).

    """

    # check inputs
    if reference is None:
        raise TypeError("Please specify an input reference list of R-peak \
                        locations.")

    if test is None:
        raise TypeError("Please specify an input test list of R-peak \
                        locations.")

    if minRR is None:
        minRR = np.inf

    sampling_rate = float(sampling_rate)

    # ensure numpy
    reference = np.array(reference)
    test = np.array(test)

    # convert to samples
    minRR = minRR * sampling_rate
    tol = tol * sampling_rate

    TP = 0
    FP = 0

    matchIdx = []
    dev = []

    for i, r in enumerate(test):
        # deviation to closest R in reference
        ref = reference[np.argmin(np.abs(reference - (r + offset)))]
        error = np.abs(ref - (r + offset))

        if error < tol:
            TP += 1
            matchIdx.append(i)
            dev.append(error)
        else:
            if len(matchIdx) > 0:
                bdf = r - test[matchIdx[-1]]
                if bdf < minRR:
                    # false positive, but removable with RR interval check
                    pass
                else:
                    FP += 1
            else:
                FP += 1

    # convert deviations to time
    dev = np.array(dev, dtype='float')
    dev /= sampling_rate
    nd = len(dev)
    if nd == 0:
        mdev = np.nan
        sdev = np.nan
    elif nd == 1:
        mdev = np.mean(dev)
        sdev = 0.
    else:
        mdev = np.mean(dev)
        sdev = np.std(dev, ddof=1)

    # interbeat interval
    th1 = 1.5  # 40 bpm
    th2 = 0.3  # 200 bpm

    rIBI = np.diff(reference)
    rIBI = np.array(rIBI, dtype='float')
    rIBI /= sampling_rate

    good = np.nonzero((rIBI < th1) & (rIBI > th2))[0]
    rIBI = rIBI[good]

    nr = len(rIBI)
    if nr == 0:
        rIBIm = np.nan
        rIBIs = np.nan
    elif nr == 1:
        rIBIm = np.mean(rIBI)
        rIBIs = 0.
    else:
        rIBIm = np.mean(rIBI)
        rIBIs = np.std(rIBI, ddof=1)

    tIBI = np.diff(test[matchIdx])
    tIBI = np.array(tIBI, dtype='float')
    tIBI /= sampling_rate

    good = np.nonzero((tIBI < th1) & (tIBI > th2))[0]
    tIBI = tIBI[good]

    nt = len(tIBI)
    if nt == 0:
        tIBIm = np.nan
        tIBIs = np.nan
    elif nt == 1:
        tIBIm = np.mean(tIBI)
        tIBIs = 0.
    else:
        tIBIm = np.mean(tIBI)
        tIBIs = np.std(tIBI, ddof=1)

    # output
    perf = float(TP) / len(reference)
    acc = float(TP) / (TP + FP)
    err = float(FP) / (TP + FP)

    args = (TP, FP, perf, acc, err, matchIdx, dev, mdev, sdev, rIBIm, rIBIs,
            tIBIm, tIBIs)
    names = ('TP', 'FP', 'performance', 'acc', 'err', 'match', 'deviation',
             'mean_deviation', 'std_deviation', 'mean_ref_ibi', 'std_ref_ibi',
             'mean_test_ibi', 'std_test_ibi',)

    return utils.ReturnTuple(args, names)


def correct_rpeaks(signal=None, rpeaks=None, sampling_rate=1000., tol=0.05):
    """Correct R-peak locations to the maximum within a tolerance.

    Parameters
    ----------
    signal : array
        ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    tol : int, float, optional
        Correction tolerance (seconds).

    Returns
    -------
    rpeaks : array
        Cerrected R-peak location indices.

    Notes
    -----
    * The tolerance is defined as the time interval :math:`[R-tol, R+tol[`.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peaks.")

    tol = int(tol * sampling_rate)
    length = len(signal)

    newR = []
    for r in rpeaks:
        a = r - tol
        if a < 0:
            continue
        b = r + tol
        if b > length:
            break
        newR.append(a + np.argmax(signal[a:b]))

    newR = sorted(list(set(newR)))
    newR = np.array(newR, dtype='int')

    return utils.ReturnTuple((newR,), ('rpeaks',))


def ssf_segmenter(signal=None, sampling_rate=1000., threshold=20, before=0.03,
                  after=0.01):
    """ECG R-peak segmentation based on the Slope Sum Function (SSF).

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        SSF threshold.
    before : float, optional
        Search window size before R-peak candidate (seconds).
    after : float, optional
        Search window size after R-peak candidate (seconds).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    winB = int(before * sampling_rate)
    winA = int(after * sampling_rate)

    Rset = set()
    length = len(signal)

    # diff
    dx = np.diff(signal)
    dx[dx >= 0] = 0
    dx = dx ** 2

    # detection
    idx, = np.nonzero(dx > threshold)
    idx0 = np.hstack(([0], idx))
    didx = np.diff(idx0)

    # search
    sidx = idx[didx > 1]
    for item in sidx:
        a = item - winB
        if a < 0:
            a = 0
        b = item + winA
        if b > length:
            continue

        r = np.argmax(signal[a:b]) + a
        Rset.add(r)

    # output
    rpeaks = list(Rset)
    rpeaks.sort()
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def christov_segmenter(signal=None, sampling_rate=1000.):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Christov [Chri04]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Chri04] Ivaylo I. Christov, "Real time electrocardiogram QRS
       detection using combined adaptive threshold", BioMedical Engineering
       OnLine 2004, vol. 3:28, 2004

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    length = len(signal)

    # algorithm parameters
    v100ms = int(0.1 * sampling_rate)
    v50ms = int(0.050 * sampling_rate)
    v300ms = int(0.300 * sampling_rate)
    v350ms = int(0.350 * sampling_rate)
    v200ms = int(0.2 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    M_th = 0.4  # paper is 0.6

    # Pre-processing
    # 1. Moving averaging filter for power-line interference suppression:
    # averages samples in one period of the powerline
    # interference frequency with a first zero at this frequency.
    b = np.ones(int(0.02 * sampling_rate)) / 50.
    a = [1]
    X = ss.filtfilt(b, a, signal)
    # 2. Moving averaging of samples in 28 ms interval for electromyogram
    # noise suppression a filter with first zero at about 35 Hz.
    b = np.ones(int(sampling_rate / 35.)) / 35.
    X = ss.filtfilt(b, a, X)
    X, _, _ = st.filter_signal(signal=X,
                               ftype='butter',
                               band='lowpass',
                               order=7,
                               frequency=40.,
                               sampling_rate=sampling_rate)
    X, _, _ = st.filter_signal(signal=X,
                               ftype='butter',
                               band='highpass',
                               order=7,
                               frequency=9.,
                               sampling_rate=sampling_rate)

    k, Y, L = 1, [], len(X)
    for n in range(k + 1, L - k):
        Y.append(X[n] ** 2 - X[n - k] * X[n + k])
    Y = np.array(Y)
    Y[Y < 0] = 0

    # Complex lead
    # Y = abs(scipy.diff(X)) # 1-lead
    # 3. Moving averaging of a complex lead (the sintesis is
    # explained in the next section) in 40 ms intervals a filter
    # with first zero at about 25 Hz. It is suppressing the noise
    # magnified by the differentiation procedure used in the
    # process of the complex lead sintesis.
    b = np.ones(int(sampling_rate / 25.)) / 25.
    Y = ss.lfilter(b, a, Y)

    # Init
    MM = M_th * np.max(Y[:int(5 * sampling_rate)]) * np.ones(5)
    MMidx = 0
    M = np.mean(MM)
    slope = np.linspace(1.0, 0.6, int(sampling_rate))
    Rdec = 0
    R = 0
    RR = np.zeros(5)
    RRidx = 0
    Rm = 0
    QRS = []
    Rpeak = []
    current_sample = 0
    skip = False
    F = np.mean(Y[:v350ms])

    # Go through each sample
    while current_sample < len(Y):
        if QRS:
            # No detection is allowed 200 ms after the current one. In
            # the interval QRS to QRS+200ms a new value of M5 is calculated: newM5 = 0.6*max(Yi)
            if current_sample <= QRS[-1] + v200ms:
                Mnew = M_th * max(Y[QRS[-1]:QRS[-1] + v200ms])
                # The estimated newM5 value can become quite high, if
                # steep slope premature ventricular contraction or artifact
                # appeared, and for that reason it is limited to newM5 = 1.1*M5 if newM5 > 1.5* M5
                # The MM buffer is refreshed excluding the oldest component, and including M5 = newM5.
                Mnew = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                MM[MMidx] = Mnew
                MMidx = np.mod(MMidx + 1, 5)
                # M is calculated as an average value of MM.
                Mtemp = np.mean(MM)
                M = Mtemp
                skip = True
            # M is decreased in an interval 200 to 1200 ms following
            # the last QRS detection at a low slope, reaching 60 % of its
            # refreshed value at 1200 ms.
            elif current_sample >= QRS[-1] + v200ms and current_sample < QRS[-1] + v1200ms:
                M = Mtemp * slope[current_sample - QRS[-1] - v200ms]
            # After 1200 ms M remains unchanged.
            # R = 0 V in the interval from the last detected QRS to 2/3 of the expected Rm.
            if current_sample >= QRS[-1] and current_sample < QRS[-1] + (2 / 3.) * Rm:
                R = 0
            # In the interval QRS + Rm * 2/3 to QRS + Rm, R decreases
            # 1.4 times slower then the decrease of the previously discussed
            # steep slope threshold (M in the 200 to 1200 ms interval).
            elif current_sample >= QRS[-1] + (2 / 3.) * Rm and current_sample < QRS[-1] + Rm:
                R += Rdec
            # After QRS + Rm the decrease of R is stopped
            # MFR = M + F + R
        MFR = M + F + R
        # QRS or beat complex is detected if Yi = MFR
        if not skip and Y[current_sample] >= MFR:
            QRS += [current_sample]
            Rpeak += [QRS[-1] + np.argmax(Y[QRS[-1]:QRS[-1] + v300ms])]
            if len(QRS) >= 2:
                # A buffer with the 5 last RR intervals is updated at any new QRS detection.
                RR[RRidx] = QRS[-1] - QRS[-2]
                RRidx = np.mod(RRidx + 1, 5)
        skip = False
        # With every signal sample, F is updated adding the maximum
        # of Y in the latest 50 ms of the 350 ms interval and
        # subtracting maxY in the earliest 50 ms of the interval.
        if current_sample >= v350ms:
            Y_latest50 = Y[current_sample - v50ms:current_sample]
            Y_earliest50 = Y[current_sample - v350ms:current_sample - v300ms]
            F += (max(Y_latest50) - max(Y_earliest50)) / 1000.
        # Rm is the mean value of the buffer RR.
        Rm = np.mean(RR)
        current_sample += 1

    rpeaks = []
    for i in Rpeak:
        a, b = i - v100ms, i + v100ms
        if a < 0:
            a = 0
        if b > length:
            b = length
        rpeaks.append(np.argmax(signal[a:b]) + a)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def engzee_segmenter(signal=None, sampling_rate=1000., threshold=0.48):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Engelse and Zeelenberg [EnZe79]_ with the
    modifications by Lourenco *et al.* [LSLL12]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    threshold : float, optional
        Detection threshold.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [EnZe79] W. Engelse and C. Zeelenberg, "A single scan algorithm for
       QRS detection and feature extraction", IEEE Comp. in Cardiology,
       vol. 6, pp. 37-42, 1979
    .. [LSLL12] A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
       "Real Time Electrocardiogram Segmentation for Finger Based ECG
       Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # algorithm parameters
    changeM = int(0.75 * sampling_rate)
    Miterate = int(1.75 * sampling_rate)
    v250ms = int(0.25 * sampling_rate)
    v1200ms = int(1.2 * sampling_rate)
    v1500ms = int(1.5 * sampling_rate)
    v180ms = int(0.18 * sampling_rate)
    p10ms = int(np.ceil(0.01 * sampling_rate))
    p20ms = int(np.ceil(0.02 * sampling_rate))
    err_kill = int(0.01 * sampling_rate)
    inc = 1
    mmth = threshold
    mmp = 0.2

    # Differentiator (1)
    y1 = [signal[i] - signal[i - 4] for i in range(4, len(signal))]

    # Low pass filter (2)
    c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
    y2 = np.array([np.dot(c, y1[n - 9:n + 1]) for n in range(9, len(y1))])
    y2_len = len(y2)

    # vars
    MM = mmth * max(y2[:Miterate]) * np.ones(3)
    MMidx = 0
    Th = np.mean(MM)
    NN = mmp * min(y2[:Miterate]) * np.ones(2)
    NNidx = 0
    ThNew = np.mean(NN)
    update = False
    nthfpluss = []
    rpeaks = []

    # Find nthf+ point
    while True:
        # If a previous intersection was found, continue the analysis from there
        if update:
            if inc * changeM + Miterate < y2_len:
                a = (inc - 1) * changeM
                b = inc * changeM + Miterate
                Mnew = mmth * max(y2[a:b])
                Nnew = mmp * min(y2[a:b])
            elif y2_len - (inc - 1) * changeM > v1500ms:
                a = (inc - 1) * changeM
                Mnew = mmth * max(y2[a:])
                Nnew = mmp * min(y2[a:])
            if len(y2) - inc * changeM > Miterate:
                MM[MMidx] = Mnew if Mnew <= 1.5 * MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                NN[NNidx] = Nnew if abs(Nnew) <= 1.5 * abs(NN[NNidx - 1]) else 1.1 * NN[NNidx - 1]
            MMidx = np.mod(MMidx + 1, len(MM))
            NNidx = np.mod(NNidx + 1, len(NN))
            Th = np.mean(MM)
            ThNew = np.mean(NN)
            inc += 1
            update = False
        if nthfpluss:
            lastp = nthfpluss[-1] + 1
            if lastp < (inc - 1) * changeM:
                lastp = (inc - 1) * changeM
            y22 = y2[lastp:inc * changeM + err_kill]
            # find intersection with Th
            try:
                nthfplus = np.intersect1d(np.nonzero(y22 > Th)[0], np.nonzero(y22 < Th)[0] - 1)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
            # adjust index
            nthfplus += int(lastp)
            # if a previous R peak was found:
            if rpeaks:
                # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                if nthfplus - rpeaks[-1] > v250ms and nthfplus - rpeaks[-1] < v1200ms:
                    pass
                # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                elif nthfplus - rpeaks[-1] < v250ms:
                    nthfpluss += [nthfplus]
                    continue
        # no previous intersection, find the first one
        else:
            try:
                aux = np.nonzero(y2[(inc - 1) * changeM:inc * changeM + err_kill] > Th)[0]
                bux = np.nonzero(y2[(inc - 1) * changeM:inc * changeM + err_kill] < Th)[0] - 1
                nthfplus = int((inc - 1) * changeM) + np.intersect1d(aux, bux)[0]
            except IndexError:
                if inc * changeM > len(y2):
                    break
                else:
                    update = True
                    continue
        nthfpluss += [nthfplus]
        # Define 160ms search region
        windowW = np.arange(nthfplus, nthfplus + v180ms)
        # Check if the condition y2[n] < Th holds for a specified
        # number of consecutive points (experimentally we found this number to be at least 10 points)"
        i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
        hold_points = np.diff(np.nonzero(y2[i:f] < ThNew)[0])
        cont = 0
        for hp in hold_points:
            if hp == 1:
                cont += 1
                if cont == p10ms - 1:  # -1 is because diff eats a sample
                    max_shift = p20ms  # looks for X's max a bit to the right
                    if nthfpluss[-1] > max_shift:
                        rpeaks += [np.argmax(signal[i - max_shift:f]) + i - max_shift]
                    else:
                        rpeaks += [np.argmax(signal[i:f]) + i]
                    break
            else:
                cont = 0

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def gamboa_segmenter(signal=None, sampling_rate=1000., tol=0.002):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Gamboa.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    tol : float, optional
        Tolerance parameter.

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # convert to samples
    v_100ms = int(0.1 * sampling_rate)
    v_300ms = int(0.3 * sampling_rate)
    hist, edges = np.histogram(signal, 100, density=True)

    TH = 0.01
    F = np.cumsum(hist)

    v0 = edges[np.nonzero(F > TH)[0][0]]
    v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]

    nrm = max([abs(v0), abs(v1)])
    norm_signal = signal / float(nrm)

    d2 = np.diff(norm_signal, 2)

    b = np.nonzero((np.diff(np.sign(np.diff(-d2)))) == -2)[0] + 2
    b = np.intersect1d(b, np.nonzero(-d2 > tol)[0])

    if len(b) < 3:
        rpeaks = []
    else:
        b = b.astype('float')
        rpeaks = []
        previous = b[0]
        for i in b[1:]:
            if i - previous > v_300ms:
                previous = i
                rpeaks.append(np.argmax(signal[int(i):int(i + v_100ms)]) + i)

    rpeaks = sorted(list(set(rpeaks)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))


def hamilton_segmenter(signal=None, sampling_rate=1000.):
    """ECG R-peak segmentation algorithm.

    Follows the approach by Hamilton [Hami02]_.

    Parameters
    ----------
    signal : array
        Input filtered ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    rpeaks : array
        R-peak location indices.

    References
    ----------
    .. [Hami02] P.S. Hamilton, "Open Source ECG Analysis Software
       Documentation", E.P.Limited, 2002

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
    length = len(signal)
    dur = length / sampling_rate

    # algorithm parameters
    v1s = int(1. * sampling_rate)
    v100ms = int(0.1 * sampling_rate)
    TH_elapsed = np.ceil(0.36 * sampling_rate)
    sm_size = int(0.08 * sampling_rate)
    init_ecg = 8  # seconds for initialization
    if dur < init_ecg:
        init_ecg = int(dur)

    # filtering
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='lowpass',
                                      order=4,
                                      frequency=25.,
                                      sampling_rate=sampling_rate)
    filtered, _, _ = st.filter_signal(signal=filtered,
                                      ftype='butter',
                                      band='highpass',
                                      order=4,
                                      frequency=3.,
                                      sampling_rate=sampling_rate)

    # diff
    dx = np.abs(np.diff(filtered, 1) * sampling_rate)

    # smoothing
    dx, _ = st.smoother(signal=dx, kernel='hamming', size=sm_size, mirror=True)

    # buffers
    qrspeakbuffer = np.zeros(init_ecg)
    noisepeakbuffer = np.zeros(init_ecg)
    peak_idx_test = np.zeros(init_ecg)
    noise_idx = np.zeros(init_ecg)
    rrinterval = sampling_rate * np.ones(init_ecg)

    a, b = 0, v1s
    all_peaks, _ = st.find_extrema(signal=dx, mode='max')
    for i in range(init_ecg):
        peaks, values = st.find_extrema(signal=dx[a:b], mode='max')
        try:
            ind = np.argmax(values)
        except ValueError:
            pass
        else:
            # peak amplitude
            qrspeakbuffer[i] = values[ind]
            # peak location
            peak_idx_test[i] = peaks[ind] + a

        a += v1s
        b += v1s

    # thresholds
    ANP = np.median(noisepeakbuffer)
    AQRSP = np.median(qrspeakbuffer)
    TH = 0.475
    DT = ANP + TH * (AQRSP - ANP)
    DT_vec = []
    indexqrs = 0
    indexnoise = 0
    indexrr = 0
    npeaks = 0
    offset = 0

    beats = []

    # detection rules
    # 1 - ignore all peaks that precede or follow larger peaks by less than 200ms
    lim = int(np.ceil(0.2 * sampling_rate))
    diff_nr = int(np.ceil(0.045 * sampling_rate))
    bpsi, bpe = offset, 0

    for f in all_peaks:
        DT_vec += [DT]
        # 1 - Checking if f-peak is larger than any peak following or preceding it by less than 200 ms
        peak_cond = np.array((all_peaks > f - lim) * (all_peaks < f + lim) * (all_peaks != f))
        peaks_within = all_peaks[peak_cond]
        if peaks_within.any() and (max(dx[peaks_within]) > dx[f]):
            continue

        # 4 - If the peak is larger than the detection threshold call it a QRS complex, otherwise call it noise
        if dx[f] > DT:
            # 2 - look for both positive and negative slopes in raw signal
            if f < diff_nr:
                diff_now = np.diff(signal[0:f + diff_nr])
            elif f + diff_nr >= len(signal):
                diff_now = np.diff(signal[f - diff_nr:len(dx)])
            else:
                diff_now = np.diff(signal[f - diff_nr:f + diff_nr])
            diff_signer = diff_now[diff_now > 0]
            if len(diff_signer) == 0 or len(diff_signer) == len(diff_now):
                continue
            # RR INTERVALS
            if npeaks > 0:
                # 3 - in here we check point 3 of the Hamilton paper
                # that is, we check whether our current peak is a valid R-peak.
                prev_rpeak = beats[npeaks - 1]

                elapsed = f - prev_rpeak
                # if the previous peak was within 360 ms interval
                if elapsed < TH_elapsed:
                    # check current and previous slopes
                    if prev_rpeak < diff_nr:
                        diff_prev = np.diff(signal[0:prev_rpeak + diff_nr])
                    elif prev_rpeak + diff_nr >= len(signal):
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr:len(dx)])
                    else:
                        diff_prev = np.diff(signal[prev_rpeak - diff_nr:prev_rpeak + diff_nr])

                    slope_now = max(diff_now)
                    slope_prev = max(diff_prev)

                    if (slope_now < 0.5 * slope_prev):
                        # if current slope is smaller than half the previous one, then it is a T-wave
                        continue
                if dx[f] < 3. * np.median(qrspeakbuffer):  # avoid retarded noise peaks
                    beats += [int(f) + bpsi]
                else:
                    continue

                if bpe == 0:
                    rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                    indexrr += 1
                    if indexrr == init_ecg:
                        indexrr = 0
                else:
                    if beats[npeaks] > beats[bpe - 1] + v100ms:
                        rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                        indexrr += 1
                        if indexrr == init_ecg:
                            indexrr = 0

            elif dx[f] < 3. * np.median(qrspeakbuffer):
                beats += [int(f) + bpsi]
            else:
                continue

            npeaks += 1
            qrspeakbuffer[indexqrs] = dx[f]
            peak_idx_test[indexqrs] = f
            indexqrs += 1
            if indexqrs == init_ecg:
                indexqrs = 0
        if dx[f] <= DT:
            # 4 - not valid
            # 5 - If no QRS has been detected within 1.5 R-to-R intervals,
            # there was a peak that was larger than half the detection threshold,
            # and the peak followed the preceding detection by at least 360 ms,
            # classify that peak as a QRS complex
            tf = f + bpsi
            # RR interval median
            RRM = np.median(rrinterval)  # initial values are good?

            if len(beats) >= 2:
                elapsed = tf - beats[npeaks - 1]

                if elapsed >= 1.5 * RRM and elapsed > TH_elapsed:
                    if dx[f] > 0.5 * DT:
                        beats += [int(f) + offset]
                        # RR INTERVALS
                        if npeaks > 0:
                            rrinterval[indexrr] = beats[npeaks] - beats[npeaks - 1]
                            indexrr += 1
                            if indexrr == init_ecg:
                                indexrr = 0
                        npeaks += 1
                        qrspeakbuffer[indexqrs] = dx[f]
                        peak_idx_test[indexqrs] = f
                        indexqrs += 1
                        if indexqrs == init_ecg:
                            indexqrs = 0
                else:
                    noisepeakbuffer[indexnoise] = dx[f]
                    noise_idx[indexnoise] = f
                    indexnoise += 1
                    if indexnoise == init_ecg:
                        indexnoise = 0
            else:
                noisepeakbuffer[indexnoise] = dx[f]
                noise_idx[indexnoise] = f
                indexnoise += 1
                if indexnoise == init_ecg:
                    indexnoise = 0

        # Update Detection Threshold
        ANP = np.median(noisepeakbuffer)
        AQRSP = np.median(qrspeakbuffer)
        DT = ANP + 0.475 * (AQRSP - ANP)

    beats = np.array(beats)

    r_beats = []
    thres_ch = 0.85
    adjacency = 0.05 * sampling_rate
    for i in beats:
        error = [False, False]
        if i - lim < 0:
            window = signal[0:i + lim]
            add = 0
        elif i + lim >= length:
            window = signal[i - lim:length]
            add = i - lim
        else:
            window = signal[i - lim:i + lim]
            add = i - lim
        # meanval = np.mean(window)
        w_peaks, _ = st.find_extrema(signal=window, mode='max')
        w_negpeaks, _ = st.find_extrema(signal=window, mode='min')
        zerdiffs = np.where(np.diff(window) == 0)[0]
        w_peaks = np.concatenate((w_peaks, zerdiffs))
        w_negpeaks = np.concatenate((w_negpeaks, zerdiffs))

        pospeaks = sorted(zip(window[w_peaks], w_peaks), reverse=True)
        negpeaks = sorted(zip(window[w_negpeaks], w_negpeaks))

        try:
            twopeaks = [pospeaks[0]]
        except IndexError:
            twopeaks = []
        try:
            twonegpeaks = [negpeaks[0]]
        except IndexError:
            twonegpeaks = []

        # getting positive peaks
        for i in range(len(pospeaks) - 1):
            if abs(pospeaks[0][1] - pospeaks[i + 1][1]) > adjacency:
                twopeaks.append(pospeaks[i + 1])
                break
        try:
            posdiv = abs(twopeaks[0][0] - twopeaks[1][0])
        except IndexError:
            error[0] = True

        # getting negative peaks
        for i in range(len(negpeaks) - 1):
            if abs(negpeaks[0][1] - negpeaks[i + 1][1]) > adjacency:
                twonegpeaks.append(negpeaks[i + 1])
                break
        try:
            negdiv = abs(twonegpeaks[0][0] - twonegpeaks[1][0])
        except IndexError:
            error[1] = True

        # choosing type of R-peak
        n_errors = sum(error)
        try:
            if not n_errors:
                if posdiv > thres_ch * negdiv:
                    # pos noerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg noerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif n_errors == 2:
                if abs(twopeaks[0][1]) > abs(twonegpeaks[0][1]):
                    # pos allerr
                    r_beats.append(twopeaks[0][1] + add)
                else:
                    # neg allerr
                    r_beats.append(twonegpeaks[0][1] + add)
            elif error[0]:
                # pos poserr
                r_beats.append(twopeaks[0][1] + add)
            else:
                # neg negerr
                r_beats.append(twonegpeaks[0][1] + add)
        except IndexError:
            continue

    rpeaks = sorted(list(set(r_beats)))
    rpeaks = np.array(rpeaks, dtype='int')

    return utils.ReturnTuple((rpeaks,), ('rpeaks',))
