# -*- coding: utf-8 -*-
"""
biosppy.SQIs
---------------

This module provides signal quality index by [Zhao18]

:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# 3rd party
import numpy as np
from scipy import stats, integrate, signal


def bSQI(detector_1, detector_2, fs=1000., mode='simple', search_window=150):
    """ Comparison of the output of two detectors.

    Parameters
    ----------
    detector_1 : array
        Output of the first detector.
    detector_2 : array
        Output of the second detector.
    fs: int, optional
        Sampling rate, in Hz.
    mode : str, optional
        If 'simple', return only the percentage of beats detected by both. If 'matching', return the peak matching degree.
        If 'n_double' returns the number of matches divided by the sum of all minus the matches.
    search_window : int, optional
        Search window around each peak, in ms.

    Returns
    -------
    bSQI : float
        Performance of both detectors.

   """

    if detector_1 is None or detector_2 is None:
        raise TypeError("Input Error, check detectors outputs")
    search_window = int(search_window / 1000 * fs)
    both = 0
    for i in detector_1:
        for j in range(max([0, i - search_window]), i + search_window):
            if j in detector_2:
                both += 1
                break

    if mode == 'simple':
        return (both / len(detector_1)) * 100
    elif mode == 'matching':
        return (2 * both) / (len(detector_1) + len(detector_2))
    elif mode == 'n_double':
        return both / (len(detector_1) + len(detector_2) - both)


def sSQI(signal):
    """ Return the skewness of the signal

    Parameters
    ----------
    signal : array
        ECG signal.

    Returns
    -------
    skewness : float
        Skewness value.

    """
    if signal is None:
        raise TypeError("Please specify an input signal")

    return stats.skew(signal)


def kSQI(signal, fisher=True):
    """ Return the kurtosis of the signal

    Parameters
    ----------
    signal : array
        ECG signal.
    fisher : bool, optional
        If True,Fisher’s definition is used (normal ==> 0.0). If False, Pearson’s definition is used (normal ==> 3.0).

    Returns
    -------
    kurtosis : float
        Kurtosis value.
    """

    if signal is None:
        raise TypeError("Please specify an input signal")

    return stats.kurtosis(signal, fisher=fisher)


def pSQI(signal, f_thr=0.01):
    """ Return the flatline percentage of the signal

    Parameters
    ----------
    signal : array
        ECG signal.
    f_thr : float, optional
        Flatline threshold, in mV / sample

    Returns
    -------
    flatline_percentage : float
        Percentage of signal where the absolute value of the derivative is lower then the threshold.

    """

    if signal is None:
        raise TypeError("Please specify an input signal")

    diff = np.diff(signal)
    length = len(diff)

    flatline = np.where(abs(diff) < f_thr)[0]

    return (len(flatline) / length) * 100


def fSQI(ecg_signal, fs=1000.0, nseg=1024, num_spectrum=[5, 20], dem_spectrum=None, mode='simple'):
    """ Returns the ration between two frequency power bands.

    Parameters
    ----------
    ecg_signal : array
        ECG signal.
    fs : float, optional
        ECG sampling frequency, in Hz.
    nseg : int, optional
        Frequency axis resolution.
    num_spectrum : array, optional
        Frequency bandwidth for the ratio's numerator, in Hz.
    dem_spectrum : array, optional
        Frequency bandwidth for the ratio's denominator, in Hz. If None, then the whole spectrum is used.
    mode : str, optional
        If 'simple' just do the ration, if is 'bas', then do 1 - num_power.

    Returns
    -------
    Ratio : float
        Ratio between two powerbands.
    """

    def power_in_range(f_range, f, Pxx_den):
        _indexes = np.where((f >= f_range[0]) & (f <= f_range[1]))[0]
        _power = integrate.trapz(Pxx_den[_indexes], f[_indexes])
        return _power

    if (ecg_signal is None):
        raise TypeError("Please specify an input signal")

    from scipy import signal, integrate
    import numpy as np

    f, Pxx_den = signal.welch(ecg_signal, fs, nperseg=nseg)
    num_power = power_in_range(num_spectrum, f, Pxx_den)

    if (dem_spectrum is None):
        dem_power = power_in_range([0, float(fs / 2.0)], f, Pxx_den)
    else:
        dem_power = power_in_range(dem_spectrum, f, Pxx_den)

    if (mode == 'simple'):
        return num_power / dem_power
    elif (mode == 'bas'):
        return 1 - num_power / dem_power


def ZZ2018(signal, detector_1, detector_2, fs=1000, search_window=100, nseg=1024, mode='simple'):
    import numpy as np
    """ Signal quality estimator. Designed for signal with a lenght of 10 seconds.
        Follows the approach by Zhao *et la.* [Zhao18]_.

    Parameters
    ----------
    signal : array
        Input ECG signal in mV.
    detector_1 : array
        Input of the first R peak detector.
    detector_2 : array
        Input of the second R peak detector.
    fs : int, float, optional
        Sampling frequency (Hz).
    search_window : int, optional
        Search window around each peak, in ms.
    nseg : int, optional
        Frequency axis resolution.
    mode : str, optional
        If 'simple', simple heurisitc. If 'fuzzy', employ a fuzzy classifier.

    Returns
    -------
    noise : str
        Quality classification.

    References
    ----------
    .. [Zhao18] Zhao, Z., & Zhang, Y. (2018).
    SQI quality evaluation mechanism of single-lead ECG signal based on simple heuristic fusion and fuzzy comprehensive evaluation.
    Frontiers in Physiology, 9, 727.
    """

    if (len(detector_1) == 0 or len(detector_2) == 0):
        return 'Unacceptable'

    ## compute indexes
    qsqi = bSQI(detector_1, detector_2, fs=fs, mode='matching', search_window=search_window)
    psqi = fSQI(signal, fs=fs, nseg=nseg, num_spectrum=[5, 15], dem_spectrum=[5, 40])
    ksqi = kSQI(signal)
    bassqi = fSQI(signal, fs=fs, nseg=nseg, num_spectrum=[0, 1], dem_spectrum=[0, 40], mode='bas')

    if mode == 'simple':
        ## First stage rules (0 = unqualified, 1 = suspicious, 2 = optimal)
        ## qSQI rules
        if qsqi > 0.90:
            qsqi_class = 2
        elif qsqi < 0.60:
            qsqi_class = 0
        else:
            qsqi_class = 1

        ## pSQI rules
        import numpy as np

        ## Get the maximum bpm
        if (len(detector_1) > 1):
            RR_max = 60000.0 / (1000.0 / fs * np.min(np.diff(detector_1)))
        else:
            RR_max = 1

        if RR_max < 130:
            l1, l2, l3 = 0.5, 0.8, 0.4
        else:
            l1, l2, l3 = 0.4, 0.7, 0.3

        if psqi > l1 and psqi < l2:
            pSQI_class = 2
        elif psqi > l3 and psqi < l1:
            pSQI_class = 1
        else:
            pSQI_class = 0

        ## kSQI rules
        if ksqi > 5:
            kSQI_class = 2
        else:
            kSQI_class = 0

        ## basSQI rules
        if bassqi >= 0.95:
            basSQI_class = 2
        elif bassqi < 0.9:
            basSQI_class = 0
        else:
            basSQI_class = 1

        class_matrix = np.array([qsqi_class, pSQI_class, kSQI_class, basSQI_class])
        n_optimal = len(np.where(class_matrix == 2)[0])
        n_suspics = len(np.where(class_matrix == 1)[0])
        n_unqualy = len(np.where(class_matrix == 0)[0])
        if n_unqualy >= 3 or (n_unqualy == 2 and n_suspics >= 1) or (n_unqualy == 1 and n_suspics == 3):
            return 'Unacceptable'
        elif n_optimal >= 3 and n_unqualy == 0:
            return 'Excellent'
        else:
            return 'Barely acceptable'

    elif mode == 'fuzzy':
        # Transform qSQI range from [0, 1] to [0, 100]
        qsqi = qsqi * 100.0
        # UqH (Excellent)
        if qsqi <= 80:
            UqH = 0
        elif qsqi >= 90:
            UqH = qsqi / 100.0
        else:
            UqH = 1.0 / (1 + (1 / np.power(0.3 * (qsqi - 80), 2)))

        # UqI (Barely acceptable)
        UqI = 1.0 / (1 + np.power((qsqi - 75) / 7.5, 2))

        # UqJ (unacceptable)
        if qsqi <= 55:
            UqJ = 1
        else:
            UqJ = 1.0 / (1 + np.power((qsqi - 55) / 5.0, 2))

        # Get R1
        R1 = np.array([UqH, UqI, UqJ])

        # pSQI
        # UpH
        if psqi <= 0.25:
            UpH = 0
        elif psqi >= 0.35:
            UpH = 1
        else:
            UpH = 0.1 * (psqi - 0.25)

        # UpI
        if psqi < 0.18:
            UpI = 0
        elif psqi >= 0.32:
            UpI = 0
        elif psqi >= 0.18 and psqi < 0.22:
            UpI = 25 * (psqi - 0.18)
        elif psqi >= 0.22 and psqi < 0.28:
            UpI = 1
        else:
            UpI = 25 * (0.32 - psqi)

        # UpJ
        if psqi < 0.15:
            UpJ = 1
        elif psqi > 0.25:
            UpJ = 0
        else:
            UpJ = 0.1 * (0.25 - psqi)

        # Get R2
        R2 = np.array([UpH, UpI, UpJ])

        # kSQI
        # Get R3
        if ksqi > 5:
            R3 = np.array([1, 0, 0])
        else:
            R3 = np.array([0, 0, 1])

        # basSQI
        # UbH
        if bassqi <= 90:
            UbH = 0
        elif bassqi >= 95:
            UbH = bassqi / 100.0
        else:
            UbH = 1.0 / (1 + (1 / np.power(0.8718 * (bassqi - 90), 2)))

        # UbI
        if bassqi <= 85:
            UbI = 1
        else:
            UbI = 1.0 / (1 + np.power((bassqi - 85) / 5.0, 2))

        # UbJ
        UbJ = 1.0 / (1 + np.power((bassqi - 95) / 2.5, 2))

        # R4
        R4 = np.array([UbH, UbI, UbJ])

        # evaluation matrix R
        R = np.vstack([R1, R2, R3, R4])

        # weight vector W
        W = np.array([0.4, 0.4, 0.1, 0.1])

        S = np.array([np.sum((R[:, 0] * W)), np.sum((R[:, 1] * W)), np.sum((R[:, 2] * W))])

        # classify
        V = np.sum(np.power(S, 2) * [1, 2, 3]) / np.sum(np.power(S, 2))

        if (V < 1.5):
            return 'Excellent'
        elif (V >= 2.40):
            return 'Unnacceptable'
        else:
            return 'Barely acceptable'