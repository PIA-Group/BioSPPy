# -*- coding: utf-8 -*-
"""
biosppy.signals.ecg
-------------------

This module provides methods to synthesize Electrocardiographic (ECG) signals.

:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
from math import pi

# 3rd party
import numpy as np
import biosppy.signals
import warnings
import matplotlib.pyplot as plt

# local
from biosppy.signals import tools as st
from .. import plotting, utils


def B(l, Kb):
    """Generates the amplitude values of the first isoelectric line (B segment) of the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameter introduced doesn't make sense in this context, an error will raise.
    Parameters
    ----------
    l  : float
        Inverse of the sampling rate.
    Kb : int
        B segment width (miliseconds).
    Returns
    -------
    B_segment : array
        B segment amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Kb > 130:
        raise Exception("Warning! Kb is out of boundaries.")
    else:
        a = np.zeros(Kb * l)
        B_segment = a.tolist()
    return B_segment


def P(i, Ap, Kp):
    """Generates the amplitude values of the P wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    Ap : int
        P wave amplitude (milivolts).
    Kp : int
        P wave width (miliseconds).
    Returns
    -------
    P_wave : array
        P wave amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Ap < -0.2 or Ap > 0.5:
        raise Exception("Warning! Ap is out of boundaries.")
    elif Kp < 10 or Kp > 100:
        raise Exception("Warning! Kp is out of boundaries.")
    else:
        k = np.arange(0, Kp, i)
        a = -(Ap / 2.0) * np.cos((2 * np.pi * k + 15) / Kp) + Ap / 2.0
        P_wave = a.tolist()
    return P_wave


def Pq(l, Kpq):
    """Generates the amplitude values of the PQ segment in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    l  : float
        Inverse of the sampling rate.
    Kpq : int
        PQ segment width (miliseconds).
    Returns
    -------
    PQ_segment : array
        PQ segment amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Kpq < 0 or Kpq > 60:
        raise Exception("Warning! Kpq is out of boundaries.")
    else:
        a = np.zeros(Kpq * l)
        PQ_segment = a.tolist()
    return PQ_segment


def Q1(i, Aq, Kq1):
    """Generates the amplitude values of the first 5/6 of the Q wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i   : int
        Sampling rate.
    Aq  : int
        Q wave amplitude (milivolts).
    Kq1 : int
        First 5/6 of the Q wave width (miliseconds).
    Returns
    -------
    Q1_wave : array
        First 5/6 of the Q wave amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Aq < 0 or Aq > 0.5:
        raise Exception("Warning! Aq is out of boundaries.")
    elif Kq1 < 0 or Kq1 > 70:
        raise Exception("Warning! Kq1 is out of boundaries.")
    else:
        k = np.arange(0, Kq1, i)
        a = -Aq * (k / Kq1)
        Q1_wave = a.tolist()
    return Q1_wave


def Q2(i, Aq, Kq2):
    """Generates the amplitude values of the last 1/6 of the Q wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i   : int
        Sampling rate.
    Aq  : int
        Q wave amplitude (milivolts).
    Kq2 : int
        Last 1/6 of the Q wave width (miliseconds).
    Returns
    -------
    Q2_wave : array
        Last 1/6 of the Q wave amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Aq < 0 or Aq > 0.5:
        raise Exception("Warning! Aq is out of boundaries.")
    elif Kq2 < 0 or Kq2 > 50:
        raise Exception("Warning! Kq2 is out of boundaries.")
    else:
        k = np.arange(0, Kq2, i)
        a = Aq * (k / Kq2) - Aq
        Q2_wave = a.tolist()
    return Q2_wave


def R(i, Ar, Kr):
    """Generates the amplitude values of the R wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    Ar : int
        R wave amplitude (milivolts).
    Kr : int
        R wave width (miliseconds).
    Returns
    -------
    R_wave : array
        R wave amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if Ar < 0.5 or Ar > 2:
        raise Exception("Warning! Ar is out of boundaries.")
    elif Kr < 10 or Kr > 150:
        raise Exception("Warning! Kr is out of boundaries.")
    else:
        k = np.arange(0, Kr, i)
        a = Ar * np.sin((np.pi * k) / Kr)
        R_wave = a.tolist()
    return R_wave


def S(i, As, Ks, Kcs, k=0):
    """Generates the amplitude values of the S wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    As : int
        S wave amplitude (milivolts).
    Ks : int
        S wave width (miliseconds).
    Kcs : int
        Parameter which allows slight adjustment of S wave shape by cutting away a portion at the end.
    k : int, optional
    Returns
    -------
    S : array
        If k = 0, S wave amplitude values (milivolts).
    S : int
        If k != 0, value obtained by using the S wave expression for the given k value.

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if As < 0 or As > 1:
        raise Exception("Warning! As is out of boundaries.")
    elif Ks < 10 or Ks > 200:
        raise Exception("Warning! Ks is out of boundaries.")
    elif Kcs < -5 or Kcs > 150:
        raise Exception("Warning! Kcs is out of boundaries.")
    else:
        if k == 0:
            k = np.arange(0, Ks - Kcs, i)
            a = (
                -As
                * i
                * k
                * (19.78 * np.pi)
                / Ks
                * np.exp(-2 * (((6 * np.pi) / Ks) * i * k) ** 2)
            )
            S = a.tolist()
        else:
            S = (
                -As
                * i
                * k
                * (19.78 * np.pi)
                / Ks
                * np.exp(-2 * (((6 * np.pi) / Ks) * i * k) ** 2)
            )
    return S


def St(i, As, Ks, Kcs, sm, Kst, k=0):
    """Generates the amplitude values of the ST segment in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    As : int
        S wave amplitude (milivolts).
    Ks : int
        S wave width (miliseconds).
    Kcs : int
        Parameter which allows slight adjustment of S wave shape by cutting away a portion at the end.
    sm : int
        Slope parameter in the ST segment.
    Kst : int
        ST segment width (miliseconds).
    k : int, optional
    Returns
    -------
    ST : array
        If k = 0, ST segment amplitude values (milivolts).
    ST : int
        If k != 0, value obtained by using the ST segment expression for the given k value.

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if sm < 1 or sm > 150:
        raise Exception("Warning! sm is out of boundaries.")
    elif Kst < 0 or Kst > 110:
        raise Exception("Warning! Kst is out of boundaries.")
    else:
        if k == 0:
            k = np.arange(0, Kst, i)
            a = -S(i, As, Ks, Kcs, Ks - Kcs) * (k / sm) + S(i, As, Ks, Kcs, Ks - Kcs)
            ST = a.tolist()
        else:
            ST = -S(i, As, Ks, Kcs, Ks - Kcs) * (k / sm) + S(i, As, Ks, Kcs, Ks - Kcs)
    return ST


def T(i, As, Ks, Kcs, sm, Kst, At, Kt, k=0):
    """Generates the amplitude values of the T wave in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    As : int
        S wave amplitude (milivolts).
    Ks : int
        S wave width (miliseconds).
    Kcs : int
        Parameter which allows slight adjustment of S wave shape by cutting away a portion at the end.
    sm : int
        Slope parameter in the ST segment.
    Kst : int
        ST segment width (miliseconds).
    At : int
        1/2 of the T wave amplitude (milivolts).
    Kt : int
        T wave width (miliseconds).
    k : int, optional
    Returns
    -------
    T : array
        If k = 0, T wave amplitude values (milivolts).
    T : int
        If k != 0, value obtained by using the T wave expression for the given k value.

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if At < -0.5 or At > 1:
        raise Exception("Warning! At is out of boundaries.")
    elif Kt < 50 or Kt > 300:
        raise Exception("Warning! Kt is out of boundaries.")
    else:
        if k == 0:
            k = np.arange(0, Kt, i)
            a = (
                -At * np.cos((1.48 * np.pi * k + 15) / Kt)
                + At
                + St(i, As, Ks, Kcs, sm, Kst, Kst)
            )
            T = a.tolist()
        else:
            T = (
                -At * np.cos((1.48 * np.pi * k + 15) / Kt)
                + At
                + St(i, As, Ks, Kcs, sm, Kst, Kst)
            )
    return T


def I(i, As, Ks, Kcs, sm, Kst, At, Kt, si, Ki):
    """Generates the amplitude values of the final isoelectric segment (I segment) in the ECG signal.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced don't make sense in this context, an error will raise.
    Parameters
    ----------
    i  : int
        Sampling rate.
    As : int
        S wave amplitude (milivolts).
    Ks : int
        S wave width (miliseconds).
    Kcs : int
        Parameter which allows slight adjustment of S wave shape by cutting away a portion at the end.
    sm : int
        Slope parameter in the ST segment.
    Kst : int
        ST segment width (miliseconds).
    At : int
        1/2 of the T wave amplitude (milivolts).
    Kt : int
        T wave width (miliseconds).
    si : int
        Parameter for setting the transition slope between T wave and isoelectric line.
    Ki : int
        I segment width (miliseconds).
    Returns
    -------
    I_segment : array
        I segment amplitude values (milivolts).

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    """
    if si < 0 or si > 50:
        raise Exception("Warning! si is out of boundaries.")
    else:
        k = np.arange(0, Ki, i)
        a = T(i, As, Ks, Kcs, sm, Kst, At, Kt, Kt) * (si / (k + 10))
        I_segment = a.tolist()
    return I_segment


def ecg(
    Kb=130,
    Ap=0.2,
    Kp=100,
    Kpq=40,
    Aq=0.1,
    Kq1=25,
    Kq2=5,
    Ar=0.7,
    Kr=40,
    As=0.2,
    Ks=30,
    Kcs=5,
    sm=96,
    Kst=100,
    At=0.15,
    Kt=220,
    si=2,
    Ki=200,
    var=0.01,
    sampling_rate=10000,
):  # normal values by default
    """Concatenates the segments and waves to make an ECG signal. The default values are physiological.

    Follows the approach by Dolinský, Andráš, Michaeli and Grimaldi [Model03].

    If the parameters introduced aren't within physiological values (limits based on the website [ECGwaves]), a warning will raise.

    Parameters
    ----------
    Kb : int, optional
        B segment width (miliseconds).
    Ap : float, optional
        P wave amplitude (milivolts).
    Kp : int, optional
        P wave width (miliseconds).
    Kpq : int, optional
        PQ segment width (miliseconds).
    Aq : float, optional
        Q wave amplitude (milivolts).
    Kq1 : int, optional
        First 5/6 of the Q wave width (miliseconds).
    Kq2 : int, optional
        Last 1/6 of the Q wave width (miliseconds).
    Ar : float, optional
        R wave amplitude (milivolts).
    Kr : int, optional
        R wave width (miliseconds).
    As : float, optional
        S wave amplitude (milivolts).
    Ks : int, optional
        S wave width (miliseconds).
    Kcs : int, optional
        Parameter which allows slight adjustment of S wave shape by cutting away a portion at the end.
    sm : int, optional
        Slope parameter in the ST segment.
    Kst : int, optional
        ST segment width (miliseconds).
    At : float, optional
        1/2 of the T wave amplitude (milivolts).
    Kt : int, optional
        T wave width (miliseconds).
    si : int, optional
        Parameter for setting the transition slope between T wave and isoelectric line.
    Ki : int, optional
        I segment width (miliseconds).
    var : float, optional
        Value between 0.0 and 1.0 that adds variability to the obtained signal, by changing each parameter following a normal distribution with mean value `parameter_value` and std `var * parameter_value`.
    sampling_rate : int, optional
        Sampling frequency (Hz).

    Returns
    -------
    ecg : array
        Amplitude values of the ECG wave.
    t : array
        Time values accoring to the provided sampling rate.
    params : dict
        Input parameters of the function


    Example
    -------
    sampling_rate = 10000
    beats = 3
    noise_amplitude = 0.05

    ECGtotal = np.array([])
    for i in range(beats):
        ECGwave, _, _ = ecg(sampling_rate=sampling_rate, var=0.1)
        ECGtotal = np.concatenate((ECGtotal, ECGwave))
    t = np.arange(0, len(ECGtotal)) / sampling_rate

    # add powerline noise (50 Hz)
    noise = noise_amplitude * np.sin(50 * (2 * pi) * t)
    ECGtotal += noise

    plt.plot(t, ECGtotal)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.title("ECG")

    plt.show()

    References
    ----------
    .. [Model03] Pavol DOLINSKÝ, Imrich ANDRÁŠ, Linus MICHAELI, Domenico GRIMALDI,
       "MODEL FOR GENERATING SIMPLE SYNTHETIC ECG SIGNALS",
       Acta Electrotechnica et Informatica, Vol. 18, No. 3, 2018, 3–8
    .. [ECGwaves] https://ecgwaves.com/
    """
    if Kp > 120 and Ap >= 0.25:
        warnings.warn("P wave isn't within physiological values.")

    if Kq1 + Kq2 > 30 or Aq > 0.25 * Ar:
        warnings.warn("Q wave isn't within physiological values.")

    if 120 > Kp + Kpq or Kp + Kpq > 220:
        warnings.warn("PR interval isn't within physiological limits.")

    if Kq1 + Kq2 + Kr + Ks - Kcs > 120:
        warnings.warn("QRS complex duration isn't within physiological limits.")

    if Kq1 + Kq2 + Kr + Ks - Kcs + Kst + Kt > 450:
        warnings.warn("QT segment duration isn't within physiological limits for men.")

    if Kq1 + Kq2 + Kr + Ks - Kcs + Kst + Kt > 470:
        warnings.warn(
            "QT segment duration isn't within physiological limits for women."
        )

    if var < 0 or var > 1:
        raise TypeError("Variability value should be between 0.0 and 1.0")

    if var > 0:
        # change the parameter according to the provided variability
        nd = lambda x: np.random.normal(x, x * var)
        Kb = round(np.clip(nd(Kb), 0, 130))
        Ap = np.clip(nd(Ap), -0.2, 0.5)
        Kp = np.clip(nd(Kp), 10, 100)
        Kpq = round(np.clip(nd(Kpq), 0, 60))
        Aq = np.clip(nd(Aq), 0, 0.5)
        Kq1 = round(np.clip(nd(Kq1), 0, 70))
        Kq2 = round(np.clip(nd(Kq2), 0, 50))
        Ar = np.clip(nd(Ar), 0.5, 2)
        Kr = round(np.clip(nd(Kr), 10, 150))
        As = np.clip(nd(As), 0, 1)
        Ks = round(np.clip(nd(Ks), 10, 200))
        Kcs = round(np.clip(nd(Kcs), -5, 150))
        sm = round(np.clip(nd(sm), 1, 150))
        Kst = round(np.clip(nd(Kst), 0, 110))
        At = np.clip(nd(At), -0.5, 1)
        Kt = round(np.clip(nd(Kt), 50, 300))
        si = round(np.clip(nd(si), 0, 50))

    # variable i is the time between samples (in miliseconds)
    i = 1000 / sampling_rate
    l = int(1 / i)

    B_to_S = (
        B(l, Kb)
        + P(i, Ap, Kp)
        + Pq(l, Kpq)
        + Q1(i, Aq, Kq1)
        + Q2(i, Aq, Kq2)
        + R(i, Ar, Kr)
        + S(i, As, Ks, Kcs)
    )
    St_to_I = (
        St(i, As, Ks, Kcs, sm, Kst)
        + T(i, As, Ks, Kcs, sm, Kst, At, Kt)
        + I(i, As, Ks, Kcs, sm, Kst, At, Kt, si, Ki)
    )

    # The signal is filtered in two different sizes
    ECG1_filtered, n1 = st.smoother(B_to_S, size=50)
    ECG2_filtered, n2 = st.smoother(St_to_I, size=500)

    # The signal is concatenated
    ECGwave = np.concatenate((ECG1_filtered, ECG2_filtered))

    # Time array
    t = np.arange(0, len(ECGwave)) / sampling_rate

    # output
    params = {
        "Kb": 130,
        "Ap": 0.2,
        "Kp": 100,
        "Kpq": 40,
        "Aq": 0.1,
        "Kq1": 25,
        "Kq2": 5,
        "Ar": 0.7,
        "Kr": 40,
        "As": 0.2,
        "Ks": 30,
        "Kcs": 5,
        "sm": 96,
        "Kst": 100,
        "At": 0.15,
        "Kt": 220,
        "si": 2,
        "Ki": 200,
        "var": 0.01,
        "sampling_rate": 10000,
    }

    args = (ECGwave, t, params)
    names = ("ecg", "t", "params")

    return utils.ReturnTuple(args, names)
