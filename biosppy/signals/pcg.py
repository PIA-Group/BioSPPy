# -*- coding: utf-8 -*-
"""
biosppy.signals.pcg
-------------------

This module provides methods to process Phonocardiography (PCG) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import scipy.signal as ss

# local
from . import tools as st
from .. import plotting, utils


def pcg(signal=None, sampling_rate=1000., path=None, show=True):
    """

    Parameters
    ----------
    signal : array
        Raw PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered PCG signal.
    peaks : array
        Peak location indices.
    hs: array
        Classification of peaks as S1 or S2.
    heart_rate : double
        Average heart rate (bpm).
    systolic_time_interval : double
        Average systolic time interval (seconds).
    heart_rate_ts : array
         Heart rate time axis reference (seconds).
    inst_heart_rate : array
        Instantaneous heart rate (bpm).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # Filter Design
    order = 2
    passBand = np.array([25, 400])
    
    # Band-Pass filtering of the PCG:        
    filtered,fs,params = st.filter_signal(signal,'butter','bandpass',order,passBand,sampling_rate)

    # find peaks
    peaks,envelope = find_peaks(signal=filtered, sampling_rate=sampling_rate)
    
    # classify heart sounds
    hs, = identify_heart_sounds(beats=peaks, sampling_rate=sampling_rate)
    s1_peaks = peaks[np.where(hs==1)[0]]
    
    # get heart rate
    heartRate,systolicTimeInterval = get_avg_heart_rate(envelope,sampling_rate)
    
    # get instantaneous heart rate
    hr_idx,hr = st.get_heart_rate(s1_peaks, sampling_rate)
    
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]

    # plot
    if show:
        plotting.plot_pcg(ts=ts,
                raw=signal,
                filtered=filtered,
                peaks=peaks,
                heart_sounds=hs,
                heart_rate_ts=ts_hr,
                inst_heart_rate=hr,
                path=path,
                show=True)
        
        
    # output
    args = (ts, filtered, peaks, hs, heartRate, systolicTimeInterval, ts_hr, hr)
    names = ('ts', 'filtered', 'peaks', 'heart_sounds',
             'heart_rate', 'systolic_time_interval','heart_rate_ts','inst_heart_rate')

    return utils.ReturnTuple(args, names)

def find_peaks(signal=None,sampling_rate=1000.):
    
    """Finds the peaks of the heart sounds from the homomorphic envelope

    Parameters
    ----------
    signal : array
        Input filtered PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    peaks : array
        peak location indices.
    envelope : array
        Homomorphic envelope (normalized).

    """
    
    # Compute homomorphic envelope
    envelope, = homomorphic_filter(signal,sampling_rate)
    envelope, = st.normalize(envelope)
    
    # Find the prominent peaks of the envelope
    peaksIndices, _ = ss.find_peaks(envelope, height = 0.2 * np.amax(envelope), distance = 0.10*sampling_rate, prominence = 0.25)
    
    peaks = np.array(peaksIndices, dtype='int')

    return utils.ReturnTuple((peaks,envelope), 
                             ('peaks','homomorphic_envelope'))

def homomorphic_filter(signal=None, sampling_rate=1000.):
    
    """Finds the homomorphic envelope of a signal

    Follows the approach described by Schmidt et al. [Schimdt10]_.

    Parameters
    ----------
    signal : array
        Input filtered PCG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    envelope : array
        Homomorphic envelope (non-normalized).

    References
    ----------
    .. [Schimdt10] S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
       duration-dependent hidden Markov model", Physiol. Meas., 2010

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    sampling_rate = float(sampling_rate)
        
    # LP-filter Design (to reject the oscillating component of the signal):
    order = 1; fc = 8
    sos = ss.butter(order, fc, btype = 'low', analog = False, output = 'sos', fs = sampling_rate) 
    envelope = np.exp( ss.sosfiltfilt(sos, np.log(np.abs(signal))))    

    return utils.ReturnTuple((envelope,), 
                             ('homomorphic_envelope',))

def get_avg_heart_rate(envelope=None, sampling_rate=1000.):
    
    """Compute average heart rate from the signal's homomorphic envelope.
    
    Follows the approach described by Schmidt et al. [Schimdt10]_, with
    code adapted from David Springer [Springer16]_.
    
    Parameters
    ----------
    envelope : array
        Signal's homomorphic envelope
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
        
    Returns
    -------
    heart_rate : double
        Average heart rate (bpm).
    systolic_time_interval : double
        Average systolic time interval (seconds).

    Notes
    -----
    * Assumes normal human heart rate to be between 40 and 200 bpm.
    * Assumes normal human systole time interval to be between 0.2 seconds and half a heartbeat
    
    References
    ----------
    .. [Schimdt10] S. E. Schmidt et al., "Segmentation of heart sound recordings by a 
       duration-dependent hidden Markov model", Physiol. Meas., 2010
    .. [Springer16] D.Springer, "Heart sound segmentation code based on duration-dependant
       HMM", 2016. Available at: https://github.com/davidspringer/Springer-Segmentation-Code   
    
    """

    # check inputs
    if envelope is None:
        raise TypeError("Please specify the signal's homomorphic envelope.")
    
    autocorrelation = np.correlate(envelope,envelope,mode='full')
    autocorrelation = autocorrelation[(autocorrelation.size)//2:]
    
    min_index = int(0.3*sampling_rate)
    max_index = int(1.5*sampling_rate)

    index = np.argmax(autocorrelation[min_index-1:max_index-1])
    true_index = index+min_index-1
    heartRate = 60/(true_index/sampling_rate)
    
    max_sys_duration = int(np.round(((60/heartRate)*sampling_rate)/2))
    min_sys_duration = int(np.round(0.2*sampling_rate))
    
    pos = np.argmax(autocorrelation[min_sys_duration-1:max_sys_duration-1])
    systolicTimeInterval = (min_sys_duration+pos)/sampling_rate
    

    return utils.ReturnTuple((heartRate,systolicTimeInterval),
                             ('heart_rate','systolic_time_interval'))

def identify_heart_sounds(beats = None, sampling_rate = 1000.):
    
    """Classify heart sound peaks as S1 or S2
     
    Parameters
    ----------
    beats : array
        Peaks of heart sounds
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
        
    Returns
    -------
    classification : array
        Classification of heart sound peaks. 1 is S1, 2 is S2
    
    """

    one_peak_ahead = np.roll(beats, -1)

    SS_intervals = (one_peak_ahead[0:-1] - beats[0:-1]) / sampling_rate
    
    # Initialize the vector to store the classification of the peaks:
    classification = np.zeros(len(beats))
        
    # Classify the peaks. 
    # Terrible algorithm, but good enough for now    
    for i in range(1,len(beats)-1):
        if SS_intervals[i-1] > SS_intervals[i]:
            classification[i] = 0
        else:
            classification[i] = 1
    classification[0] = int(not(classification[1]))
    classification[-1] = int(not(classification[-2]))    
    
    classification += 1    
        
    return utils.ReturnTuple((classification,), ('heart_sounds',))