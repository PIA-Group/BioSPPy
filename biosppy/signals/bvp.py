# -*- coding: utf-8 -*-
"""
    biosppy.signals.bvp
    -------------------
    
    This module provides methods to process Blood Volume Pulse (BVP) signals.
    
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


def bvp(signal=None, sampling_rate=1000., show=True):
    """Process a raw BVP signal and extract relevant signal features using default parameters.
    
    Args:
        signal (array): Raw BVP signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
        
        show (bool): If True, show a summary plot (optional).
    
    Returns:
        (ReturnTuple): containing:
            ts (array): Signal time axis reference (seconds).
            
            filtered (array): Filtered BVP signal.
            
            onsets (array): Indices of BVP pulse onsets.
            
            heart_rate_ts (array): Heart rate time axis reference (seconds).
            
            heart_rate (array): Instantaneous heart rate (bpm).
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # ensure numpy
    signal = np.array(signal)
    
    sampling_rate = float(sampling_rate)
    
    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal, ftype='butter', band='lowpass',
                                      order=2, frequency=4, sampling_rate=sampling_rate)
    
    # find onsets
    onsets, = find_onsets(signal=filtered, sampling_rate=sampling_rate)
    
    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=onsets, sampling_rate=sampling_rate, smooth=True, size=3)
    
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    ts_hr = ts[hr_idx]
    
    # plot
    if show:
        plotting.plot_bvp(ts=ts, raw=signal, filtered=filtered,
                          onsets=onsets, heart_rate_ts=ts_hr,
                          heart_rate=hr, path=None, show=True)
    
    # output
    args = (ts, filtered, onsets, ts_hr, hr)
    names = ('ts', 'filtered', 'onsets', 'heart_rate_ts', 'heart_rate')
    
    return utils.ReturnTuple(args, names)


def find_onsets(signal=None, sampling_rate=1000.):
    """Determine onsets of BVP pulses.
    
    Skips corrupted signal parts.
    
    Args:
        signal (array): Input filtered BVP signal.
        
        sampling_rate (int, float): Sampling frequency (Hz).
    
    Returns:
        (ReturnTuple): containing:
            onsets (array): Indices of BVP pulse onsets.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # parameters
    sm_size = int(0.25 * sampling_rate)
    size = int(5 * sampling_rate)
    alpha = 2.
    wrange = int(0.1 * sampling_rate)
    d1_th = 0
    d2_th = 150
    
    length = len(signal)
    
    # slope sum function
    dy = np.diff(signal)
    dy[dy < 0] = 0
    
    ssf, _ = st.smoother(signal=dy, kernel='boxcar', size=sm_size, mirror=True)
    
    # main loop
    start = 0
    stop = size
    if stop > length:
        stop = length
    
    idx = []
    
    while True:
        sq = np.copy(signal[start:stop])
        sq -= sq.mean()
        # sq = sq[1:]
        ss = 25 * ssf[start:stop]
        sss = 100 * np.diff(ss)
        sss[sss < 0] = 0
        sss = sss - alpha * np.mean(sss)
        
        # find maxima
        pk, pv = st.find_extrema(signal=sss, mode='max')
        pk = pk[np.nonzero(pv > 0)]
        pk += wrange
        dpidx = pk
        
        # analyze between maxima of 2nd derivative of ss
        detected = False
        for i in xrange(1, len(dpidx) + 1):
            try:
                v, u = dpidx[i-1], dpidx[i]
            except IndexError:
                v, u = dpidx[-1], -1
            
            s = sq[v:u]
            Mk, Mv = st.find_extrema(signal=s, mode='max')
            mk, mv = st.find_extrema(signal=s, mode='min')
            
            try:
                M = Mk[np.argmax(Mv)]
                m = mk[np.argmax(mv)]
            except ValueError:
                continue
            
            if (s[M] - s[m] > d1_th) and (m - M > d2_th):
                idx += [v + start]
                detected = True
        
        # next round continues from previous detected beat
        if detected:
            start = idx[-1] + wrange
        else:
            start += size
        
        # stop condition
        if start > length:
            break
        
        # update stop
        stop += size
        if stop > length:
            stop = length
    
    idx = np.array(idx, dtype='int')
    
    return utils.ReturnTuple((idx, ), ('onsets', ))

