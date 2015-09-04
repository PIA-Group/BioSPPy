# -*- coding: utf-8 -*-
"""
    biosppy.signals.tools
    ---------------------

    This module provides various signal analysis methods in the time and
    frequency domains.

    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in

# 3rd party
import numpy as np
import scipy.signal as ss
from scipy import interpolate, optimize
from scipy.stats import stats

# local
from .. import utils

# Globals


def _norm_freq(frequency=None, sampling_rate=1000.):
    """Normalize frequency to Nyquist Frequency (Fs/2).

    Args:
        frequency (int, float, list, array): Frequencies to normalize.

        sampling_rate (int, float): Sampling frequency (Hz).

    Returns:
        wn (float, array): Normalized frequencies.

    """

    # check inputs
    if frequency is None:
        raise TypeError("Please specify a frequency to normalize.")

    # convert inputs to correct representation
    try:
        frequency = float(frequency)
    except TypeError:
        # maybe frequency is a list or array
        frequency = np.array(frequency, dtype='float')

    Fs = float(sampling_rate)

    wn = 2. * frequency / Fs

    return wn


def _filter_init(b, a, alpha=1.):
    """Get an initial filter state that corresponds to the steady-state
    of the step response.

    Args:
        b (array): Numerator coefficients.

        a (array): Denominator coefficients.

        alpha (float): Scaling factor.

    Returns:
        zi (array): Initial filter state.

    """

    zi = alpha * ss.lfilter_zi(b, a)

    return zi


def _filter_signal(b, a, signal, zi=None, check_phase=True, **kwargs):
    """Filter a signal with given coefficients.

    Args:
        b (array): Numerator coefficients.

        a (array): Denominator coefficients.

        signal (array): Signal to filter.

        zi (array): Initial filter state (optional).

        check_phase (bool): If True, use the forward-backward technique (optional).

        **kwargs (dict): Additional keyword arguments are passed to the
            underlying filtering function.

    Returns:
        (tulpe): containing:
            filtered (array): Filtered signal.

            zf (array): Final filter state.

    Notes:
        * If check_phase is True, zi cannot be set.

    """

    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True.")

    if check_phase:
        filtered = ss.filtfilt(b, a, signal, **kwargs)
        zf = None
    else:
        filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)

    return filtered, zf


def _filter_resp(b, a, sampling_rate=1000., nfreqs=512):
    """Compute the filter frequency response.

    Args:
        b (array): Numerator coefficients.

        a (array): Denominator coefficients.

        sampling_rate (int, float): Sampling frequency (Hz).

        nfreqs (int): Number of frequency points to compute.

    Returns:
        (tulpe): containing:
            freqs (array): Array of frequencies (Hz) at which the response
                was computed.

            resp (array): Frequency response.

    """

    w, resp = ss.freqz(b, a, nfreqs)

    # convert frequencies
    freqs = w * sampling_rate / (2. * np.pi)

    return freqs, resp


def _get_window(kernel, size, **kwargs):
    """Return a window with the specified parameters.

    Args:
        kernel (str): Type of window to create.

        size(int): Size of the window.

        **kwargs (dict): Additional keyword arguments are passed to the
            underlying scipy.signal.windows function.

    Returns:
        window (array): Created window.

    """

    # mimics scipy.signal.get_window
    if kernel in ['blackman', 'black', 'blk']:
        winfunc = ss.blackman
    elif kernel in ['triangle', 'triang', 'tri']:
        winfunc = ss.triang
    elif kernel in ['hamming', 'hamm', 'ham']:
        winfunc = ss.hamming
    elif kernel in ['bartlett', 'bart', 'brt']:
        winfunc = ss.bartlett
    elif kernel in ['hanning', 'hann', 'han']:
        winfunc = ss.hann
    elif kernel in ['blackmanharris', 'blackharr', 'bkh']:
        winfunc = ss.blackmanharris
    elif kernel in ['parzen', 'parz', 'par']:
        winfunc = ss.parzen
    elif kernel in ['bohman', 'bman', 'bmn']:
        winfunc = ss.bohman
    elif kernel in ['nuttall', 'nutl', 'nut']:
        winfunc = ss.nuttall
    elif kernel in ['barthann', 'brthan', 'bth']:
        winfunc = ss.barthann
    elif kernel in ['flattop', 'flat', 'flt']:
        winfunc = ss.flattop
    elif kernel in ['kaiser', 'ksr']:
        winfunc = ss.kaiser
    elif kernel in ['gaussian', 'gauss', 'gss']:
        winfunc = ss.gaussian
    elif kernel in ['general gaussian', 'general_gaussian', 'general gauss',
                    'general_gauss', 'ggs']:
        winfunc = ss.general_gaussian
    elif kernel in ['boxcar', 'box', 'ones', 'rect', 'rectangular']:
        winfunc = ss.boxcar
    elif kernel in ['slepian', 'slep', 'optimal', 'dpss', 'dss']:
        winfunc = ss.slepian
    elif kernel in ['cosine', 'halfcosine']:
        winfunc = ss.cosine
    elif kernel in ['chebwin', 'cheb']:
        winfunc = ss.chebwin
    else:
        raise ValueError("Unknown window type.")

    try:
        window = winfunc(size, **kwargs)
    except TypeError as e:
        raise TypeError("Invalid window arguments: %s." % e)

    return window


def get_filter(ftype='FIR',
               band='lowpass',
               order=None,
               frequency=None,
               sampling_rate=1000., **kwargs):
    """Compute digital (FIR or IIR) filter coefficients with the given
    parameters.

    Args:
        ftype (str): Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').

        band (str): Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').

        order (int): Order of the filter.

        frequency (int, float, list, array): Cutoff frequencies; format depends
            on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.

        sampling_rate (int, float): Sampling frequency (Hz).

        **kwargs (dict): Additional keyword arguments are passed to the
            underlying scipy.signal function.

    Returns:
        (ReturnTuple): containing:
            b (array): Numerator coefficients.

            a (array): Denominator coefficients.

    See Also:
        scipy.signal

    """

    # check inputs
    if order is None:
        raise TypeError("Please specify the filter order.")
    if frequency is None:
        raise TypeError("Please specify the cutoff frequency.")
    if band not in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
        raise ValueError(
            "Unknown filter type '%r'; choose 'lowpass', 'highpass', \
            'bandpass', or 'bandstop'."
            % band)

    # convert frequencies
    frequency = _norm_freq(frequency, sampling_rate)

    # get coeffs
    b, a = [], []
    if ftype == 'FIR':
        # FIR filter
        if order % 2 == 0:
            order += 1
        a = np.array([1])
        if band in ['lowpass', 'bandstop']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=True, **kwargs)
        elif band in ['highpass', 'bandpass']:
            b = ss.firwin(numtaps=order,
                          cutoff=frequency,
                          pass_zero=False, **kwargs)
    elif ftype == 'butter':
        # Butterworth filter
        b, a = ss.butter(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby1':
        # Chebyshev type I filter
        b, a = ss.cheby1(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'cheby2':
        # chevyshev type II filter
        b, a = ss.cheby2(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)
    elif ftype == 'ellip':
        # Elliptic filter
        b, a = ss.ellip(N=order,
                        Wn=frequency,
                        btype=band,
                        analog=False,
                        output='ba', **kwargs)
    elif ftype == 'bessel':
        # Bessel filter
        b, a = ss.bessel(N=order,
                         Wn=frequency,
                         btype=band,
                         analog=False,
                         output='ba', **kwargs)

    return utils.ReturnTuple((b, a), ('b', 'a'))


def filter_signal(signal=None,
                  ftype='FIR',
                  band='lowpass',
                  order=None,
                  frequency=None,
                  sampling_rate=1000., **kwargs):
    """Filter a signal according to the given parameters.

    Args:
        signal (array): Signal to filter.

        ftype (str): Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').

        band (str): Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').

        order (int): Order of the filter.

        frequency (int, float, list, array): Cutoff frequencies; format depends
            on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.

        sampling_rate (int, float): Sampling frequency (Hz).

        **kwargs (dict): Additional keyword arguments are passed to the
            underlying scipy.signal function.

    Returns:
        (ReturnTuple): containing:
            signal (array): Filtered signal.

            sampling_rate (float): Sampling frequency (Hz).

            params (dict): Filter parameters.

    Notes:
        * Uses a forward-backward filter implementation. Therefore, the
          combined filter has linear phase.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to filter.")

    # get filter
    b, a = get_filter(ftype=ftype,
                      order=order,
                      frequency=frequency,
                      sampling_rate=sampling_rate,
                      band=band, **kwargs)

    # filter
    filtered, _ = _filter_signal(b, a, signal, check_phase=True)

    # output
    params = {
        'ftype': ftype,
        'order': order,
        'frequency': frequency,
        'band': band,
    }
    params.update(kwargs)

    args = (filtered, sampling_rate, params)
    names = ('signal', 'sampling_rate', 'params')

    return utils.ReturnTuple(args, names)


def smoother(signal=None, kernel='boxzen', size=10, mirror=True, **kwargs):
    """Smooth a signal using an N-point moving average filter.

    This implementation uses the convolution of a filter kernel with the input
    signal to compute the smoothed signal.

    Availabel kernels: median, boxzen, boxcar, triang, blackman, hamming, hann,
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    kaiser (needs beta), gaussian (needs std), general_gaussian (needs power,
    width), slepian (needs width), chebwin (needs attenuation).

    Args:
        signal (array): Signal to smooth.

        kernel (str, array): Type of kernel to use; if array, use directly
            as the kernel.

        size (int): Size of the kernel; ignored if kernel is an array.

        mirror (bool): If True, signal edges are extended to avoid
            boundary effects (optional).

        **kwargs (dict): Additional keyword arguments are passed to the
            underlying scipy.signal.windows function.

    Returns:
        (ReturnTuple): containing:
            signal (array): Smoothed signal.

            params (dict): Smoother parameters.

    Notes:
        * When the kernel is 'median', mirror is ignored.

    References:
        [1] Wikipedia, "Moving Average". http://en.wikipedia.org/wiki/Moving_average
        [2] S. W. Smith, "Moving Average Filters - Implementation by Convolution",
            http://www.dspguide.com/ch15/1.htm

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to smooth.")

    length = len(signal)

    if isinstance(kernel, basestring):
        # check length
        if size > length:
            size = length - 1

        if kernel == 'boxzen':
            # hybrid method
            # 1st pass - boxcar kernel
            aux, _ = smoother(signal,
                              kernel='boxcar',
                              size=size,
                              mirror=mirror)

            # 2nd pass - parzen kernel
            smoothed, _ = smoother(aux,
                                   kernel='parzen',
                                   size=size,
                                   mirror=mirror)

            params = {'kernel': kernel, 'size': size, 'mirror': mirror}

            args = (smoothed, params)
            names = ('signal', 'params')

            return utils.ReturnTuple(args, names)

        elif kernel == 'median':
            # median filter
            if size % 2 == 0:
                raise ValueError(
                    "When the kernel is 'median', size must be odd.")

            smoothed = ss.medfilt(signal, kernel_size=size)

            params = {'kernel': kernel, 'size': size, 'mirror': mirror}

            args = (smoothed, params)
            names = ('signal', 'params')

            return utils.ReturnTuple(args, names)

        else:
            win = _get_window(kernel, size, **kwargs)

    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)

        # check length
        if size > length:
            raise ValueError("Kernel size is bigger than signal length.")

    else:
        raise TypeError("Unknown kernel type.")

    # convolve
    w = win / win.sum()
    if mirror:
        aux = np.concatenate(
            (signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
        smoothed = np.convolve(w, aux, mode='same')
        smoothed = smoothed[size:-size]
    else:
        smoothed = np.convolve(w, signal, mode='same')

    # output
    params = {'kernel': kernel, 'size': size, 'mirror': mirror}
    params.update(kwargs)

    args = (smoothed, params)
    names = ('signal', 'params')

    return utils.ReturnTuple(args, names)


def analytic_signal(signal=None, N=None):
    """Compute analytic signal, using the Hilbert Transform.

    Args:
        signal (array): Input signal.

        N (int): Number of Fourier components; default is len(signal) (optional).

    Returns:
        (ReturnTuple): containing:
            amplitude (array): Amplitude envelope of the analytic signal.

            phase (array): Instantaneous phase component of the analystic signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # hilter transform
    asig = ss.hilbert(signal, N=N)

    # amplitude envelope
    amp = np.absolute(asig)

    # instantaneous
    phase = np.angle(asig)

    return utils.ReturnTuple((amp, phase), ('amplitude', 'phase'))


def phase_locking(signal1=None, signal2=None, N=None):
    """Compute the Phase-Locking Factor (PLF) between two signals.

    Args:
        signal1 (array): First input signal.

        signal2 (array): Second input signal.

        N (int): Number of Fourier components (optional).

    Returns:
        (ReturnTuple): containing:
            plf (float): The PLF between the two signals.

    """

    # check inputs
    if signal1 is None:
        raise TypeError("Please specify the first input signal.")

    if signal2 is None:
        raise TypeError("Please specify the second input signal.")

    if len(signal1) != len(signal2):
        raise ValueError("The input signals must have the same length.")

    # compute analytic signal
    asig1 = ss.hilbert(signal1, N=N)
    phase1 = np.angle(asig1)

    asig2 = ss.hilbert(signal2, N=N)
    phase2 = np.angle(asig2)

    # compute PLF
    plf = np.absolute(np.mean(np.exp(1j * (phase1 - phase2))))

    return utils.ReturnTuple((plf,), ('plf',))


def power_spectrum(signal=None,
                   sampling_rate=1000.,
                   pad=None,
                   pow2=False,
                   decibel=True):
    """Compute the power spectrum of a signal (one-sided).

    Args:
        signal (array): Input signal.

        sampling_rate (int, float): Sampling frequency (Hz).

        pad (int): Padding for the Fourier Transform (number of zeros added) (optional).

        pow2 (bool): If True, rounds the number of points
            'N = len(signal) + pad' to the nearest power of 2 greater
            than N (optional).

        decibel (bool): If True, return the power in decibels (optional).

    Returns:
        (ReturnTuple): containing:
            freqs (array): Array of frequencies (Hz) at which the power
                was computed.

            power (array): Power spectrum.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    npoints = len(signal)

    if pad is not None:
        if pad >= 0:
            npoints += pad
        else:
            raise ValueError("Padding must be a positive integer.")

    # power of 2
    if pow2:
        npoints = 2 ** (np.ceil(np.log2(npoints)))

    Nyq = float(sampling_rate) / 2
    hpoints = npoints / 2

    freqs = np.linspace(0, Nyq, hpoints)
    power = np.abs(np.fft.fft(signal, npoints)) / npoints

    # one-sided
    power = power[:hpoints]
    power[1:] *= 2
    power = np.power(power, 2)

    if decibel:
        power = 10. * np.log10(power)

    return utils.ReturnTuple((freqs, power), ('freqs', 'power'))


def band_power(freqs=None, power=None, frequency=None, decibel=True):
    """Compute the avearge power in a frequency band.

    Args:
        freqs (array): Array of frequencies (Hz) at which the power
            was computed.

        power (array): Input power spectrum.

        frequency (list, array): Pair of frequencies defining the band.

        decibel (bool): If True, input powerspectrum is in decibels.

    Returns:
        (ReturnTuple): containing:
            avg_power (float): The average power in the band.

    """

    # check inputs
    if freqs is None:
        raise TypeError("Please specify the 'freqs' array.")

    if power is None:
        raise TypeError("Please specify the input power spectrum.")

    if len(freqs) != len(power):
        raise ValueError(
            "The input 'freqs' and 'power' arrays must have the same length.")

    if frequency is None:
        raise TypeError("Please specify the band frequencies.")

    try:
        f1, f2 = frequency
    except ValueError:
        raise ValueError("Input 'frequency' must be a pair of frequencies.")

    # make frequencies sane
    if f1 > f2:
        f1, f2 = f2, f1

    if f1 < freqs[0]:
        f1 = freqs[0]
    if f2 > freqs[-1]:
        f2 = freqs[-1]

    # average
    sel = np.nonzero(np.logical_and(f1 <= freqs, freqs <= f2))[0]

    if decibel:
        aux = 10 ** (power / 10.)
        avg = np.mean(aux[sel])
        avg = 10. * np.log10(avg)
    else:
        avg = np.mean(power[sel])

    return utils.ReturnTuple((avg,), ('avg_power',))


def signal_stats(signal=None):
    """Compute various metrics describing the signal.

    Args:
        signal (array): Input signal.

    Returns:
        (ReturnTuple): containing:
            mean (float): Mean of the signal.

            median (float): Median of the signal.

            max (float): Maximum signal amplitude.

            var (float): Signal variance (unbiased).

            std_dev (float): Standard signal deviation (unbiased).

            abs_dev (float): Absolute signal deviation.

            kurtosis (float): Signal kurtosis (unbiased).

            skew (float): Signal skewness (unbiased).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    # mean
    mean = np.mean(signal)

    # median
    median = np.median(signal)

    # maximum amplitude
    maxAmp = np.abs(signal - mean).max()

    # variance
    sigma2 = signal.var(ddof=1)

    # standard deviation
    sigma = signal.std(ddof=1)

    # absolute deviation
    ad = np.sum(np.abs(signal - median))

    # kurtosis
    kurt = stats.kurtosis(signal, bias=False)

    # skweness
    skew = stats.skew(signal, bias=False)

    # output
    args = (mean, median, maxAmp, sigma2, sigma, ad, kurt, skew)
    names = ('mean', 'median', 'max', 'var', 'std_dev', 'abs_dev', 'kurtosis',
             'skewness')

    return utils.ReturnTuple(args, names)


def normalize(signal=None):
    """Normalize a signal.

    Args:
        signal (array): Input signal.

    Returns:
        (ReturnTuple): containing:
            normalized (array): Normalized signal.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    normalized = signal - signal.mean()
    normalized /= normalized.std(ddof=1)

    return utils.ReturnTuple((normalized,), ('normalized',))


def zero_cross(signal=None, detrend=False):
    """Locate the indices where the signal crosses zero.

    Args:
        signal (array): Input signal.

        detrend (bool): If True, remove signal mean before computation.

    Returns:
        (ReturnTuple): containing:
            zeros (array): Indices of zero crossings.

    Notes:
        * When the signal crosses zero between samples, the first index
          is returned.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if detrend:
        signal = signal - np.mean(signal)

    # zeros
    df = np.diff(np.sign(signal))
    zeros = np.nonzero(np.abs(df) > 0)[0]

    return utils.ReturnTuple((zeros,), ('zeros',))


def find_extrema(signal=None, mode='both'):
    """Locate local extrema points in a signal.

    Based on Fermat's Theorem.

    Args:
        signal (array): Input signal.

        mode (str): Whether to find maxima ('max'), minima ('min'),
            or both ('both').

    Returns:
        (ReturnTuple): containing:
            extrema (array): Indices of the extrama points.

            values (array): Signal values at the extrema points.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if mode not in ['max', 'min', 'both']:
        raise ValueError("Unknwon mode %r." % mode)

    aux = np.diff(np.sign(np.diff(signal)))

    if mode == 'both':
        aux = np.abs(aux)
        extrema = np.nonzero(aux > 0)[0] + 1
    elif mode == 'max':
        extrema = np.nonzero(aux < 0)[0] + 1
    elif mode == 'min':
        extrema = np.nonzero(aux > 0)[0] + 1

    values = signal[extrema]

    return utils.ReturnTuple((extrema, values), ('extrema', 'values'))


def windower(signal=None,
             size=None,
             step=None,
             fcn=None,
             fcn_kwargs=None,
             kernel='boxcar',
             kernel_kwargs=None):
    """Apply a function to a signal in sequential windows, with optional overlap.

    Availabel window kernels: boxcar, triang, blackman, hamming, hann,
    bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
    kaiser (needs beta), gaussian (needs std), general_gaussian (needs power,
    width), slepian (needs width), chebwin (needs attenuation).

    Args:
        signal (array): Input signal.

        size (int): Size of the signal window.

        step (int): Size of window shift; if None, there is no overlap (optional).

        fcn (callable): Function to apply to each window.

        fcn_kwargs (dict): Additional keyword arguments to pass to 'fcn' (optional).

        kernel (str, array): Type of kernel to use; if array, use directly
            as the kernel.

        kernel_kwargs (dict): Additional keyword arguments to pass on window
            creation; ignored if 'kernel' is an array.

    Returns:
        (ReturnTuple): containing:
            index (array): Indices characterizing window locations
                (start of the window).

            values (array): Concatenated output of calling 'fcn' on each window.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if fcn is None:
        raise TypeError("Please specify a function to apply to each window.")

    if kernel_kwargs is None:
        kernel_kwargs = {}

    length = len(signal)

    if isinstance(kernel, basestring):
        # check size
        if size > length:
            raise ValueError("Window size must be smaller than signal length.")

        win = _get_window(kernel, size, **kernel_kwargs)
    elif isinstance(kernel, np.ndarray):
        win = kernel
        size = len(win)

        # check size
        if size > length:
            raise ValueError("Window size must be smaller than signal length.")

    if step is None:
        step = size

    if step <= 0:
        raise ValueError("Step size must be at least 1.")

    # number of windows
    nb = 1 + (length - size) / step

    # check signal dimensionality
    if np.ndim(signal) == 2:
        # time along 1st dim, tile window
        nch = np.shape(signal)[1]
        win = np.tile(np.reshape(win, (size, 1)), nch)

    index = []
    values = []
    for i in xrange(nb):
        start = i * step
        stop = start + size
        index.append(start)

        aux = signal[start:stop] * win

        # apply function
        out = fcn(aux, **fcn_kwargs)
        values.append(out)

    # transform to numpy
    index = np.array(index, dtype='int')
    values = np.array(values)

    return utils.ReturnTuple((index, values), ('index', 'values'))


def synchronize(signal1=None, signal2=None):
    """Align two signals based on cross-correlation.

    Args:
        signal1 (array): First input signal.

        signal2 (array): Second input signal.

    Returns:
        (ReturnTuple): containing:
            delay (int): Delay (number of samples) of 'signal1' in relation to
                'signal2'; if 'delay' < 0 , 'signal1' is ahead in relation
                to 'signal2'; if 'delay' > 0 , 'signal1' is delayed in relation
                to 'signal2'.

            corr (float): Value of maximum correlation.

            synch1 (array): Biggest possible portion of 'signal1' in
                synchronization.

            synch2 (array): Biggest possible portion of 'signal2' in
                synchronization.

    """

    # check inputs
    if signal1 is None:
        raise TypeError("Please specify the first input signal.")

    if signal2 is None:
        raise TypeError("Please specify the second input signal.")

    n1 = len(signal1)
    n2 = len(signal2)

    # correlate
    corr = np.correlate(signal1, signal2, mode='full')
    x = np.arange(-n2 + 1, n1, dtype='int')
    ind = np.argmax(corr)

    delay = x[ind]
    maxCorr = corr[ind]

    # get synchronization overlap
    if delay < 0:
        c = min([n1, len(signal2[-delay:])])
        synch1 = signal1[:c]
        synch2 = signal2[-delay:-delay + c]
    elif delay > 0:
        c = min([n2, len(signal1[delay:])])
        synch1 = signal1[delay:delay + c]
        synch2 = signal2[:c]
    else:
        c = min([n1, n2])
        synch1 = signal1[:c]
        synch2 = signal2[:c]

    # output
    args = (delay, maxCorr, synch1, synch2)
    names = ('delay', 'corr', 'synch1', 'synch2')

    return utils.ReturnTuple(args, names)


def get_heart_rate(beats=None, sampling_rate=1000., smooth=False, size=3):
    """Compute instantaneous heart rate from an array of beat indices.

    Args:
        beats (array): Beat location indices.

        sampling_rate (int, float): Sampling frequency (Hz).

        smooth (bool): If True, perform smoothing on the resulting heart rate (optional).

        size (int): Size of smoothing window; ignored if 'smooth' is False (optional).

    Returns:
        (ReturnTuple): containing:
            index (array): Heart rate location indices.

            heart_rate (array): Instantaneous heart rate (bpm).

    Notes:
        * Assumes normal human heart rate to be between 40 and 190 bpm.

    """

    # check inputs
    if beats is None:
        raise TypeError("Please specify the input beat indices.")

    if len(beats) < 2:
        raise ValueError("Not enough beats to compute heart rate.")

    # compute heart rate
    ts = beats[1:]
    hr = sampling_rate * (60. / np.diff(beats))

    # physiological limits
    indx = np.nonzero(np.logical_and(hr >= 40, hr <= 200))
    ts = ts[indx]
    hr = hr[indx]

    # smooth with moving average
    if smooth:
        size = 5
        hr, _ = smoother(signal=hr, kernel='boxcar', size=size, mirror=True)

    return utils.ReturnTuple((ts, hr), ('index', 'heart_rate'))


def _pdiff(x, p1, p2):
    """Compute the squared difference between two interpolators, given the
        x-coordinates.

    Args:
        x (array): Array of x-coordinates.

        p1 (object): First interpolator.

        p2 (object): Second interpolator.

    Returns:
        diff (array): Squared differences.

    """

    diff = (p1(x) - p2(x)) ** 2

    return diff


def find_intersection(x1=None,
                      y1=None,
                      x2=None,
                      y2=None,
                      alpha=1.5,
                      xtol=1e-6,
                      ytol=1e-6):
    """Find the intersection points between two lines using piecewise
    polynomial interpolation.

    Args:
        x1 (array): Array of x-coordinates of the first line.

        y1 (array): Array of y-coordinates of the first line.

        x2 (array): Array of x-coordinates of the second line.

        y2 (array): Array of y-coordinates of the second line.

        alpha (float): Resolution factor for the x-axis; fraction of total
            number of x-coordinates (optional).

        xtol (float): Tolerance for the x-axis (optional).

        ytol (float): Tolerance for the y-axis (optional).

    Returns:
        (ReturnTuple): containing:
            roots (array): Array of x-coordinates of found intersection points.

            values (array): Array of y-coordinates of found intersection points.

    Notes:
        * If no intersection is found, returns the closest point.

    """

    # check inputs
    if x1 is None:
        raise TypeError("Please specify the x-coordinates of the first line.")
    if y1 is None:
        raise TypeError("Please specify the y-coordinates of the first line.")
    if x2 is None:
        raise TypeError("Please specify the x-coordinates of the second line.")
    if y2 is None:
        raise TypeError("Please specify the y-coordinates of the second line.")

    # ensure numpy
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)

    if x1.shape != y1.shape:
        raise ValueError(
            "Input coordinates for the first line must have the same shape.")
    if x2.shape != y2.shape:
        raise ValueError(
            "Input coordinates for the second line must have the same shape.")

    # interpolate
    p1 = interpolate.BPoly.from_derivatives(x1, y1[:, np.newaxis])
    p2 = interpolate.BPoly.from_derivatives(x2, y2[:, np.newaxis])

    # combine x intervals
    x = np.r_[x1, x2]
    x_min = x.min()
    x_max = x.max()
    npoints = int(len(np.unique(x)) * alpha)
    x = np.linspace(x_min, x_max, npoints)

    # initial estimates
    pd = p1(x) - p2(x)
    zerocs, = zero_cross(pd)

    pd_abs = np.abs(pd)
    zeros = np.nonzero(pd_abs < ytol)[0]

    ind = np.unique(np.concatenate((zerocs, zeros)))
    xi = x[ind]

    # search for solutions
    roots = set()
    for v in xi:
        root, _, ier, _ = optimize.fsolve(_pdiff, v, (p1, p2),
                                          full_output=True,
                                          xtol=xtol)
        if ier == 1 and x_min <= root <= x_max:
            roots.add(root[0])

    if len(roots) == 0:
        # no solution was found => give the best from the initial estimates
        aux = np.abs(pd)
        bux = aux.min() * np.ones(npoints, dtype='float')
        roots, _ = find_intersection(x, aux, x, bux,
                                     alpha=1.,
                                     xtol=xtol,
                                     ytol=ytol)

    # compute values
    roots = list(roots)
    roots.sort()
    roots = np.array(roots)
    values = np.mean(np.vstack((p1(roots), p2(roots))), axis=0)

    return utils.ReturnTuple((roots, values), ('roots', 'values'))
