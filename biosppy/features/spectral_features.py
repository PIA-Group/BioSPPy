import numpy as np
from .. import utils
from .. import tools as st


def spectral_roll(f, ff, cum_ff, TH):
    """ Frequency so TH % of the signal energy is below that value.

    Parameters
    ----------
    sign : ndarray
        Signal from which spectral slope is computed.
    fs : int
        Sampling frequency of the signal.
    Returns
    -------
    roll_off : float
        Spectral roll-off.

    References
    ----------
    TSFEL library: https://github.com/fraunhoferportugal/tsfel
    Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

    """
    output = None
    try:
        value = TH*(np.sum(ff))
        for i in range(len(ff)):
            if cum_ff[i] >= value:
                output = f[i]
                break
    except:
        output = None
    args = (output,)
    names = ('spectral_roll',)
    return utils.ReturnTuple(args, names)


def signal_spectral(signal, FS, hist=True):
    """Compute spectral metrics describing the signal.
        Parameters
        ----------
        signal : array
            Input signal.
        FS : float
            Sampling frequency

        Returns
        -------
        spectral_maxpeaks : int
            Number of peaks in the spectrum signal.

        spect_var : float
            Amount of the variation of the spectrum across time.

        curve_distance : float
            Euclidean distance between the cumulative sum of the signal spectrum and evenly spaced numbers across the signal lenght.

        spectral_roll_off : float
            Frequency so 95% of the signal energy is below that value.

        spectral_roll_on : float
            Frequency so 5% of the signal energy is below that value.

        spectral_dec : float
            Amount of decreasing in the spectral amplitude.

        spectral_slope : float
            Amount of decreasing in the spectral amplitude.

        spectral_centroid : float
            Centroid of the signal spectrum.

        spectral_spread : float
            Variance of the signal spectrum i.e. how it spreads around its mean value.

        spectral_kurtosis : float
            Kurtosis of the signal spectrum i.e. describes the flatness of the spectrum distribution.

        spectral_skewness : float
            Skewness of the signal spectrum i.e. describes the asymmetry of the spectrum distribution.

        max_frequency : float
            Maximum frequency of the signal spectrum maximum amplitude.

        fundamental_frequency : float
            Fundamental frequency of the signal.

        max_power_spectrum : float
            Spectrum maximum value.

        mean_power_spectrum : float
            Spectrum mean value.

        _hist : list
            Histogram of the signal spectrum.

        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        """
    # check inputs
    if signal is None or signal == []:
        print("Signal is empty.")

    # ensure numpy
    signal = np.array(signal)
    # f, spectrum = st.welch_spectrum(signal, sampling_rate=FS)
    f, spectrum = st.power_spectrum(signal, sampling_rate=FS)

    cum_ff = np.cumsum(spectrum)
    spect_diff = np.diff(spectrum)
    energy, _ = st.signal_energy(spectrum, f)[:]

    # spectral_maxpeaks
    try:
        spectral_maxpeaks = np.sum([1 for nd in range(len(spect_diff[:-1])) if (spect_diff[nd+1]<0 and spect_diff[nd]>0)])
    except:
        spectral_maxpeaks = None

    # spect_variation
    try:
        corr = np.convolve(energy)
        corr /= np.max(np.abs(corr))
    except:
        spect_var = None

    # curve_distance
    try:
        curve_distance = np.sum(np.linspace(0, cum_ff[-1], len(cum_ff)) - cum_ff)
    except:
        curve_distance = None

    # spectral_roll_off
    try:
        spectral_roll_off = spectral_roll(f, spectrum, cum_ff, 0.95)[0]
    except:
        spectral_roll_off = None

    # spectral_roll_on
    try:
        spectral_roll_on = spectral_roll(f, spectrum, cum_ff, 0.05)[0]
    except:
        spectral_roll_on = None

    # spectral_decrease
    try:
        spectral_dec = (1/np.sum(spectrum)) * np.sum((spectrum[:] - spectrum[1])/np.linspace(1, len(spectrum), len(spectrum),1))
    except:
        spectral_dec = None

    # spectral_slope
    sum_f = np.sum(f)
    len_f = len(f)
    try:
        spectral_slope = (len_f * np.dot(f, spectrum) - sum_f * np.sum(spectrum)) / (len_f * np.dot(f, f) - sum_f ** 2)
    except:
        spectral_slope = None

    # spectral_centroid
    sum_spectrum = np.sum(spectrum)
    norm_spectrum = None
    try:
        norm_spectrum = spectrum / sum_spectrum
        spectral_centroid = np.dot(f, spectrum/sum_spectrum)
    except:
        spectral_centroid = None

    # spectral_spread
    try:
        spectral_spread = np.dot(((f - spectral_centroid) ** 2), norm_spectrum)
    except:
        spectral_spread = None

    # spectral_kurtosis
    try:
        spectral_kurtosis = np.sum(((f - spectral_centroid) ** 4) * norm_spectrum) / (spectral_spread**2)
    except:
        spectral_kurtosis = None

    # spectral_skewness
    try:
        spectral_skewness = np.sum(((f - spectral_centroid) ** 3) * norm_spectrum) / (spectral_spread ** (3 / 2))
    except:
        spectral_skewness = None

    # max_frequency
    try:
        max_frequency = f[np.where(cum_ff > cum_ff[-1]*0.95)[0][0]]
    except:
        max_frequency = None

    # fundamental_frequency
    try:
        fundamental_frequency = f[np.where(cum_ff > cum_ff[-1]*0.5)[0][0]]
    except:
        fundamental_frequency = None

    # max_power_spectrum
    try:
        max_power_spectrum = np.max(spectrum)
    except:
        max_power_spectrum = None

    # mean_power_spectrum
    try:
        mean_power_spectrum = np.mean(spectrum)
    except:
        mean_power_spectrum = None

    # histogram
    if hist:
        try:
            _hist = list(np.histogram(spectrum, bins=int(np.sqrt(len(spectrum))), density=True)[0])
        except:
            _hist = [None] * int(np.sqrt(len(spectrum)))

    # output
    args = (
        spectral_maxpeaks, spect_var, curve_distance, spectral_roll_off, spectral_roll_on, spectral_dec, spectral_slope,
        spectral_kurtosis, spectral_skewness, spectral_spread, spectral_centroid, max_frequency, fundamental_frequency, max_power_spectrum, mean_power_spectrum)

    if hist:
        ar = list(args)
        ar += [i for i in _hist]
        args = tuple(ar)

    names = ('spectral_maxpeaks', 'spect_var', 'curve_distance', 'spectral_roll_off', 'spectral_roll_on', 'spectral_decrease',
             'spectral_slope', 'spectral_kurtosis', 'spectral_skewness', 'spectral_spread', 'spectral_centroid', 'max_frequency', 'fundamental_frequency', 'max_power_spectrum', 'mean_power_spectrum')

    if hist:
        n = list(names)
        n += ['spectral_histbin_' + str(i) for i in range(len(_hist))]
        names = tuple(n)

    return utils.ReturnTuple(args, names)
