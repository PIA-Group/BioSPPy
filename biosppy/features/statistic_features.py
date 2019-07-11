import numpy as np
from scipy.stats import stats
from .. import utils


def signal_stats(signal=None, hist=True):
    """Compute statistical metrics describing the signal.
        Parameters
        ----------
        signal : array
            Input signal.

        Returns
        -------
        mean : float
            Mean of the signal.
        median : float
            Median of the signal.
        max : float
            Maximum signal amplitude.
        var : float
            Signal variance (unbiased).
        std : float
            Standard signal deviation (unbiased).
        abs_dev : float
            Absolute signal deviation.
        kurtosis : float
            Signal kurtosis (unbiased).
        skew : float
            Signal skewness (unbiased).
        iqr : float
            Interquartile Range.
        meanadev : float
            Mean absolute deviation.
        medadev : float
            Mean absolute deviation.
        rms : float
            Root Mean Square.
        _hist : list
            Histogram.

        References
        ----------
        TSFEL library: https://github.com/fraunhoferportugal/tsfel
        Peeters, Geoffroy. (2004). A large set of audio features for sound description (similarity and classification) in the CUIDADO project.

        """

    # check inputs
    # if signal is None or signal == []:
    #     print("Signal is empty.")

    # ensure numpy
    signal = np.array(signal)

    # mean
    try:
        mean = np.mean(signal)
    except:
        mean = None

    # median
    try:
        median = np.median(signal)
    except:
        median = None

    # variance
    try:
        var = signal.var(ddof=1)
    except:
        var = None

    # standard deviation
    try:
        std = signal.std(ddof=1)
    except:
        std = None

    # absolute deviation
    try:
        ad = np.sum(np.abs(signal - median))
    except:
        ad = None

    # kurtosis
    try:
        kurt = stats.kurtosis(signal, bias=False)
    except:
        kurt = None

    # skweness
    try:
        skew = stats.skew(signal, bias=False)
    except:
        skew = None

    # interquartile range
    try:
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
    except:
        iqr = None

    # mean absolute deviation
    try:
        meanadev = np.mean([abs(x - mean) for x in signal])
    except:
        meanadev = None

    # median absolute deviation
    try:
        medadev = np.median([abs(x - median) for x in signal])
    except:
        medadev = None

    # root mean square
    try:
        rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
    except:
        rms = None

    # histogram
    if hist:
        try:
            _hist = list(np.histogram(signal, bins=int(np.sqrt(len(signal))), density=True)[0])
        except:
            _hist = [None] * int(np.sqrt(len(signal)))

    # output
    args = (mean, median, var, std, ad, kurt, skew, iqr, meanadev, medadev, rms)

    if hist:
        ar = list(args)
        ar += [i for i in _hist]
        args = tuple(ar)

    names = ('mean', 'median', 'var', 'std_dev', 'abs_dev', 'kurtosis',
             'skewness', 'interquartile_range', 'mean_abs_dev', 'median_abs_dev', 'rms')
    if hist:
        n = list(names)
        n += ['histbin_' + str(i) for i in range(len(_hist))]
        names = tuple(n)

    return utils.ReturnTuple(args, names)