import numpy as np
from scipy.stats import stats
from .. import utils
import json


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
        var : float
            Signal variance (unbiased).
        std : float
            Standard signal deviation (unbiased).
        abs_dev : float
            Absolute signal deviation.
        kurtosis : float
            Signal kurtosis (unbiased).
        skewness : float
            Signal skewness (unbiased).
        iqr : float
            Interquartile Range.
        meanadev : float
            Mean absolute deviation.
        medadev : float
            Median absolute deviation.
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
    dict = json.load(open('statistic_features_log.json'))
    args, names = [], []

    # mean
    try:
        mean = np.mean(signal)
    except:
        mean = None

    if dict['mean']['use'] == 'yes':
        args += [mean]
        names += ['mean']

    # median
    try:
        median = np.median(signal)
    except:
        median = None

    if dict['median']['use'] == 'yes':
        args += [median]
        names += ['median']

    if dict['var']['use'] == 'yes':
        # variance
        try:
            var = signal.var(ddof=1)
        except:
            var = None
        args += [var]
        names += ['var']

    if dict['std']['use'] == 'yes':
        # standard deviation
        try:
            std = signal.std(ddof=1)
        except:
            std = None
        args += [std]
        names += ['std']

    if dict['abs_dev']['use'] == 'yes':
        # absolute deviation
        try:
            abs_dev = np.sum(np.abs(signal - median))
        except:
            abs_dev = None
        args += [abs_dev]
        names += ['abs_dev']

    if dict['kurtosis']['use'] == 'yes':
        # kurtosis
        try:
            kurtosis = stats.kurtosis(signal, bias=False)
        except:
            kurtosis = None
        args += [kurtosis]
        names += ['kurtosis']

    if dict['skewness']['use'] == 'yes':
        # skweness
        try:
            skewness = stats.skew(signal, bias=False)
        except:
            skewness = None
        args += [skewness]
        names += ['skewness']

    if dict['iqr']['use'] == 'yes':
        # interquartile range
        try:
            iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        except:
            iqr = None
        args += [iqr]
        names += ['iqr']

    if dict['meanadev']['use'] == 'yes':
        # mean absolute deviation
        try:
            meanadev = np.mean([abs(x - mean) for x in signal])
        except:
            meanadev = None
        args += [meanadev]
        names += ['meanadev']

    if dict['medadev']['use'] == 'yes':
        # median absolute deviation
        try:
            medadev = np.median([abs(x - median) for x in signal])
        except:
            medadev = None
        args += [medadev]
        names += ['medadev']

    if dict['rms']['use'] == 'yes':
        # root mean square
        try:
            rms = np.sqrt(np.sum(np.array(signal) ** 2) / len(signal))
        except:
            rms = None
        args += [rms]
        names += ['rms']
    if hist:
        if dict['statistic_hist']['use'] == 'yes':
            # histogram
            try:
                _hist = list(np.histogram(signal, bins=int(np.sqrt(len(signal))//2))[0])
                _hist = _hist/np.sum(_hist)
            except:
                if len(signal) > 1:
                    _hist = [None] * int(np.sqrt(len(signal))//2)
                else:
                    _hist = [None]
            args += [i for i in _hist]
            names += ['statistic_hist' + str(i) for i in range(len(_hist))]

    return utils.ReturnTuple(tuple(args), tuple(names))
