import numpy as np
from .. import utils
import biosppy as bs
import json


def signal_temp(signal, FS):
    """Compute various metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.

    FS : float
        Sampling frequency
    Returns
    -------
    maxAmp : float
        Signal maximum amplitude.

    minAmp : float
        Signal minimum amplitude.

    max : float
        Signal max value.

    min : float
        Signal min value.

    dist : float
        Length of the signal.

    autocorr : float
        Signal autocorrelation.

    zero_cross : int
        Number of times the sinal crosses the zero axis.

    meanadiff : float
        Mean absolute differences.

    medadiff : float
        Median absolute differences.

    mindiff : float
        Min differences.

    maxdiff : float
        Maximum differences.

    sadiff : float
        Sum of absolute differences.

    meandiff : float
        Mean of differences.

    meddiff : float
        Median of differences.

    temp_centroid : float
        Temporal centroid.

    total_energy : float
        Total energy.

    minpeaks : int
        Number of minimum peaks.

    maxpeaks : int
        Number of maximum peaks.

    temp_dev : float
        Temporal deviation.

    counter : int
        Length of the signal.

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
    dict = json.load(open('temporal_features_log.json'))
    args, names = [], []
    # assert len(signal) > 1, 'Signal size < 1'
    try:
        sig_diff = np.diff(signal)
    except:
        sig_diff = []

    try:
        mean = np.mean(signal)
    except:
        mean = None

    if dict['maxAmp']['use'] == 'yes':
        # maximum amplitude
        try:
            maxAmp = np.max(np.abs(signal - mean))
        except:
            maxAmp = None
        args += [maxAmp]
        names += ['maxAmp']

    if dict['minAmp']['use'] == 'yes':
        # minimum amplitude
        try:
            minAmp = np.min(np.abs(signal - mean))
        except:
            minAmp = None
        args += [minAmp]
        names += ['minAmp']

    if dict['max']['use'] == 'yes':
        try:
            max = np.max(signal)
        except:
            max = None
        args += [max]
        names += ['max']

    if dict['min']['use'] == 'yes':
        try:
            min = np.min(signal)
        except:
            min = None
        args += [min]
        names += ['min']

    if dict['dist']['use'] == 'yes':
        # distance
        try:
            dist = np.sum([np.sqrt(1+df**2) for df in sig_diff])
        except:
            dist = None
        args += [dist]
        names += ['dist']

    if dict['autocorr']['use'] == 'yes':
        # autocorrelation
        try:
            autocorr = np.correlate(signal, signal)[0]
        except:
            autocorr = None
        args += [mean]
        names += ['autocorr']

    if dict['zero_cross']['use'] == 'yes':
        # zero_cross
        try:
            zero_cross = len(np.where(np.diff(np.sign(signal)))[0])
        except:
            zero_cross = None
        args += [zero_cross]
        names += ['zero_cross']

    if dict['meanadiff']['use'] == 'yes':
        # mean absolute differences
        try:
            meanadiff = np.mean(abs(sig_diff))
        except:
            meanadiff = None
        args += [meanadiff]
        names += ['meanadiff']

    if dict['medadiff']['use'] == 'yes':
        # median absolute differences
        try:
            medadiff = np.median(abs(sig_diff))
        except:
            medadiff = None
        args += [medadiff]
        names += ['medadiff']

    if dict['mindiff']['use'] == 'yes':
        # min absolute differences
        try:
            mindiff = np.min(np.abs(sig_diff))
        except:
            mindiff = None
        args += [mindiff]
        names += ['mindiff']

    if dict['maxdiff']['use'] == 'yes':
        # max absolute differences
        try:
            maxdiff = np.max(np.abs(sig_diff))
        except:
            maxdiff = None
        args += [maxdiff]
        names += ['maxdiff']

    if dict['sadiff']['use'] == 'yes':
        # sum of absolute differences
        try:
            sadiff = np.sum(abs(sig_diff))
        except:
            sadiff = None
        args += [sadiff]
        names += ['sadiff']

    if dict['meandiff']['use'] == 'yes':
        # mean of differences
        try:
            meandiff = np.mean(sig_diff)
        except:
            meandiff = None
        args += [meandiff]
        names += ['meandiff']

    if dict['meddiff']['use'] == 'yes':
        # median of differences
        try:
            meddiff = np.median(sig_diff)
        except:
            meddiff = None
        args += [meddiff]
        names += ['meddiff']

    try:
        time = range(len(signal))
        time = [float(x) / FS for x in time]
    except:
        time = []

    try:
        energy, time_energy = bs.signal_energy(signal, time)
    except:
        energy, time_energy = [], []

    if dict['temp_centroid']['use'] == 'yes':
        # temporal centroid
        try:
            temp_centroid = np.dot(np.array(time_energy), np.array(energy)) / np.sum(energy)
        except:
            temp_centroid = None
        args += [temp_centroid]
        names += ['temp_centroid']

    if dict['total_energy']['use'] == 'yes':
        # total energy
        try:
            total_energy = np.sum(np.array(signal)**2)/(time[-1]-time[0])
        except:
            total_energy = None
        args += [total_energy]
        names += ['total_energy']

    if dict['minpeaks']['use'] == 'yes':
        # number of minimum peaks
        try:
            minpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd]<0 and sig_diff[nd+1]>0)])
        except:
            minpeaks = None
        args += [minpeaks]
        names += ['minpeaks']

    if dict['maxpeaks']['use'] == 'yes':
        # number of maximum peaks
        try:
            maxpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd+1]<0 and sig_diff[nd]>0)])
        except:
            maxpeaks = None
        args += [maxpeaks]
        names += ['maxpeaks']

    if dict['temp_dev']['use'] == 'yes':
        # temporal deviation
        try:
            temp_dev = (1/np.sum(signal)) * np.sum((signal[:] - signal[1])/np.array(time))
        except:
            temp_dev = None
        args += [temp_dev]
        names += ['temp_dev']

    if dict['counter']['use'] == 'yes':
        # counter
        try:
            counter = len(signal)
        except:
            counter = None
        args += [counter]
        names += ['counter']

    # output
    return utils.ReturnTuple(tuple(args), tuple(names))
