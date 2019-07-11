import numpy as np
from .. import utils
import biosppy as bs


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

    min : float
        Signal min value.

    dist : float
        Length of the signal.

    autocorr : float
        Signal autocorrelation

    zero_cross : int
        Number of times the sinal crosses the zero axis

    meanadiff : float
        Mean absolute differences

    medadiff : float
        Median absolute differences

    mindiff : float
        Min absolute differences

    maxdiff : float
        Maximum absolute differences

    sadiff : float
        Sum of absolute differences

    meandiff : float
        Mean of differences

    meddiff : float
        Median of differences

    temp_centroid : float
        Temporal centroid

    total_energy : float
        Total energy.

    minpeaks : int
        Number of minimum peaks

    maxpeaks : int
        Number of maximum peaks

    temp_dev : float
        Temporal deviation

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
    # assert len(signal) > 1, 'Signal size < 1'
    try:
        sig_diff = np.diff(signal)
    except:
        sig_diff = []

    try:
        mean = np.mean(signal)
    except:
        mean = None

    # maximum amplitude
    try:
        maxAmp = np.max(np.abs(signal - mean))
    except:
        maxAmp = None

    # minimum amplitude
    try:
        minAmp = np.min(np.abs(signal - mean))
    except:
        minAmp = None

    try:
        max = np.max(signal)
    except:
        max = None

    try:
        min = np.min(signal)
    except:
        min = None

    # distance
    try:
        dist = np.sum([np.sqrt(1+df**2) for df in sig_diff])
    except:
        dist = None

    # autocorrelation
    try:
        autocorr = np.correlate(signal, signal)[0]
    except:
        autocorr = None

    # zero_cross
    try:
        zero_cross = len(np.where(np.diff(np.sign(signal)))[0])
    except:
        zero_cross = None

    # mean absolute differences
    try:
        meanadiff = np.mean(abs(sig_diff))
    except:
        meanadiff = None

    # median absolute differences
    try:
        medadiff = np.median(abs(sig_diff))
    except:
        medadiff = None

    # min absolute differences
    try:
        mindiff = np.min(np.abs(sig_diff))
    except:
        mindiff = None

    # max absolute differences
    try:
        maxdiff = np.max(np.abs(sig_diff))
    except:
        maxdiff = None

    # sum of absolute differences
    try:
        sadiff = np.sum(abs(sig_diff))
    except:
        sadiff = None

    # mean of differences
    try:
        meandiff = np.mean(sig_diff)
    except:
        meandiff = None

    # median of differences
    try:
        meddiff = np.median(sig_diff)
    except:
        meddiff = None

    try:
        time = range(len(signal))
        time = [float(x) / FS for x in time]
    except:
        time = []

    try:
        energy, time_energy = bs.signal_energy(signal, time)
    except:
        energy, time_energy = [], []

    # temporal centroid
    try:
        temp_centroid = np.dot(np.array(time_energy), np.array(energy)) / np.sum(energy)
    except:
        temp_centroid = None

    # total energy
    try:
        total_energy = np.sum(np.array(signal)**2)/(time[-1]-time[0])
    except:
        total_energy = None

    # number of minimum peaks
    try:
        minpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd]<0 and sig_diff[nd+1]>0)])
    except:
        minpeaks = None

    # number of maximum peaks
    try:
        maxpeaks = np.sum([1 for nd in range(len(sig_diff[:-1])) if (sig_diff[nd+1]<0 and sig_diff[nd]>0)])
    except:
        maxpeaks = None

    # temporal deviation
    try:
        temp_dev = (1/np.sum(signal)) * np.sum((signal[:] - signal[1])/np.array(time))
    except:
        temp_dev = None

    # counter
    try:
        counter = len(signal)
    except:
        counter = None

    # output
    args = (maxAmp, minAmp, max, min, dist, autocorr, zero_cross, meanadiff, medadiff, mindiff, maxdiff, sadiff, meandiff, meddiff, temp_centroid, total_energy, minpeaks, maxpeaks, temp_dev, counter)
    names = ('maxAmp', 'minAmp', 'max', 'min', 'distance', 'autocorr', 'zero_cross', 'meanadiff', 'medadiff', 'mindiff', 'maxdiff', 'sadiff', 'meandiff', 'meddiff', 'temp_centroid', 'total_energy', 'minpeaks', 'maxpeaks', 'temp_dev', 'counter')
    return utils.ReturnTuple(args, names)
