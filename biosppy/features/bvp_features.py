import numpy as np
from .. import utils
from .. import bvp
from .. import tools as st


def bvp_features(signal=None, sampling_rate=1000.):
    """ Compute BVP characteristic metrics describing the signal.

    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : float
        Sampling frequency.
    Returns
    -------
    ons : list
        Signal onsets.

    hr: list
        Bvp heart rate.
    """

    # ensure numpy array
    signal = np.array(signal)

    # onsets
    try:
        ons = bvp.find_onsets(signal, sampling_rate)['onsets']
    except:
        ons = None

    # heart rate
    try:
        _, hr = st.get_heart_rate(beats=ons, sampling_rate=sampling_rate, smooth=True, size=3)
    except:
        hr = None

    # output
    args = (ons, hr)
    names = ('onsets', 'hr')

    return utils.ReturnTuple(args, names)
