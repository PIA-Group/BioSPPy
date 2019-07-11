import numpy as np
import pyhrv
from .. import utils


def nonlinear_geo_features(flag, signal):
    """Compute non-linear and geometric characteristic metrics describing the signal.

    Parameters
    ----------
    sampling_rate : flag
        Events location indices.
    signal : array
        Input signal.
    Returns
    -------  sd1, sd2, sd12, poincarea, sample_entropy, dfa_alpha1, dfa_alpha2, tinn_n, tinn_m, tinn, triangular_index
    sd1 : float
        Standard deviation of the major axis in the Poincaré Plot.

    sd2 : float
        Standard deviation of the minor axis in the Poincaré Plot.

    sd12 : float
        Ratio between SD2 and SD1 (SD2/SD1).

    poincarea : float
        Area of the Poincaré Plot fitted ellipse.

    sample_entropy : float
        Sample entropy of the NNI series.

    dfa_alpha1 : float
        Alpha value of the short term Detrended Fluctuation Analysis of an NNI series.

    dfa_alpha2 : float
        Alpha value of the long term Detrended Fluctuation Analysis of an NNI series.

    tinn_n : float
        N value of the TINN computation.

    tinn_m : float
        M value of the TINN computation.

    tinn : float
        Baseline width of the NNI histogram based on the triangular Interpolation (TINN).

    triangular_index : float
        Ratio between the total number of NNIs and the maximum of the NNI histogram distribution.

    References
    ----------
    Gomes, Pedro & Margaritoff, Petra & Plácido da Silva, Hugo. (2019). pyHRV: Development and Evaluation of an Open-Source Python Toolbox for Heart Rate Variability (HRV).

    """
    signal = np.array(signal)
    try:
        flag_int = signal[pyhrv.tools.nn_intervals(flag)].astype(np.float)
    except:
        flag_int = None

    # Non-Linear features
    try:
        _, sd1, sd2, sd12, poincarea = pyhrv.nonlinear.poincare(flag_int, show=False, plot=False, legend=False)[:]
    except:
        sd1, sd2, sd12, poincarea = None, None, None, None

    try:
        sample_entropy = pyhrv.nonlinear.sample_entropy(flag_int)[0]
    except:
        sample_entropy = None

    try:
        _, dfa_alpha1, dfa_alpha2 = pyhrv.nonlinear.dfa(flag_int, show=False, legend=False)[:]
    except:
        dfa_alpha1, dfa_alpha2 = None, None

    # Geometrical features
    try:
        tinn = pyhrv.time_domain.tinn(nni=flag_int, show=False, legend=False, plot=False)[0]
        tinn_n = tinn['tinn_n']
        tinn_m = tinn['tinn_m']
        tinn = tinn['tinn']
    except:
        tinn_n, tinn_m, tinn = None, None, None

    try:
        triangular_index = pyhrv.time_domain.triangular_index(flag_int, show=False, plot=False, legend=False)[0]
    except:
        triangular_index = None

    # output
    args = (sd1, sd2, sd12, poincarea, sample_entropy, dfa_alpha1, dfa_alpha2, tinn_n, tinn_m, tinn, triangular_index)

    names = ('sd1', 'sd2', 'sd12', 'poincarea', 'sample_entropy', 'dfa_alpha1', 'dfa_alpha2', 'tinn_n', 'tinn_m', 'tinn', 'triangular_index')

    return utils.ReturnTuple(args, names)
    