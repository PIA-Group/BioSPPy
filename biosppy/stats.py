# -*- coding: utf-8 -*-
"""
biosppy.stats
---------------

This module provides statistica functions and related tools.

:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
import six

# local

from . import utils

# 3rd party
import numpy as np
import matplotlib.pyplot as plt


def pearson_correlation(x=None, y=None):
    """Compute the Pearson Correlation Coefficient between two signals.

    The coefficient is given by:

    .. math::

        r_{xy} = \\frac{E[(X - \\mu_X) (Y - \\mu_Y)]}{\\sigma_X \\sigma_Y}

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    rxy : float
        Pearson correlation coefficient, ranging between -1 and +1.

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    mx = np.mean(x)
    my = np.mean(y)

    Sxy = np.sum(x * y) - n * mx * my
    Sxx = np.sum(np.power(x, 2)) - n * mx ** 2
    Syy = np.sum(np.power(y, 2)) - n * my ** 2

    rxy = Sxy / (np.sqrt(Sxx) * np.sqrt(Syy))

    if not np.isnan(rxy):
        # avoid propagation of numerical errors
        if rxy > 1.0:
            rxy = 1.0
        elif rxy < -1.0:
            rxy = -1.0

    return rxy


def linear_regression(x=None, y=None):
    """Plot the linear regression between two signals and get the equation coefficients.

    The linear regression uses the least squares method.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    coeffs : array
        Linear regression coefficients: [m, b].

    Raises
    ------
    ValueError
        If the input signals do not have the same length.

    """

    # check inputs
    if x is None:
        raise TypeError("Please specify the first input signal.")

    if y is None:
        raise TypeError("Please specify the second input signal.")

    # ensure numpy
    x = np.array(x)
    y = np.array(y)

    n = len(x)

    if n != len(y):
        raise ValueError("Input signals must have the same length.")

    coeffs = np.polyfit(x, y, 1)

    x_min = x.min()
    x_max = x.max()

    y_x_min = coeffs[0] * x_min + coeffs[1]
    y_x_max = coeffs[0] * x_max + coeffs[1]

    plt.scatter(x, y)
    plt.plot(
        [x_min, x_max],
        [y_x_min, y_x_max],
        c="orange",
        label="y={:.3f}x+{:.3f}".format(coeffs[0], coeffs[1]),
    )
    plt.title("Linear Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    return coeffs
