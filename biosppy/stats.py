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
from scipy.stats import pearsonr, ttest_rel, ttest_ind


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
    r : float
        Pearson correlation coefficient, ranging between -1 and +1.
    pvalue : float
        Two-tailed p-value. The p-value roughly indicates the probability of
        an uncorrelated system producing datasets that have a Pearson correlation
        at least as extreme as the one computed from these datasets.

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

    r, pvalue = pearsonr(x, y)

    return r, pvalue


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
    f = np.poly1d(coeffs)

    x_min = x.min()
    x_max = x.max()

    y_min = f(x_min)
    y_max = f(x_max)

    plt.scatter(x, y)
    plt.plot(
        [x_min, x_max],
        [y_min, y_max],
        c="orange",
        label="y={:.3f}x+{:.3f}".format(coeffs[0], coeffs[1]),
    )
    plt.title("Linear Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    return coeffs


def paired_test(x=None, y=None):
    """
    Perform the Student's paired t-test on the arrays x and y.
    This is a two-sided test for the null hypothesis that 2 related
    or repeated samples have identical average (expected) values.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    statistic : float
        t-statistic. The t-statistic is used in a t-test to determine
        if you should support or reject the null hypothesis.
    pvalue : float
        Two-sided p-value.

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

    statistic, pvalue = ttest_rel(x, y)

    return statistic, pvalue


def unpaired_test(x=None, y=None):
    """
    Perform the Student's unpaired t-test on the arrays x and y.
    This is a two-sided test for the null hypothesis that 2 independent
    samples have identical average (expected) values. This test assumes
    that the populations have identical variances by default.

    Parameters
    ----------
    x : array
        First input signal.
    y : array
        Second input signal.

    Returns
    -------
    statistic : float
        t-statistic. The t-statistic is used in a t-test to determine
        if you should support or reject the null hypothesis.
    pvalue : float
        Two-sided p-value.

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

    statistic, pvalue = ttest_ind(x, y)

    return statistic, pvalue
