# -*- coding: utf-8 -*-
"""
biosppy.metrics
---------------

This module provides pairwise distance computation methods.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
import six

# 3rd party
import numpy as np
import scipy.spatial.distance as ssd
from scipy import linalg


def pcosine(u, v):
    """Computes the Cosine distance (positive space) between 1-D arrays.

    The Cosine distance (positive space) between `u` and `v` is defined as

    .. math::

        d(u, v) = 1 - abs \\left( \\frac{u \\cdot v}{||u||_2 ||v||_2} \\right)

    where :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    Parameters
    ----------
    u : array
        Input array.
    v : array
        Input array.

    Returns
    -------
    cosine : float
        Cosine distance between `u` and `v`.

    """

    # validate vectors like scipy does
    u = ssd._validate_vector(u)
    v = ssd._validate_vector(v)

    dist = 1. - np.abs(np.dot(u, v) / (linalg.norm(u) * linalg.norm(v)))

    return dist


def pdist(X, metric='euclidean', p=2, w=None, V=None, VI=None):
    """Pairwise distances between observations in n-dimensional space.

    Wraps scipy.spatial.distance.pdist.

    Parameters
    ----------
    X : array
        An m by n array of m original observations in an n-dimensional space.
    metric : str, function, optional
        The distance metric to use; the distance can be 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'pcosine', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    p : float, optional
        The p-norm to apply (for Minkowski, weighted and unweighted).
    w : array, optional
        The weight vector (for weighted Minkowski).
    V : array, optional
        The variance vector (for standardized Euclidean).
    VI : array, optional
        The inverse of the covariance matrix (for Mahalanobis).

    Returns
    -------
    Y : array
        Returns a condensed distance matrix Y.  For each :math:`i` and
        :math:`j` (where :math:`i<j<n`), the metric ``dist(u=X[i], v=X[j])``
        is computed and stored in entry ``ij``.

    """

    if isinstance(metric, six.string_types):
        if metric == 'pcosine':
            metric = pcosine

    return ssd.pdist(X, metric, p, w, V, VI)


def cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None, w=None):
    """Computes distance between each pair of the two collections of inputs.

    Wraps scipy.spatial.distance.cdist.

    Parameters
    ----------
    XA : array
        An :math:`m_A` by :math:`n` array of :math:`m_A` original observations
        in an :math:`n`-dimensional space.
    XB : array
        An :math:`m_B` by :math:`n` array of :math:`m_B` original observations
        in an :math:`n`-dimensional space.
    metric : str, function, optional
        The distance metric to use; the distance can be 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'pcosine', 'rogerstanimoto', 'russellrao',
        'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'.
    p : float, optional
        The p-norm to apply (for Minkowski, weighted and unweighted).
    w : array, optional
        The weight vector (for weighted Minkowski).
    V : array, optional
        The variance vector (for standardized Euclidean).
    VI : array, optional
        The inverse of the covariance matrix (for Mahalanobis).

    Returns
    -------
    Y : array
        An :math:`m_A` by :math:`m_B` distance matrix is returned. For each
        :math:`i` and :math:`j`, the metric ``dist(u=XA[i], v=XB[j])``
        is computed and stored in the :math:`ij` th entry.

    """

    if isinstance(metric, six.string_types):
        if metric == 'pcosine':
            metric = pcosine

    return ssd.cdist(XA, XB, metric, p, V, VI, w)


def squareform(X, force="no", checks=True):
    """Converts a vector-form distance vector to a square-form distance matrix,
    and vice-versa.

    Wraps scipy.spatial.distance.squareform.

    Parameters
    ----------
    X : array
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to 'tovector' or 'tomatrix', the
        input will be treated as a distance matrix or distance vector
        respectively.
    checks : bool, optional
        If `checks` is set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero. These values
        are ignored any way so they do not disrupt the squareform
        transformation.

    Returns
    -------
    Y : array
        If a condensed distance matrix is passed, a redundant one is returned,
        or if a redundant one is passed, a condensed distance matrix is
        returned.

    """

    return ssd.squareform(X, force, checks)
