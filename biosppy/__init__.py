# -*- coding: utf-8 -*-
"""
biosppy
-------

A toolbox for biosignal processing written in Python.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# compat
from __future__ import absolute_import, division, print_function

# get version
from .__version__ import __version__

# allow lazy loading
from .signals import bvp, ecg, eda, eeg, emg, resp, tools
from .features import feature_vector, temporal_features, statistic_features
from .classification import dissimilarity_based, supervised_learning
