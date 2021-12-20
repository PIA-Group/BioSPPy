# -*- coding: utf-8 -*-
"""
biosppy
-------

A toolbox for biosignal processing written in Python.

:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# compat
from __future__ import absolute_import, division, print_function

# get version
from .__version__ import __version__

# allow lazy loading
from .signals import acc, abp, bvp, ppg, pcg, ecg, eda, eeg, emg, resp, tools
from .synthesizers import ecg
