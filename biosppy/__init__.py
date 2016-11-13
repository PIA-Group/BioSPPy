# -*- coding: utf-8 -*-
"""
    biosppy
    -------

    A toolbox for biosignal processing written in Python.

    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# get version
from .version import version as __version__

# Allow lazy loading
from biosppy.signals import ecg
from biosppy.signals import eda
from biosppy.signals import eeg
from biosppy.signals import emg
from biosppy.signals import bvp
from biosppy.signals import resp
