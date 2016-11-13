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

# Lazy load
from .signals.ecg import *
from .signals.eda import *
from .signals.eeg import *
from .signals.emg import *
from .signals.rsp import *
