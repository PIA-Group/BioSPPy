# -*- coding: utf-8 -*-
"""
biosppy.signals
---------------

This package provides methods to process common
physiological signals (biosignals):
    * Photoplethysmogram (PPG)
    * Electrocardiogram (ECG)
    * Electrodermal Activity (EDA)
    * Electroencephalogram (EEG)
    * Electromyogram (EMG)
    * Respiration (Resp)

:copyright: (c) 2015-2021 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# compat
from __future__ import absolute_import, division, print_function

# allow lazy loading
from . import acc, abp, bvp, pcg, ppg, ecg, eda, eeg, emg, resp, tools
