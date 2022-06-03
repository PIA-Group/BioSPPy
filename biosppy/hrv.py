# -*- coding: utf-8 -*-
"""
biosppy.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics and visualization.
 

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# 3rd party
import numpy as np

# local


# use the same functions for ECG and BVP?
# build different functions or modules for short term and long term? or use a parameter?
# use different functions to compute each metric?



def hrv(rpeaks=None, npeaks=None, short_term=True, show=True):
    
    # returns an hrv report
    
    # check if recording is Short term or Long-term (can be overriden)
    # and call hrv functions to output the recommended data and plots
    
    # calls plot functions if show=True
    
    return None


def heart_rate():
    
    # HR
    # HR Max - HR Min

    return None


def hrv_timedomain(nni=None, rri=None, short_term=True):
    
    # SDNN
    # SDRR
    # SDANN
    # SDNNI
    # pNN50
    # RMSSD
    # HRV triangular index
    # TINN
    
    return None

def hrv_frequencydomain(nni=None, rri=None, method=None, short_term=True):
    
    # parameter to define method for frequency estimation? 
    # FFT / AR ?
    
    # ulf power (≤0.003 Hz)
    # VLF power (0.0033–0.04 Hz)
    # LF peak, rel power and absolute power (0.04–0.15 Hz)
    # HF peak, rel power and absolute power (0.15–0.4 Hz)
    # LF/HF ratio
    
    
    return None

def hrv_nonlinear(nni=None, rri=None, short_term=True):
    
    # S, SD1, SD2, SD1/SD2
    # ApEn
    # SampEn
    # DFA short term
    # DFA long-term
    # D2 or CD (correlation dimension?)
    
    return None
    

def plot_hr():
    
    # already exists
    
    return None

def plot_hrv():
    
    # use separate functions for each plot?
    
    # plot tachogram. If both available, plot rri and nni for comparison (?)
    # plot hrv histogram with triangle
    # plot poincaré
    # plot power spectrum with frequency bands (use available functions?)
    
    return None