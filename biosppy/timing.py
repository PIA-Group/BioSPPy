# -*- coding: utf-8 -*-
"""
biosppy.timing
--------------

This module provides simple methods to measure computation times.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
# from six.moves import map, range, zip
# import six

# built-in
import time

# 3rd party

# local

# Globals
CLOCKS = dict()
DFC = '__default_clock__'


def tic(name=None):
    """Start the clock.
    
    Parameters
    ----------
    name : str, optional
        Name of the clock; if None, uses the default name.
    
    """
    
    if name is None:
        name = DFC
    
    CLOCKS[name] = time.time()


def tac(name=None):
    """Stop the clock.
    
    Parameters
    ----------
    name : str, optional
        Name of the clock; if None, uses the default name.
    
    Returns
    -------
    delta : float
        Elapsed time, in seconds.
    
    Raises
    ------
    KeyError if the name of the clock is unknown.
    
    """
    
    toc = time.time()
    
    if name is None:
        name = DFC
    
    try:
        delta = toc - CLOCKS[name]
    except KeyError:
        raise KeyError('Unknown clock.')
    
    return delta


def clear(name=None):
    """Clear the clock.
    
    Parameters
    ----------
    name : str, optional
        Name of the clock; if None, uses the default name.
    
    """
    
    if name is None:
        name = DFC
    
    CLOCKS.pop(name)


def clear_all():
    """Clear all clocks."""
    
    CLOCKS.clear()
