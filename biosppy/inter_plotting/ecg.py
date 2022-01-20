# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides an interactive display option for the ECG plot.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
from matplotlib import gridspec
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backends.backend_wx import *
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import os

MAJOR_LW = 2.5
MINOR_LW = 1.5
MAX_ROWS = 10


def plot_ecg(ts=None,
             raw=None,
             filtered=None,
             rpeaks=None,
             templates_ts=None,
             templates=None,
             heart_rate_ts=None,
             heart_rate=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.ecg.ecg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ECG signal.
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    # creating a root widget
    root = Tk()
    root.resizable(False, False)  # default

    fig_raw, axs_raw = plt.subplots(3, 1, sharex=True)
    fig_raw.set_size_inches(5, 5, forward=True)
    fig_raw.suptitle('ECG Summary')

    # raw signal plot (1)
    axs_raw[0].plot(ts, raw, linewidth=MAJOR_LW, label='Raw', color='C0')
    axs_raw[0].set_ylabel('Amplitude')
    axs_raw[0].legend()
    axs_raw[0].grid()

    # filtered signal with R-Peaks (2)
    axs_raw[1].plot(ts, filtered, linewidth=MAJOR_LW, label='Filtered', color='C0')

    ymin = np.min(filtered)
    ymax = np.max(filtered)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha

    # adding the R-Peaks
    axs_raw[1].vlines(ts[rpeaks], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='R-peaks')

    axs_raw[1].set_ylabel('Amplitude')
    axs_raw[1].legend(loc='upper right')
    axs_raw[1].grid()

    # heart rate (3)
    axs_raw[2].plot(heart_rate_ts, heart_rate, linewidth=MAJOR_LW, label='Heart Rate')
    axs_raw[2].set_xlabel('Time (s)')
    axs_raw[2].set_ylabel('Heart Rate (bpm)')
    axs_raw[2].legend()
    axs_raw[2].grid()


    canvas_raw = FigureCanvasTkAgg(fig_raw, master=root)
    canvas_raw.get_tk_widget().grid(row=0, column=0, columnspan=1, rowspan=6, sticky='w')
    canvas_raw.draw()

    toolbarFrame = Frame(master=root)
    toolbarFrame.grid(row=6, column=0, columnspan=1, sticky=W)
    toolbar = NavigationToolbar2Tk(canvas_raw, toolbarFrame)
    toolbar.update()


    fig = fig_raw

    # fig_2, axs_2 = plt.subplots(6, 1)
    fig_2 = Figure()
    gs = gridspec.GridSpec(6, 1)

    # raw signal (acc_x)
    axs_2 = fig_2.add_subplot(gs[2:4, 0])

    fig_2.set_size_inches(10, 10, forward=True)

    axs_2.plot(templates_ts, templates.T, 'm', linewidth=MINOR_LW, alpha=0.7)
    axs_2.set_xlabel('Time (s)')
    axs_2.set_ylabel('Amplitude')
    axs_2.set_title('Templates')
    axs_2.grid()

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'

        fig.savefig(path, dpi=200, bbox_inches='tight')

    grid_params = {'row': 0, 'column': 1, 'columnspan': 2, 'rowspan' : 6, 'sticky': 'w'}
    # add an empty canvas for plotting
    canvas_2 = FigureCanvasTkAgg(fig_2, master=root)
    canvas_2.get_tk_widget().grid(**grid_params)
    canvas_2.draw()

    toolbarFrame_2 = Frame(master=root)
    toolbarFrame_2.grid(row=6, column=1, columnspan=1, sticky=W)
    toolbar_2 = NavigationToolbar2Tk(canvas_2, toolbarFrame_2)
    toolbar_2.update()

    if show:
        # window icon and title
        base, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
        base, _ = os.path.split(base)
        icon_path = os.path.join(base, 'docs', 'favicon.ico')
        root.iconbitmap(icon_path)
        root.wm_title("BioSPPy: ECG signal")

        mainloop()

    else:
        # close
        plt.close(fig)