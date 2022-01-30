# -*- coding: utf-8 -*-
"""
biosppy.inter_plotting.ecg
-------------------

This module provides an interactive display option for the ACC plot.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import tkinter.font as tkFont
import sys
import os

# Globals
from biosppy import utils

MAJOR_LW = 2.5
MINOR_LW = 1.5
MAX_ROWS = 10


def plot_acc(ts=None, raw=None, vm=None, sm=None, spectrum=None, path=None):
    """Create a summary plot from the output of signals.acc.acc.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw ACC signal.
    vm : array
        Vector Magnitude feature of the signal.
    sm : array
        Signal Magnitude feature of the signal
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    raw_t = np.transpose(raw)
    acc_x, acc_y, acc_z = raw_t[0], raw_t[1], raw_t[2]

    root = Tk()
    root.resizable(False, False)  # default
    fig, axs_1 = plt.subplots(3, 1)
    axs_1[0].plot(ts, acc_x, linewidth=MINOR_LW, label="Raw acc along X", color="C0")
    axs_1[1].plot(ts, acc_y, linewidth=MINOR_LW, label="Raw acc along Y", color="C1")
    axs_1[2].plot(ts, acc_z, linewidth=MINOR_LW, label="Raw acc along Z", color="C2")

    axs_1[0].set_ylabel("Amplitude ($m/s^2$)")
    axs_1[1].set_ylabel("Amplitude ($m/s^2$)")
    axs_1[2].set_ylabel("Amplitude ($m/s^2$)")
    axs_1[2].set_xlabel("Time (s)")

    fig.suptitle("Acceleration signals")

    global share_axes_check_box

    class feature_figure:
        def __init__(self, reset=False):
            self.figure, self.axes = plt.subplots(1, 1)
            if not reset:
                self.avail_plots = []

        def on_xlims_change(self, event_ax):
            print("updated xlims: ", event_ax.get_xlim())

        def set_labels(self, x_label, y_label):
            self.axes.set_xlabel(x_label)
            self.axes.set_ylabel(y_label)

        def hide_content(self):
            # setting every plot element to white
            self.axes.spines["bottom"].set_color("white")
            self.axes.spines["top"].set_color("white")
            self.axes.spines["right"].set_color("white")
            self.axes.spines["left"].set_color("white")
            self.axes.tick_params(axis="x", colors="white")
            self.axes.tick_params(axis="y", colors="white")

        def show_content(self):
            # setting every plot element to black
            self.axes.spines["bottom"].set_color("black")
            self.axes.spines["top"].set_color("black")
            self.axes.spines["right"].set_color("black")
            self.axes.spines["left"].set_color("black")
            self.axes.tick_params(axis="x", colors="black")
            self.axes.tick_params(axis="y", colors="black")
            self.axes.legend()

        def draw_in_canvas(self, root: Tk, grid_params=None):
            if grid_params is None:
                grid_params = {
                    "row": 2,
                    "column": 1,
                    "columnspan": 4,
                    "sticky": "w",
                    "padx": 10,
                }
            # add an empty canvas for plotting
            self.canvas = FigureCanvasTkAgg(self.figure, master=root)
            self.canvas.get_tk_widget().grid(**grid_params)
            self.canvas.draw()
            # self.axes.callbacks.connect('xlim_changed', on_xlims_change)

        def dump_canvas(self, root):
            self.axes.clear()
            self.figure.clear()
            self.figure.canvas.draw_idle()
            self.canvas = FigureCanvasTkAgg(self.figure, master=root)
            self.canvas.get_tk_widget().destroy()

        def add_toolbar(self, root: Tk, grid_params=None):
            if grid_params is None:
                grid_params = {"row": 5, "column": 1, "columnspan": 2, "sticky": "w"}
            toolbarFramefeat = Frame(master=root)
            toolbarFramefeat.grid(**grid_params)

            toolbarfeat = NavigationToolbar2Tk(self.canvas, toolbarFramefeat)
            toolbarfeat.update()

        def add_plot(self, feature_name: str, xdata, ydata, linewidth, label, color):
            feature_exists = False
            for features in self.avail_plots:
                if features["feature_name"] == feature_name:
                    feature_exists = True
                    break
            if not feature_exists:
                self._add_plot(feature_name, xdata, ydata, linewidth, label, color)

        def _add_plot(self, feature_name: str, xdata, ydata, linewidth, label, color):

            plot_params = {
                "feature_name": feature_name,
                "x": xdata,
                "y": ydata,
                "linewidth": linewidth,
                "label": label,
                "color": color,
            }

            plot_data = dict(plot_params)
            self.avail_plots.append(plot_data)
            del plot_params["feature_name"]
            del plot_params["x"]
            del plot_params["y"]
            self.axes.plot(xdata, ydata, **plot_params)
            self.show_content()

        def remove_plot(self, feature_name: str):
            if self.avail_plots:
                removed_index = [
                    i
                    for i, x in enumerate(self.avail_plots)
                    if x["feature_name"] == feature_name
                ]
                if len(removed_index) == 1:
                    self._remove_plot(removed_index)

        def _remove_plot(self, removed_index):
            del self.avail_plots[removed_index[0]]
            self.__init__(reset=True)

            for params in self.avail_plots:
                temp_params = dict(params)
                del temp_params["feature_name"]
                del temp_params["x"]
                del temp_params["y"]
                self.axes.plot(params["x"], params["y"], **temp_params)

            if self.avail_plots == []:
                self.hide_content()
            else:
                self.show_content()

        def get_axes(self):
            return self.axes

        def get_figure(self):
            return self.figure

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root_, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ["png", "jpg"]:
            path = root_ + ".png"

        fig.savefig(path, dpi=200, bbox_inches="tight")

    # window title
    root.wm_title("BioSPPy: acceleration signal")

    root.columnconfigure(0, weight=4)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=1)
    root.columnconfigure(3, weight=1)
    root.columnconfigure(4, weight=1)

    helv = tkFont.Font(family="Helvetica", size=20)

    # checkbox
    show_features_var = IntVar()
    share_axes_var = IntVar()

    def show_features():
        global feat_fig
        global toolbarfeat
        global share_axes_check_box
        if show_features_var.get() == 0:
            drop_features2.get_menu().config(state="disabled")
            domain_feat_btn.config(state="disabled")

            # remove canvas for plotting
            feat_fig.dump_canvas(root)

        if show_features_var.get() == 1:
            # enable option menu for feature selection
            drop_features2.get_menu().config(state="normal")
            domain_feat_btn.config(state="normal")
            # canvas_features.get_tk_widget().grid(row=2, column=1, columnspan=1, sticky='w', padx=10)

            # add an empty canvas for plotting
            feat_fig = feature_figure()
            share_axes_check_box = Checkbutton(
                root,
                text="Share axes",
                variable=share_axes_var,
                onvalue=1,
                offvalue=0,
                command=lambda feat_fig=feat_fig: share_axes(feat_fig.get_axes()),
            )
            share_axes_check_box.config(font=helv)
            share_axes_check_box.grid(row=4, column=1, sticky=W)

            feat_fig.hide_content()
            feat_fig.draw_in_canvas(root)

    def share_axes(ax2):
        if share_axes_var.get() == 1:
            axs_1[0].get_shared_x_axes().join(axs_1[0], ax2)
            axs_1[1].get_shared_x_axes().join(axs_1[1], ax2)
            axs_1[2].get_shared_x_axes().join(axs_1[2], ax2)

        else:
            for ax in axs_1:
                ax.get_shared_x_axes().remove(ax2)
                ax2.get_shared_x_axes().remove(ax)
                ax.autoscale()
                canvas_raw.draw()

    check1 = Checkbutton(
        root,
        text="Show features",
        variable=show_features_var,
        onvalue=1,
        offvalue=0,
        command=show_features,
    )
    check1.config(font=helv)
    check1.grid(row=0, column=0, sticky=W)

    # FEATURES to be chosen
    clicked_features = StringVar()
    clicked_features.set("features")

    def domain_func():
        global share_axes_check_box

        if feat_domain_var.get() == 1:
            domain_feat_btn["text"] = "Domain: frequency"
            feat_domain_var.set(0)
            feat_fig.remove_plot("VM")
            feat_fig.remove_plot("SM")
            feat_fig.draw_in_canvas(root)
            drop_features2.reset()
            drop_features2.reset_fields(["Spectra"])
            share_axes_check_box.config(state="disabled")
            share_axes_var.set(0)

        else:
            domain_feat_btn["text"] = "Domain: time"
            feat_domain_var.set(1)
            feat_fig.remove_plot("SPECTRA X")
            feat_fig.remove_plot("SPECTRA Y")
            feat_fig.remove_plot("SPECTRA Z")
            feat_fig.draw_in_canvas(root)
            drop_features2.reset()
            drop_features2.reset_fields(["VM", "SM"])
            share_axes_check_box.config(state="normal")

    feat_domain_var = IntVar()
    feat_domain_var.set(1)

    class feat_menu:
        def __init__(self, fieldnames: list, entry_name="Select Features", font=helv):
            self.feat_menu = Menubutton(root, text=entry_name, relief="raised")
            self.feat_menu.grid(row=0, column=2, sticky=W)
            self.feat_menu.menu = Menu(self.feat_menu, tearoff=0)
            self.feat_menu["menu"] = self.feat_menu.menu
            self.feat_menu["font"] = font
            self.font = font
            self.feat_activation = {}

            # setting up disabled fields
            for field in fieldnames:
                self.feat_activation[field] = False

            for field in fieldnames:
                self.feat_menu.menu.add_command(
                    label=field,
                    font=helv,
                    command=lambda field=field: self.update_field(field),
                    foreground="gray",
                )
            self.fieldnames = fieldnames

            self.feat_menu.update()

        def reset(self, entry_name="Select Features"):
            self.feat_menu = Menubutton(root, text=entry_name, relief="raised")
            self.feat_menu.grid(row=0, column=2, sticky=W)
            self.feat_menu.menu = Menu(self.feat_menu, tearoff=0)
            self.feat_menu["menu"] = self.feat_menu.menu
            self.feat_menu["font"] = self.font
            self.feat_menu.update()

        def update_field(self, field):

            self.feat_activation[field] = not self.feat_activation[field]
            self.feat_menu.configure(text=field)  # Set menu text to the selected event

            self.reset()

            for field_ in self.fieldnames:
                if self.feat_activation[field_]:
                    self.feat_menu.menu.add_command(
                        label=field_,
                        font=helv,
                        command=lambda field=field_: self.update_field(field),
                    )
                else:
                    self.feat_menu.menu.add_command(
                        label=field_,
                        font=helv,
                        command=lambda field=field_: self.update_field(field),
                        foreground="gray",
                    )

                if field == "SM":
                    if not self.feat_activation[field]:
                        feat_fig.remove_plot("SM")
                        if any(self.feat_activation.values()):
                            feat_fig.set_labels("Time (s)", "Amplitude ($m/s^2$)")

                    else:
                        feat_fig.add_plot(
                            "SM",
                            ts,
                            sm,
                            linewidth=MINOR_LW,
                            label="Signal Magnitude feature",
                            color="C4",
                        )
                        feat_fig.set_labels("Time (s)", "Amplitude ($m/s^2$)")

                    feat_fig.draw_in_canvas(root)
                    feat_fig.add_toolbar(root)

                elif field == "VM":
                    if not self.feat_activation[field]:
                        feat_fig.remove_plot("VM")
                        if any(self.feat_activation.values()):
                            feat_fig.set_labels("Time (s)", "Amplitude ($m/s^2$)")
                    else:
                        feat_fig.add_plot(
                            "VM",
                            ts,
                            vm,
                            linewidth=MINOR_LW,
                            label="Vector Magnitude feature",
                            color="C3",
                        )
                        feat_fig.set_labels("Time (s)", "Amplitude ($m/s^2$)")

                    feat_fig.draw_in_canvas(root)
                    feat_fig.add_toolbar(root)

                elif field == "Spectra":
                    if not self.feat_activation[field]:
                        feat_fig.remove_plot("SPECTRA X")
                        feat_fig.remove_plot("SPECTRA Y")
                        feat_fig.remove_plot("SPECTRA Z")

                    else:

                        feat_fig.add_plot(
                            "SPECTRA X",
                            spectrum["freq"]["x"],
                            spectrum["abs_amp"]["x"],
                            linewidth=MINOR_LW,
                            label="Spectrum along X",
                            color="C0",
                        )
                        feat_fig.draw_in_canvas(root)

                        feat_fig.add_plot(
                            "SPECTRA Y",
                            spectrum["freq"]["y"],
                            spectrum["abs_amp"]["y"],
                            linewidth=MINOR_LW,
                            label="Spectrum along Y",
                            color="C1",
                        )
                        feat_fig.draw_in_canvas(root)

                        feat_fig.add_plot(
                            "SPECTRA Z",
                            spectrum["freq"]["z"],
                            spectrum["abs_amp"]["z"],
                            linewidth=MINOR_LW,
                            label="Spectrum along Z",
                            color="C2",
                        )
                        feat_fig.set_labels(
                            "Frequency ($Hz$)", "Normalized Amplitude [a.u.]"
                        )

                    feat_fig.draw_in_canvas(root)
                    feat_fig.add_toolbar(root)

                self.feat_menu.config(state="normal")
                self.feat_menu.update()

        def reset_fields(self, fieldnames):
            self.feat_activation = {}

            # setting up disabled fields
            for field in fieldnames:
                self.feat_activation[field] = False

            for field in fieldnames:
                self.feat_menu.menu.add_command(
                    label=field,
                    font=helv,
                    command=lambda field=field: self.update_field(field),
                    foreground="gray",
                )

            self.fieldnames = fieldnames

            self.feat_menu.update()

        def get_menu(self):
            return self.feat_menu

    domain_feat_btn = Button(root, text="Domain: time", command=domain_func)
    domain_feat_btn.config(font=helv, state="disabled")
    domain_feat_btn.grid(row=0, column=1, sticky=W, padx=10)

    temp_features = ["VM", "SM"]

    drop_features2 = feat_menu(temp_features, entry_name="Select Features", font=helv)
    drop_features2.get_menu().config(state="disabled")
    drop_features2.get_menu().update()

    canvas_raw = FigureCanvasTkAgg(fig, master=root)
    canvas_raw.get_tk_widget().grid(row=2, column=0, columnspan=1, sticky="w", padx=10)
    canvas_raw.draw()

    toolbarFrame = Frame(master=root)
    toolbarFrame.grid(row=5, column=0, columnspan=1, sticky=W)
    toolbar = NavigationToolbar2Tk(canvas_raw, toolbarFrame)
    toolbar.update()

    # Add key functionality
    def on_key(event):
        # print('You pressed {}'.format(event.key))
        key_press_handler(event, canvas_raw, toolbar)

    canvas_raw.mpl_connect("key_press_event", on_key)

    # tkinter main loop
    mainloop()
