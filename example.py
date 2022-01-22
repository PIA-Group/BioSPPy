import os
import sys

from biosppy import storage

import warnings

from biosppy.signals import ecg

warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw ECG signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# Setting current path
current_dir = os.path.dirname(sys.argv[0])
my_plot_path = os.path.join(current_dir, 'output.png')

# Process it and plot. Set interactive=True to display an interactive window
out = ecg.ecg(signal=signal, sampling_rate=1000., path=my_plot_path, interactive=True)