import os
import sys

from biosppy import storage

import warnings

from biosppy.signals import ecg
from biosppy.signals.acc import acc

warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw ECG and ACC signals
ecg_signal, _ = storage.load_txt('./examples/ecg.txt')
acc_signal, _ = storage.load_txt('./examples/acc.txt')


# Setting current path
current_dir = os.path.dirname(sys.argv[0])
ecg_plot_path = os.path.join(current_dir, 'ecg.png')
acc_plot_path = os.path.join(current_dir, 'acc.png')

# Process it and plot. Set interactive=True to display an interactive window
out_ecg = ecg.ecg(signal=ecg_signal, sampling_rate=1000., path=ecg_plot_path, interactive=True)
out_acc = acc(signal=acc_signal, sampling_rate=1000., path=acc_plot_path, interactive=True)

