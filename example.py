from biosppy import storage

import warnings

from biosppy.signals import ecg

warnings.simplefilter(action='ignore', category=FutureWarning)

# load raw ECG signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)