# BioSPPy - Biosignal Processing in Python

*A toolbox for biosignal processing written in Python.*

[![Image](https://github.com/PIA-Group/BioSPPy/raw/master/docs/logo/logo_400.png "I know you're listening! - xkcd.com/525")](http://biosppy.readthedocs.org/)

The toolbox bundles together various signal processing and pattern recognition
methods geared towards the analysis of biosignals.

Highlights:

- Support for various biosignals: BVP, ECG, EDA, EEG, EMG, Respiration
- Signal analysis primitives: filtering, frequency analysis
- Clustering
- Biometrics

Documentation can be found at: <http://biosppy.readthedocs.org/>

## Installation

Installation can be easily done with `pip`:

```bash
$ pip install biosppy
```

## Simple Example

The code below loads an ECG signal from the `examples` folder, filters it,
performs R-peak detection, and computes the instantaneous heart rate.

```python
from biosppy import storage
from biosppy.signals import ecg

# load raw ECG signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)
```

This should produce a plot similar to the one below.

[![Image](https://github.com/PIA-Group/BioSPPy/raw/master/docs/images/ECG_summary.png "ECG Summary Plot")]()

## Dependencies

- bidict
- h5py
- matplotlib
- numpy
- scikit-learn
- scipy
- shortuuid
- six

## Citing
Please use the following if you need to cite BioSPPy:

- Carreiras C, Alves AP, Lourenço A, Canento F, Silva H, Fred A, *et al.*
  **BioSPPy - Biosignal Processing in Python**, 2015-,
  https://github.com/PIA-Group/BioSPPy/ [Online; accessed ```<year>-<month>-<day>```].

```latex
@Misc{,
  author = {Carlos Carreiras and Ana Priscila Alves and Andr\'{e} Louren\c{c}o and Filipe Canento and Hugo Silva and Ana Fred and others},
  title = {{BioSPPy}: Biosignal Processing in {Python}},
  year = {2015--},
  url = "https://github.com/PIA-Group/BioSPPy/",
  note = {[Online; accessed <today>]}
}
```

## License

BioSPPy is released under the BSD 3-clause license. See LICENSE for more details.

## Disclaimer

This program is distributed in the hope it will be useful and provided
to you "as is", but WITHOUT ANY WARRANTY, without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This
program is NOT intended for medical diagnosis. We expressly disclaim any
liability whatsoever for any direct, indirect, consequential, incidental
or special damages, including, without limitation, lost revenues, lost
profits, losses resulting from business interruption or loss of data,
regardless of the form of action or legal theory under which the
liability may be asserted, even if advised of the possibility of such
damages.
