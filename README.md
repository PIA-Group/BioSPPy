# BioSPPy - Biosignal Processing in Python

A toolbox for biosignal processing written in Python.

The toolbox bundles together various signal processing and pattern recognition methods geared towards the analysis of biosignals.

Highlights:

- Support for various biosignals: BVP, ECG, EDA, EEG, EMG, Respiration;
- Signal analysis primitives: filtering, frequency analysis
- Clustering
- Biometrics

## Installation

Installation can be easily done with `pip`:

```bash
$ pip install biosppy
```

## Dependencies

- numpy
- matplotlib
- scipy
- scikit-learn
- h5py
- shortuuid
- bidict

## License and Citation

BioSPPy is released under the BSD 3-clause license. See LICENSE for more details.

Please cite BioSPPy in your publication if it helps your research:

    @article{biosppy2015,
      Author = {},
      Journal = {},
      Title = {},
      Year = {2015}
    }

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
