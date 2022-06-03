BioSPPy Changelog
=================

Here you can see the full list of changes between each BioSPPy release.

Version 0.8.0
-------------

Released on December 20th 2021

- Added PCG module to signals.
- Fixed some bugs.

Version 0.7.3
-------------

Released on June 29th 2021

- Removed BCG from master until some issues are fixed.

Version 0.7.2
-------------

Released on May 14th 2021

- Fixed BCG dependencies.

Version 0.7.1
-------------

Released on May 14th 2021

- Included BCG module.

Version 0.7.0
-------------

Released on May 7th 2021

- GitHub and PyPI versions synced.

Version 0.6.1
-------------

Released on August 20th 2018

- Fixed source file encoding

Version 0.6.0
-------------

Released on August 20th 2018

- Added reference for BVP onset detection algorithm (closes #36)
- Updated readme file
- New setup.py style
- Added online filtering class in signals.tools
- Added Pearson correlation and RMSE methods in signals.tools
- Added method to compute Welch's power spectrum in signals.tools
- Don't use detrended derivative in signals.eda.kbk_scr (closes #43)
- Various minor changes

Version 0.5.1
-------------

Released on November 29th 2017

- Fixed bug when correcting r-peaks (closes #35)
- Fixed a bug in the generation of the classifier thresholds
- Added citation information to readme file (closes #34)
- Various minor changes

Version 0.5.0
-------------

Released on August 28th 2017

- Added a simple timing module
- Added methods to help with file manipulations
- Added a logo :camera:
- Added the Matthews Correlation Coefficient as another authentication metric.
- Fixed an issue in the ECG Hamilton algorithm (closes #28)
- Various bug fixes

Version 0.4.0
-------------

Released on May 2nd 2017

- Fixed array indexing with floats (merges #23)
- Allow user to modify SCRs rejection treshold (merges #24)
- Fixed the Scikit-Learn cross-validation module deprecation (closes #18)
- Addd methods to compute mean and meadian of a set of n-dimensional data points
- Added methods to compute the matrix profile
- Added new EMG onset detection algorithms (merges #17)
- Added finite difference method for numerial derivatives
- Fixed inconsistent decibel usage in plotting (closes #16)

Version 0.3.0
-------------

Released on December 30th 2016

- Lazy loading (merges #15)
- Python 3 compatibility (merges #13)
- Fixed usage of clustering linkage parameters
- Fixed a bug when using filtering without the forward-backward technique
- Bug fixes (closes #4, #8)
- Allow BVP parameters as inputs (merges #7)

Version 0.2.2
-------------

Released on April 20th 2016

- Makes use of new bidict API (closes #3)
- Updates package version in the requirements file
- Fixes incorrect EDA filter parameters
- Fixes heart rate smoothing (size parameter)

Version 0.2.1
-------------

Released on January 6th 2016

- Fixes incorrect BVP filter parameters (closes #2)

Version 0.2.0
-------------

Released on October 1st 2015

- Added the biometrics module, including k-NN and SVM classifiers
- Added outlier detection methods to the clustering module
- Added text-based data storage methods to the storage module
- Changed docstring style to napoleon-numpy
- Complete code style formatting
- Initial draft of the tutorial
- Bug fixes

Version 0.1.2
-------------

Released on August 29th 2015

- Alpha release
