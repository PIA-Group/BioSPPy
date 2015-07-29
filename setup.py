# -*- coding: utf-8 -*-
"""
BioSPPy
-------

A toolbox for biosignal processing written in Python.

"""

# Imports
import biosppy
import os
from setuptools import find_packages, setup



def read(*paths):
    """Build a file path from *paths and return the contents."""
    
    with open(os.path.join(*paths), 'r') as fid:
        return fid.read()


def get_version():
    """Get the module version"""
    
    return biosppy.__version__


setup(name='biosppy',
      version=get_version(),
      description="A toolbox for biosignal processing written in Python.",
      long_description=read('README.rst'),
      url='https://github.com/PIA-Group/BioSPPy',
      license='BSD 3-clause',
      author='Instituto de Telecomunicacoes',
      author_email='carlos.carreiras@lx.it.pt',
      packages=find_packages(exclude=['tests*', 'docs*', 'examples']),
      # install_requires=[],
      include_package_data=True,
      classifiers=['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ],
      )
