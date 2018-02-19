#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='libSpineML2NK',
      version='1.0',
      packages=['libSpineML2NK'],

      install_requires=[
        'neurokernel >= 0.1',
        'libSpineML  >= 0.3'
      ]
     )
