#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='libSpineML2NK',
      version='1.0',
      packages=['libSpineML2NK',
                'libSpineML2NK.neurons',
                'libSpineML2NK.synapses',
		        'libSpineML2NK.utils',
		        'libSpineML2NK.LPU',
                'libSpineML2NK.models'],

      install_requires=[
        'neurokernel >= 0.1',
        'libSpineML  >= 0.1'
      ]
     )
