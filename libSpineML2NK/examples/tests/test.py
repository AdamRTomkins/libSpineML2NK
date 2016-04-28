#!/usr/bin/env python

"""
requires import neurokernel.mpi_relaunch to run 
"""

import neurokernel.mpi_relaunch
from libSpineML2NK import nk_executable

e = nk_executable.Executable('Premade/experiment0.xml')
e.execute()
print "Closing Files"
import time
time.sleep(1) # delays for 5 seconds
print "Finished"
