#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""nk_exectuable will store a single executable experiment, designed in SpineML

SpineML is a declaratice Spiking neuron modelling language.
"""

from neurokernel.core_gpu import Manager
from neurokernel.tools.logging import setup_logger
from libSpineML2NK.LPU import LPU


def launch_nk(params):
    """ Launch Neurokernel
    """

    screen = True
    logger = setup_logger(file_name=params['name'] +'.log', screen=screen)
    
    man = Manager()

    man.add(LPU, 
            params['name'], 
            params['dt'], 
            params['n_dict'], 
            params['s_dict'],
            input_file=params['input_file'],
            output_file=params['output_file'],
            device=0, 
            debug=True,
            components=params['components'])

    man.spawn()
    man.start(int(params['steps']))
    man.wait()

if __name__=='__main__':
    params = {}
    params['components'] = None
    launch_nk(params)


