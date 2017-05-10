#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""nk_exectuable will store a single executable experiment, designed in SpineML

SpineML is a declaratice Spiking neuron modelling language.
"""

from neurokernel.core_gpu import Manager
from neurokernel.tools.logging import setup_logger
from libSpineML2NK.LPU.LPU_ND import LPU


def launch_nk(params,debug=True,device=0,log=True):
    """ Launch Neurokernel
    """
    if log:
        screen = True
        logger = setup_logger(file_name=params['name'] +'.log', screen=screen)
    
    man = Manager()

    man.add(LPU, 
            params['name'], 
            params['dt'], 
            params['comp_dict'], 
            params['conns'],
            device=device, 
            input_processors = params['input_processors'],
            output_processors = params['output_processors'],
            debug=True,
            extra_comps=params['extra_comps'],
            SpineMLComponents=params['components'])

    man.spawn()
    man.start(int(params['steps']))
    man.wait()

if __name__=='__main__':

    params = {}
    params['components'] = None
    launch_nk(params)


