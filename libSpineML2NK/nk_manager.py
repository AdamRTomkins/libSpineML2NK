#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""nk_exectuable will store a single executable experiment, designed in SpineML

SpineML is a declaratice Spiking neuron modelling language.
"""

def launch_nk(params):
    """ Launch Neurokernel

    Lazy imports are used to maximize testable code
    """
    
    from neurokernel.core_gpu import Manager
    from neurokernel.tools.logging import setup_logger
    from libSpineML2NK.LPU import LPU

    man = Manager()
    #(n_dict, s_dict) = LPU.lpu_parser(data_dir+network_file+'.gexf.gz')


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
    man.start(params['steps'])
    man.wait()



