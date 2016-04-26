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
    

    ############
    man = Manager()
    (n_dict, s_dict) = LPU.lpu_parser('examples/Premade/BasicExperiment.gexf.gz')


    man.add(LPU, 'spine_ml', 0.01, n_dict, s_dict,
                input_file='examples/Premade/BasicExperiment_input.h5',
                output_file='examples/Premade/BasicExperimen_output.h5',
                device=0, debug=True,components=params['components'])
    man.spawn()
    man.start(params['steps'])
    man.wait()
    print "finished"
    ###########
    """
    #man = Manager()
    #(n_dict, s_dict) = LPU.graph_to_dicts(params['graph'])
    #params['n_dict'] = n_dict
    #params['s_dict'] = s_dict
    
    man.add(LPU, 
            params['name'], 
            params['dt'], 
            params['n_dict'], 
            params['s_dict'],
            0,#input_file=params['input_file'],
            0,#output_file=params['output_file'],
            device=0, 
            debug=True,
            components=params['components'])
    """



