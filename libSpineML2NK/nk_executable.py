#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""nk_exectuable will store a single executable experiment, designed in SpineML

SpineML is a declaratice Spiking neuron modelling language.
"""


#import os  # STD lib imports first
#import sys  # alphabetical

#import some_third_party_lib  # 3rd party stuff next
#import some_third_party_other_lib  # alphabetical
import argparse
import pdb

import libSpineML
from libSpineML import smlBundle
#import local_stuff  # local stuff last
#import more_local_stuff
#import dont_import_two, modules_in_one_line  # IMPORTANT!
#from pyflakes_cannot_handle import *  # and there are other reasons it should be avoided # noqa
# Using # noqa in the line above avoids flake8 warnings about line length!


#_a_global_var = 2  # so it won't get imported by 'from foo import *'
#_b_global_var = 3

#A_CONSTANT = 'ugh.'


# 2 empty lines between top-level funcs + classes
def units(value,unit):
    """Write docstrings for ALL public classes, funcs and methods.

    Functions use snake_case.
    """
    units = {
         'GV': 1*(10**9),'MV': 1*(10**6),'kV': 1*(10**3),'V' : 1,'cV': 1*(10**-1),'mV': 1*(10**-3),'uV': 1*(10**-6),'nV': 1*(10**-9),'pV': 1*(10**-12),'fV': 1*(10**-15),
         'GOhm': 1*(10**9),'MOhm': 1*(10**6),'kOhm': 1*(10**3),'Ohm' : 1,'cOhm': 1*(10**-1),'mOhm': 1*(10**-3),'uOhm': 1*(10**-6),'nOhm': 1*(10**-9),'pOhm': 1*(10**-12),'fOhm': 1*(10**-15),
         'GA': 1*(10**9),'MA': 1*(10**6),'kA': 1*(10**3),'A' : 1,'cA': 1*(10**-1),'mA': 1*(10**-3),'uA': 1*(10**-6),'nA': 1*(10**-9),'pA': 1*(10**-12),'fA': 1*(10**-15),
         'GF': 1*(10**9),'MF': 1*(10**6),'kF': 1*(10**3),'F' : 1,'cF': 1*(10**-1),'mF': 1*(10**-3),'uF': 1*(10**-6),'nF': 1*(10**-9),'pF': 1*(10**-12),'fF': 1*(10**-15),
         'GS': 1*(10**9),'MS': 1*(10**6),'kS': 1*(10**3),'S' : 1,'cS': 1*(10**-1),'mS': 1*(10**-3),'uS': 1*(10**-6),'nS': 1*(10**-9),'pS': 1*(10**-12),'fS': 1*(10**-15),'?':1
        }

    if unit in units:
        return(value * units[unit])
    else:
        raise ValueError("Unit not in units list: %s" % unit)




class Executable(object):
    """Executable Neurokernel Object

    Can take a libSpineML bundle, or a SpineML Experiment file
    """

    # some examples of how to wrap code to conform to 79-columns limit:
    def __init__(self, experiment=None):
        if type(experiment) is str:
            self.bundle = smlBundle.Bundle()
            self.bundle.add_experiment(experiment,True)

        elif type(experiment) is smlBundle.Bundle:
            self.bundle = experiment

        else:
            self.bundle = smlBundle.Bundle()
        self.params = {}



    # 1 empty line between in-class def'ns
    def execute(self):
        """Execute the model, after processing the class

        This method will create the Input file and Network file dynamically

        As a minimum, this fucntion will need to create a params dictionary as required by nk_manager
        with the following keys:

            params['name'], 
            params['dt'], 
            params['n_dict'], 
            params['s_dict'],
            input_file=params['input_file'],
            output_file=params['output_file'],
            components=params['components'])
            params['steps']
        """
        process_experiment()
        process_network()
        process_component()
        # from nk_manager import launch_nk
        #launch_nk(self.params)

    

    def process_experiment(self,bundleIndex=0,expIndex=0):
        """Process to the experiment file to extract NK relevant objects
    
            Each bundle can store many experiments, bundleIndex dictates the
            SpineML experient to use. Similary each SpineML Experiment can
            contain several experiment types, and so provisions are made for 
            acommodating multiple experiments.
        """
        # Extract input and output files

        exp = self.bundle.experiments[bundleIndex].Experiment[expIndex] 
        self.params['name'] = exp.get_name()

        # save everything in standard units before saving
        pdb.set_trace()

        self.params['dt'] = units(float(exp.Simulation.AbstractIntegrationMethod.dt),'mS')
        self.params['steps'] = units(float(exp.Simulation.duration),'S') / self.params['dt']

    def process_network(self):
        """Process to the experiment file to extract NK relevant objects
        """
        # extract input file
        # extract network file
        # create n_dict
        # create s_dict

        # get number or neurons
        num_neurons = 0;
        for n in self.bundle.networks[0].Population:
            num_neurons += n.Neuron.size
            print num_neurons
        # create Input file 
        # for every population in Network
            
        
        



    def process_component(self):
        """Process to the experiment file to extract NK relevant objects
        """
        pass




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', default='none', type=str,
                        help='Path to a SpineML experiment file')
    args = parser.parse_args()

    if args.experiment is not 'none':
        exe = Executable(args.experiment)
        print exe.bundle
    else:
        print "No Experiment Provided"

if __name__=='__main__':
    main()

