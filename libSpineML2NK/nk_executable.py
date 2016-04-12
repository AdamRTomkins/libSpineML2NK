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
import numpy as np
import pdb
import networkx as nx


import libSpineML
from libSpineML import smlBundle
import nk_utils
import nk_inputs
#import local_stuff  # local stuff last
#import more_local_stuff
#import dont_import_two, modules_in_one_line  # IMPORTANT!
#from pyflakes_cannot_handle import *  # and there are other reasons it should be avoided # noqa
# Using # noqa in the line above avoids flake8 warnings about line length!


#_a_global_var = 2  # so it won't get imported by 'from foo import *'
#_b_global_var = 3

#A_CONSTANT = 'ugh.'



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
        self.network = nx.DiGraph()   



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
        self.params['dt'] = nk_utils.units(float(exp.Simulation.AbstractIntegrationMethod.dt),'mS')
        self.params['steps'] = nk_utils.units(float(exp.Simulation.duration),'S') / self.params['dt']

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
        
        exp_name = self.bundle.keys()[0]
        populations = self.bundle.index['exp_name']['network']['model.xml'].Population


        lpu_index = 0

        for p in populations:
            lpu_start = lpu_index;          # Start position for each neuron

            for n in np.arange(0,p.Neuron.size):
                self.add_neuron(p.Neuron.url,p.Neuron.Property,lpu_index,n,p.Neuron.name)
                lpu_index +=1

            for i in self.index[exp_name]['experiment'][exp_name].Experiment[0].AbstractInput:
                if p.Neuron.name ==  i.target:
                    self.initialise_input(i,lpu_start,pop.size)
            
    def initialise_input(self,params,lpu_start,lpu_size):
        """ initialise an input in the matrix for a given input to a population  

        itype=  params['type']

        if (itype == 'TimeVaryingArrayInput'):
            self.TimeVaryingArrayInput(params,lpu_start,lpu_size)

        if (itype == 'ConstantInput'):
            self.ConstantInput(params,lpu_start,lpu_size)

        if (itype == 'ConstantArrayInput'):
            self.ConstantArrayInput(params,lpu_start,lpu_size)

        if (itype == 'TimeVaryingInput'):
            self.TimeVaryingInput(params,lpu_start,lpu_size)
        """
        pass

        # create Input file 
        # for every population in Network
            
    def standard_neurons(self,model):
        """ provide the base neuron parameters from neurokernel, which are not in SpineML """
        """ DEPRECIATED TO ALLOW C_GENERATION 
            URL is used to load the correct kernel 
            WIP: Automatic discovery of extern, spiking and public based on component and connections
            external is true, to work with input generation, this will not scale well
      
        """
        return {'model': 'SpineMLNeuron','name': 'neuron_x','extern': True,'public': False,'spiking': True,'selector': '/a[0]','V':0,"url":model}


    def add_neuron(self,model,props,lpu_index,p_index,pop_name):
        """ add a neuron to the gexf population,
            where p_index is the neuron index within a population
        """
        
        neuron = self.standard_neurons(model)
                
        for p in props:
            """  p example: 'C': {'dimension': 'nS','input':{'type':'FixedValue','value':1}} """
            neuron[p.name] = nk_utils.gen_value(p,p_index)
            
            

        neuron['name'] =     'neuron_' +str(lpu_index)  # + '_' + str(p_index)
        neuron['selector'] = '/'+pop_name+'[' +str(lpu_index) +']'    #+ '[' + str(p_index)+']'
       
        self.network.add_node(lpu_index,attr_dict=neuron)
      


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

