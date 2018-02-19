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
import h5py

import libSpineML
from libSpineML import smlBundle as smlBundle
from libSpineML import smlExperiment as smlExperiment
from neurokernel.core_gpu import Manager

from libSpineML.smlComponent import AnalogReducePortType
from libSpineML.smlComponent import AnalogSendPortType
from libSpineML.smlComponent import EventSendPortType

from libSpineML2NK.LPU.LPU_ND import LPU
#from neurokernel.LPU.LPU import LPU

import nk_utils
import nk_manager

from libSpineML2NK.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from libSpineML2NK.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from libSpineML2NK.LPU.OutputProcessors.CrossbarOutputProcessor import CrossbarOutputProcessor
from libSpineML2NK.LPU.NDComponents.AxonHillockModels.SpikingSpineML import SpikingSpineML

class Executable(object):
    """Executable Neurokernel Object

    Can take a libSpineML bundle, or a SpineML Experiment file
    """

    def __init__(self, experiment=None):
        self.params     = {}
        self.params['extra_comps'] = [SpikingSpineML]

        if type(experiment) is str:
            self.bundle = smlBundle.Bundle()
            self.bundle.add_experiment(experiment,True)
            exp = self.bundle.experiments[0].Experiment[0]
            self.params['name'] = exp.get_name()

        elif type(experiment) is smlBundle.Bundle:
            self.bundle = experiment
            exp = self.bundle.experiments[0].Experiment[0] 
            self.params['name'] = exp.get_name()

        elif type(experiment) is dict:
            # Catch crossbar dictionaries:
            """
            {
                'experiment': "..."
                'network':    "..."  
                'components': ["..."]
            }
            """
            self.bundle = smlBundle.Bundle(project_dict=experiment)
            exp = self.bundle.experiments[0].Experiment[0] 
            self.params['name'] = exp.get_name()



        else:
            self.bundle = smlBundle.Bundle()
            self.params['name'] = 'No_name'
       
        self.network    = nx.MultiDiGraph()
        self.populations = {}
        #self.network    = nx.DiGraph()

        self.inputs     = np.zeros((0, 0), dtype=np.double)
        self.params['input_processors'] = []
        self.time     = np.zeros((0, 0), dtype=np.double)
        self.debug = False
        self.log = False


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
        
        self.params['input_file'] = self.params['name'] + '_input.h5'
        self.params['output_file'] = self.params['name'] + '_output.h5'

       

        print "process_experiment"
        self.process_experiment()
        print "process_network"

        self.process_network()
        print "process_component"
        self.process_component()
        print "save network and input"

        SpineMLComponent = self.params['components'][self.params['components'].keys()[0]]        
        updates = []
        accesses = []

        # Extract Outputs from Component
        for p in SpineMLComponent.ComponentClass.Port:
            
            if type(p) == AnalogSendPortType:
                updates.append(p.name)
            if type(p) == EventSendPortType:
                updates.append(p.name)

        # This needs to be populated dynamically from the variables
        record_vars = [ (x,None) for x in updates]
        fl_output_processors = []        
        for r in record_vars:
            #fl_output_processors.append(CrossbarOutputProcessor([r],url='http://127.0.0.1:8080/publish', topic= 'ffbo.sharc.update', sample_interval=1)) # TODO! ADD CLIENT ID TO topic!
            fl_output_processors.append(FileOutputProcessor([r],'./output.h5', sample_interval=1)) # TODO! ADD CLIENT ID TO topic!

        self.params['output_processors'] = fl_output_processors
    
        self.save_network()

        from nk_manager import launch_nk
    
        print "Launching nk"
        launch_nk(self.params,self.debug,self.log)

    def set_debug(self, debug=True):
        self.debug = debug

    def set_log(self,log=True):
        self.log = log

    def process_experiment(self,bundleIndex=0,expIndex=0):
        """Process to the experiment file to extract NK relevant objects
    
            Each bundle can store many experiments, bundleIndex dictates the
            SpineML experient to use. Similary each SpineML Experiment can
            contain several experiment types, and so provisions are made for 
            acommodating multiple experiments.
        """
        # Extract input and output files

        exp = self.bundle.experiments[bundleIndex].Experiment[expIndex] 

        #message = " For each input, create a specific Input processor here, and link it at the end"
        #print message        
        #assert message == "TODO: For each input, create a specific Input processor here, and link it at the end!"
       

        self.params['name'] = exp.get_name()

        # save everything in standard units before saving
        self.params['dt'] = nk_utils.units(float(exp.Simulation.AbstractIntegrationMethod.dt),'mS')
        self.params['steps'] = nk_utils.units(float(exp.Simulation.duration),'S') / self.params['dt']
        
        self.params['num_neurons'] = 0;

        for n in self.bundle.networks[0].Population:
            self.params['num_neurons']+= n.Neuron.size
  
        ######################################################################
        # Correct dt and time to be in standard
        #####################################################################
        print self.params
        self.inputs = np.zeros((int(self.params['steps']), int(self.params['num_neurons'])), dtype=np.double)
        self.time   = time = np.arange(0,self.params['dt']*self.params['steps'] , self.params['dt'])

        # Provess Lesions
        # Process Configutations

    def create_input_processor(self,abstract_input):
        lpu_start,lpu_size = self.populations[abstract_input.target]

        # Extract  uids
        # Extract data
        # Create file
        # Create input processor
        data = self.initialise_input(abstract_input,lpu_start,lpu_size)
        uids = ["neuron_%s" % (n + lpu_start) for n in np.arange(lpu_size)]
        input_port = abstract_input.get_port()
        

        # TODO: Expand to multiple inputs
        with h5py.File(self.params['input_file'] , 'w') as f:
            print "Creating input I, this should be dynamically made!"
     
            f.create_dataset('%s/uids' % input_port, data=uids)
            f.create_dataset('%s/data' % input_port, data.shape,
                             dtype=np.float64,
                             data=data)

        fl_input_processor = FileInputProcessor(self.params['input_file'])
        self.params['input_processors'].append(fl_input_processor)

        pass

    def process_network(self):
        """Process to the experiment file to extract NK relevant objects
        """
        # extract input file
        # extract network file
        # create n_dict
        # create s_dict

        exp_name = self.bundle.index.keys()[0]

        print "Experiment Name : %s" % self.bundle.index[exp_name]

        model_name = self.bundle.index[exp_name]['network'].keys()[0]
        populations = self.bundle.index[exp_name]['network'][model_name].Population

        lpu_index = 0      
        for p in populations:
            lpu_start = lpu_index;          # Start position for each neuron
            
            for n in np.arange(0,p.Neuron.size):
                self.add_neuron(p.Neuron.url,p.Neuron.Property,lpu_index,n,p.Neuron.name,exp_name)
                lpu_index += 1
            
            self.populations[p.Neuron.name] = (lpu_start,p.Neuron.size)


        exp = self.bundle.experiments[0].Experiment[0] 

        for i in exp.AbstractInput:
            self.create_input_processor(i)


        self.params['graph'] = self.network

        # requires a filename not a graph
        (comp_dict, conns) = LPU.graph_to_dicts(self.network)
        
        self.params['comp_dict'] = comp_dict
        self.params['conns'] = conns
            
    def initialise_input(self,params,lpu_start,lpu_size):
        # initialise an input in the matrix for a given input to a population  

        itype =  type(params)

        if (itype == smlExperiment.TimeVaryingArrayInputType):
            return nk_utils.TimeVaryingArrayInput(params,lpu_start,lpu_size,self.time,self.inputs)

        elif (itype == smlExperiment.ConstantInputType):
            return nk_utils.ConstantInput(params,lpu_start,lpu_size,self.time,self.inputs)

        elif (itype == smlExperiment.ConstantArrayInputType):
            return nk_utils.ConstantArrayInput(params,lpu_start,lpu_size,self.time,self.inputs)

        elif (itype == smlExperiment.TimeVaryingInputType):
            return nk_utils.TimeVaryingInput(params,lpu_start,lpu_size,self.time,self.inputs)
        else:
            raise TypeError('type %s is not recognised as an input type' %str(itype))            
        return None

    def standard_neurons(self,model):
        """ provide the base neuron parameters from neurokernel, which are not in SpineML """
        """ DEPRECIATED TO ALLOW C_GENERATION 
            URL is used to load the correct kernel 
            WIP: Automatic discovery of extern, spiking and public based on component and connections
            external is true, to work with input generation, this will not scale well
      
        """
        #return {'model': 'SpineMLNeuron','name': 'neuron_x','extern': True,'public': False,'spiking': True,'selector': '/a[0]','V':0,"url":model}
        
        return {'class': 'SpikingSpineML','name': 'neuron_x','extern': True,'public': False,'spiking': True,'selector': '/a[0]','V':0,"url":model}



    def add_neuron(self,model,props,lpu_index,p_index,pop_name,exp_name):
        """ add a neuron to the gexf population,
            where p_index is the neuron index within a population
        """
        
        neuron = self.standard_neurons(model)
                
        for p in props:
            """  p example: 'C': {'dimension': 'nS','input':{'type':'FixedValue','value':1}} """
            neuron[p.name] = nk_utils.gen_value(p,p_index)
         
        neuron['name'] =     'neuron_' +str(lpu_index)  # + '_' + str(p_index)
        neuron['selector'] = '/'+pop_name+'[' +str(lpu_index) +']'    #+ '[' + str(p_index)+']'

        # Determine if the neuron will be spiking or gpot
        # requires that only one output port exists

        comp = self.bundle.index[exp_name]['component'][model]
 
        for port in comp.ComponentClass.Port:
            if type(port) is libSpineML.smlComponent.AnalogSendPortType:
                neuron['spiking'] = False
                break
            if type(port) is libSpineML.smlComponent.ImpulseSendPortType:
                neuron['spiking'] = True
                break

##################################
#
#   Swap V out with default parameter from output port
#
##################################


#        self.network.add_node(str(lpu_index),attr_dict=neuron)
        self.network.add_node(neuron['name'],attr_dict=neuron)
     

    def process_component(self):
        """Process to the experiment file to extract NK relevant objects
        """
 
        exp_name = self.bundle.index.keys()[0]
        self.params['components'] = self.bundle.index[exp_name]['component']      
        
    ###############################################
    # Added for output testing
    ##############################################        
    


    def save_network(self):
        """ save the network file before running """
        nx.write_gexf(self.network, self.params['input_file'] +'.gexf.gz')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', default='none', type=str,
                        help='Path to a SpineML experiment file')
    args = parser.parse_args()

    if args.experiment is not 'none':
        exe = Executable(args.experiment)
        exe.execute()
    else:
        print "No Experiment Provided"

if __name__=='__main__':
    main()

