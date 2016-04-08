

# Imports here
import pdb
import argparse
import itertools

from neurokernel.core_gpu import Module

import numpy as np

import networkx as nx
import h5py

import cPickle
import scipy
from scipy import io

import os
import sys
import json

import translator_files

from translator_files.LPU import LPU

from translator_files.translator import nk_spineml
from translator_files.translator import nk_component
#from nk_spineml import Experiment

import neurokernel.mpi_relaunch
# Neurokernal Experiment Object


exp = nk_spineml.Experiment('BasicExperiment','add a description for the experiment here')

        
exp.set_simulation(0.1,0.1,'BRAHMS')


# Define Components
  
LeakyIAF = nk_component.Component("neuron_body")
# Dynamics
LeakyIAF.add_state_variable("V","mV")
integrating = nk_component.Regime("integrating")

# Time Derivatives
integrating.add_derivative("V","((I_Syn) / C) + (Er - V) / (R*C)")


# OnConditions are named using random strings to allow multiple conditions per target regime
idm139794361044160 = nk_component.Condition("integrating")

#con.add_assignment("V","Vr")
idm139794361044160.add_assignment("V","Vr")


#con.add_trigger("V > Vt")
idm139794361044160.add_trigger("V > Vt")



#con.add_event("spike")
integrating.add_condition(idm139794361044160)      

LeakyIAF.add_regime(integrating,

    True
  )

LeakyIAF.add_parameter("C","nS")
LeakyIAF.add_parameter("Vt","mV")
LeakyIAF.add_parameter("Er","mV")
LeakyIAF.add_parameter("Vr","mV")
LeakyIAF.add_parameter("R","MOhm")
LeakyIAF.add_send_port("Analog","V")
LeakyIAF.add_recieve_port("AnalogReduce","I_Syn","mA","+")

exp.add_component("LeakyIAF.xml",LeakyIAF)



model_params = {
                    
                        'C': {
                        'dimension': 'nS',
                        
                        'input':{'type':'FixedValue',
                            'value':1}
                            },
                        'Vt': {
                        'dimension': 'mV',
                        
                        'input':{'type':'FixedValue',
                            'value':-10}
                            },
                        'Er': {
                        'dimension': 'mV',
                        
                        'input':{'type':'FixedValue',
                            'value':-70}
                            },
                        'Vr': {
                        'dimension': 'mV',
                        
                        'input':{'type':'UniformDistribution',
                            
                                'seed':123,
                                'maximum':-70,
                                'minimum':-100}
                            },
                        'R': {
                        'dimension': 'MOhm',
                        
                        'input':{'type':'FixedValue',
                            'value':3}
                            },
                        'V': {
                        'dimension': 'mV',
                        
                        'input':{'type':'FixedValue',
                            'value':-65}
                            }
                }
exp.add_population('Population',5,'LeakyIAF.xml', model_params)


                

exp.add_input({'type':'ConstantInput',
            
                'name':'Inpu',
                
                'target':'Population',
                
                'port':'I_Syn',
                
                'start_time':0,
                
                'value':0
		}) ##
            

exp.run()
import matplotlib.pyplot as plt; import h5py
f = h5py.File('spine_ml_V.h5')['array']; plt.subplot(311);plt.plot(f);
f = h5py.File('spine_ml_I.h5')['array']; plt.subplot(312);plt.plot(f);

plt.show()
