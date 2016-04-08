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

from neurokernel.tools.logging import setup_logger
from neurokernel.core_gpu import Manager
from translator_files.LPU import LPU

import os
import sys
import json

def run_lif(payload):
    """
    Run an example lif model through the nk_server json interface
    """

    print payload
    payload = json.loads(payload)
    params = payload['params']
    sim_input = payload['input']

    data_dir =  'data/'+params['sim_uid'] + '/' + params['sim_exp'] +'/'

    try:
        os.stat('data/'+params['sim_uid']+'/')
    except:
        os.mkdir('data/'+params['sim_uid']+'/')
    try:
        os.stat(data_dir)
    except:
        os.mkdir(data_dir)
 
    dt = params['sim_dt']
    Nt = params['sim_steps']
    dur = Nt/dt

    G = nx.DiGraph() # or nx.MultiDiGraph()
    G.add_nodes_from([0])
    G.node[0] = {
        'model': 'LeakyIAF_rest',
        'name': 'neuron_0',
        'extern': True,      # indicates whether the neuron can receive an external input signal
        'public': True,      # indicates whether the neuron can emit output to other LPUs 
        'spiking': True,     # indicates whether the neuron outputs spikes or a membrane voltage
        'selector': '/a[0]', # every public neuron must have a selector
        'V': params['par_V'], # initial membrane voltage
        'Vr': params['par_Vr'],     # reset voltage ## The same as the implicit resting potential
        'Vt': params['par_Vt'],       # spike threshold
        'R': params['par_R'],           # membrane resistance
        'C': params['par_C'],          # membrane capacitance
        'Er': params['par_rest']          # membrane capacitance
    }

    nx.write_gexf(G, data_dir +'lif_graph.gexf.gz')

    N_neurons = G.number_of_nodes()
    if sim_input == 'Default':
        t = np.arange(0, params['sim_dt']*params['sim_steps'], params['sim_dt'])
        I = np.zeros((params['sim_steps'], N_neurons), dtype=np.double)
        I[t>0.2] = 1e-9
        I[t>0.4] = 2e-9
        I[t>0.8] = 2.5e-9
       

        with h5py.File(data_dir + 'lif_input.h5', 'w') as f:
            f.create_dataset('array', (Nt, N_neurons),
                         dtype=np.double,
                         data=I)
    else:
        print 'loading non-default inputs (WIP)'
        with h5py.File(data_dir + 'lif_input.h5', 'w') as f:
            f.create_dataset('array', (Nt, N_neurons),
                         dtype=np.double,
                         data=sim_input)

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')

    parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')

    parser.add_argument('-s', '--steps', default=params['sim_steps'], type=int,
                    help='Number of steps [default: %s]' % params['sim_steps'])

    args = parser.parse_args()


    file_name = None

    screen = False
    if args.log.lower() in ['file', 'both']:
        file_name = 'lif.log'
    if args.log.lower() in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)


    man = Manager()
    (n_dict, s_dict) = LPU.lpu_parser(data_dir+'lif_graph.gexf.gz')

    if params['sim_output'] != 'spike':
        args.debug = True

    man.add(LPU, 'lif', dt, n_dict, s_dict, 
            input_file=data_dir +'lif_input.h5',
            output_file=data_dir +'lif_output.h5',
            device=0, debug=True)

    man.spawn()
    man.start(steps=params['sim_steps'])
    man.wait()

    if params['sim_output'] == 'spike':
        with h5py.File(data_dir +'lif_output_spike.h5') as f:
            data = np.array(f['array']).T.tolist()   
    else:
        ######## BUG: Needs to output debug to data folder
        with h5py.File('./lif_V.h5') as f:
            data = np.array(f['array']).T.tolist()   
        
    return data

def main():
    """
    Enable a default lif test run, without running nk_server.
    """

    def_params =  {  'params':{"sim_uid": "Run_Script",
                    "sim_exp": "Exp1",
                    "sim_steps": 100,
                    "sim_dt": 0.01,
                    "sim_output": "spike",
                    "sim_model":"lif",
                    "par_V": -0.05,
                    "par_Vr": -0.07,
                    "par_Vt": -0.01,
                    "par_R": 3e7,
                    "par_rest":-0.06,
                    "par_C": 1e-9},
                    'input':'Default'
                 }


    print def_params

    run_lif(json.dumps(def_params))

if __name__ == "__main__":
    main()

