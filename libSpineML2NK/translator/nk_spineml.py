import numpy as np

import networkx as nx

import h5py

import pdb

from neurokernel.core_gpu import Manager
from neurokernel.tools.logging import setup_logger
from translator_files.LPU import LPU

from translator_files.translator import nk_component
"""
WIP:

    Output Specification - will require new neuron models
                            or changes in the NK core to enable specific variable tracking
    Componant Layer      - will require code generation
    Input                - requires testing
    Connectivity         - requires testing
    Experiment Layers    - add lesions and configurations

"""

class Population:
    """ Common class for SpineML populations

        Keyword arguments:
            name -- population name
            size -- population size
            model -- population model file
            params -- population parameters supplied as a dictionary
            start -- position of the population in the LPU

    """
    def __init__(self,name,size,model,params,start):
        self.name = name
        self.size = size
        self.model = model
        self.params = params
        self.start = start

class Projection:
    """ Common class for SpineML projections

        Keyword arguments:
            source -- population name
            destination -- population size
            synapse_list -- population model file

    """
    def __init__(self,source,destination,synapse_list):
        self.source = source
        self.destination = destination
        self.synapse_list = synapse_list
    


class Experiment:
    """Common input class for Experiments"""

    def __init__(self,name='Default_Experiment',description ='None'):
        self.name = name                # Name of the experiment defined in SpineML
        self.description = description  # Description of the experiment defined in SpineML

        self.network = nx.DiGraph()     # Store the network graph
        self.network_mapping = {}       # Store the network mapping used for mapping
                                        # Inputs to Populations
        self.populations = {}           # Store populations by name
        self.projections = []           # Store projections by name

        self.inputs = {}                # Store inputs by target population name

        self.model = {}
        
        self.components = {}

        self.neurons = 0

        self.simulation = {}
        self.output = {}

        self.__output_dir = "./"

    def set_simulation(self,duration=1,dt=0.1,preffered_simulator='None',
                       integration_method='EulerIntegration',rk_order=1):
        """establish simulation parameters"""

        self.simulation['duration'] = duration
        self.simulation['dt'] = dt/1000 # Convert to miliseconds
        self.simulation['preffered_simulator'] = preffered_simulator
        if integration_method != 'EulerIntegration':
            print "Only EulerIntegration is currently supported, continuing with euler"
        self.simulation['integration_method'] = 'EulerIntegration'

    def add_output(self,name='Default_Output',target=None,port=None,indicies=[],start_time=0,duration=0):
        """ Assign an output to be recorded.
            Target must be defined

        Keyword arguments:
        N/A -- description

        """
        print "adding an ouput"


    def add_population(self,name,size,model,params):
        """ Add a population to the experiment """
        self.populations[name] = Population(name,size,model,params,self.neurons)
        self.neurons = self.neurons + size

    def add_component(self,name,component):
        """ Add a population to the experiment """
        self.components[name] = component

    def add_projection(self,source,desination,synapse_list):
        """ Add a projection to the experiment """
        self.projections.append(Projection(source,desination,synapse_list))



    def add_input(self,input_params):
        """Create an network object for the experiment"""
        if input_params['target'] in self.inputs:
            self.inputs[input_params['target']].append(input_params)
        else:
            self.inputs[input_params['target']] = [input_params]

    def add_lesion(self):
        """Create a lesion object for the model """
        print "Lesions are not yet supported in this translator"

    def add_configuration(self):
        """Create a lesion object for the model """
        print "Configurations are not yet supported in this translator"

    def __save_input(self,file_name,sim_steps,sim_total):
        """ save the input file before running """
        with h5py.File(file_name+'.h5', 'w') as f:
            f.create_dataset('array', (sim_steps, sim_total),
                         dtype=np.double,
                         data=self.I)################################################################################
                                                                                                #### THIS IS WRONG AND I ASSUME IS OVER COMPENSATING FOR A DT*  issue!
    #   __
    def __save_network(self,file_name):
        """ save the network file before running """
        nx.write_gexf(self.network, file_name +'.gexf.gz')

    #   __
    def standard_neurons(self,model):
        """ provide the base neuron parameters from neurokernel, which are not in SpineML """
        """ DEPRECIATED TO ALLOW C_GENERATION 
            URL is used to load the correct kernel 
            WIP: Automatic discovery of extern, spiking and public based on component and connections
            external is true, to work with input generation, this will not scale well
      
        """
        return {'model': 'SpineMLNeuron','name': 'neuron_x','extern': True,'public': False,'spiking': True,'selector': '/a[0]','V':0,"url":model}


    def standard_synapses(self,model):
        """ provide the base synapse parameters from neurokernel, which are not in SpineML
        From the NK LPU comments:

         All synapses must have the following attributes:

        1. class - int indicating connection class of synapse; it may assume the
           following values:
           0. spike to spike synapse
           1. spike to graded potential synapse
           2. graded potential to spike synapse
           3. graded potential to graded potential synapse
        2. model - model identifier string, e.g., 'AlphaSynapse'
        3. conductance - True if the synapse emits conductance values, False if
           it emits current values.
        4. reverse - If the `conductance` attribute is T
        """

        if (model == 'AlphaSynapse.xml'): # Needs to be more generic
            print "Warning: AlphaSynapse Model may not be 100% Specified"
            return {'model': 'AlphaSynapse','name': 'alpha_synapse_default','class': 0,'ar': 1.1*1e2,'ad': 1.9*1e3,'reverse': 65*1e-3,'gmax': 2*1e-3,'conductance': True}
        elif (model == 'ExpSynapse.xml'):
            print "Warning: ExpSynapse Model may not be 100% Specified"
            return {'model': 'ExpSynapse','name': 'exp_synapse_default','class': 0,'a': 1,'tau': 10,'reverse': 0, 'gmax': 2*1e-3, 'eff':0, 'conductance': True}

        else:
            print "Warning: No Synapse Template Used ################"
            return {'model': 'Unknown','name': 'Unknown','class': 0, 'conductance': False}

    def get_units(self,unit):
        """ resolve dimentsions """
        units = {
         'GV': 1*(10**9),'MV': 1*(10**6),'kV': 1*(10**3),'V' : 1,'cV': 1*(10**-1),'mV': 1*(10**-3),'uV': 1*(10**-6),'nV': 1*(10**-9),'pV': 1*(10**-12),'fV': 1*(10**-15),
         'GOhm': 1*(10**9),'MOhm': 1*(10**6),'kOhm': 1*(10**3),'Ohm' : 1,'cOhm': 1*(10**-1),'mOhm': 1*(10**-3),'uOhm': 1*(10**-6),'nOhm': 1*(10**-9),'pOhm': 1*(10**-12),'fOhm': 1*(10**-15),
         'GA': 1*(10**9),'MA': 1*(10**6),'kA': 1*(10**3),'A' : 1,'cA': 1*(10**-1),'mA': 1*(10**-3),'uA': 1*(10**-6),'nA': 1*(10**-9),'pA': 1*(10**-12),'fA': 1*(10**-15),
         'GF': 1*(10**9),'MF': 1*(10**6),'kF': 1*(10**3),'F' : 1,'cF': 1*(10**-1),'mF': 1*(10**-3),'uF': 1*(10**-6),'nF': 1*(10**-9),'pF': 1*(10**-12),'fF': 1*(10**-15),
         'GS': 1*(10**9),'MS': 1*(10**6),'kS': 1*(10**3),'S' : 1,'cS': 1*(10**-1),'mS': 1*(10**-3),'uS': 1*(10**-6),'nS': 1*(10**-9),'pS': 1*(10**-12),'fS': 1*(10**-15),'?':1
        }
        return(units[unit])



    #   __
    def gen_value(self,params,index=0):
        """ convert a standard parameter set into a specific parameter """

        try:
            itype=  params['input']['type']
            if (itype == 'ValueList'):
                values =  params['input']['value']
                for v in values:
                    value = 0
                    if v['index'] == index:
                        value = v['value']

            if (itype == 'FixedValue'):
                value =  params['input']['value']

            if (itype == 'UniformDistribution'):
                np.random.seed(seed=params['input']['seed'])
                value = np.random.uniform(params['input']['minimum'],params['input']['maximum'])

            if (itype == 'NormalDistribution'):
                np.random.seed(seed=params['input']['seed'])
                value = np.random.normal(params['input']['mean']*1.0,params['input']['variance']*1.0)


            if (itype == 'PoissonDistribution'):
                np.random.seed(seed=params['input']['seed'])
                value = np.random.poisson(params['input']['mean'])


            value = value * self.get_units(params['dimension'])


        except:
            #print "warning: no input found, assuming zero"
            value = 0
        return value

    #   __
    def add_neuron(self,model,params,lpu_index,p_index,pop_name):
        """ add a neuron to the gexf population,
            where p_index is the neuron index within a population
        """
        neuron = self.standard_neurons(model)
        for p in params:
            """  p example: 'C': {'dimension': 'nS','input':{'type':'FixedValue','value':1}} """
            neuron[p] = self.gen_value(params[p],p_index)
        neuron['name'] =     'neuron_' +str(lpu_index)  # + '_' + str(p_index)
        neuron['selector'] = '/'+pop_name+'[' +str(lpu_index) +']'    #+ '[' + str(p_index)+']'
       
        print lpu_index
        self.network.add_node(lpu_index,attr_dict=neuron)
        print self.network.node[lpu_index]

    #   __ static
    def poisson(self,dt,freq,samples):
        """ return a poisson spike train """
        return  np.random.uniform(size = samples)<dt*freq

    def regular(self,duration,steps,frequency):
        """ return a regular spike train """
        step_jump = (duration/frequency) / (duration/steps)
        a= np.zeros(steps);
        a[1::int(step_jump)]
        return a

    def rate_based_distrobution(self, distrobution, duration,steps,frequency):
        """ filter between distrobutions """

        if distrobution == 'regular':
            return regular(duration, steps,frequency)
        elif distrobution == 'poisson':
            return poisson(duration, steps,frequency)
        else:
            log("Error","Rate Distrobution is not correctly defined")

    def log(self,level,message):
        """ expandable log function """
        print Error + ' ' + message

    def TimeVaryingInput(self,params,lpu_start,lpu_size):
        tp_values = params['TimePointValue']                  #  TODO
        if 'target_indicies' not in params:
            params['target_indicies'] = np.arange(lpu_size)
        
        for tp_value in tp_values:
            #for every array time
            if 'rate_based_distribution' in params:
                select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                for i in params['target_indicies']:
                    self.I[select,lpu_start+i] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),int(tp_value['value']))
            else:
                if 'value' in tp_value:
                    for i in params['target_indicies']:
                        self.I[self.t>tp_value['time'],lpu_start:lpu_start+lpu_size] =  (tp_value['value']);

                else: # Single spike input
                    for i in params['target_indicies']:
                        self.I[self.t==tp_value['time'],lpu_start+i] =  1

    def ConstantArrayInput(self,params,lpu_start,lpu_size):

        if not('start_time' in params):
            params['start_time'] = 0
        if not( 'duration' in params):
            end_time = np.max(self.t)
        else:
            end_time = params['start_time'] + params['duration']
            if end_time > np.max(self.t):
                end_time = np.max(self.t)

        if 'rate_based_distribution' in params:
            for i in np.arange(params['array_size']):
                select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                self.I[select,lpu_start+i] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),params['array_value'][i])

        else:

            for i in np.arange(params['array_size']):
                self.I[np.logical_and(self.t>params['start_time'],self.t<end_time),lpu_start+i] =  params['array_value'][i];




    def ConstantInput(self,params,lpu_start,lpu_size):
        if not( 'start_time' in params):
            params['start_time'] = 0
        if not( 'duration' in params):
            end_time = np.max(self.t)
        else:
            end_time = params['start_time'] + params['duration']
            if end_time > np.max(self.t):
                end_time = np.max(self.t)

        if 'rate_based_distribution' in params:
            if ('target_indicies' in params):
                for i in params['target_indicies']:
                    select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                    self.I[select,lpu_start+i] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),params['value'])
            else:
                for i in np.arange(lpu_size):
                    select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                    self.I[select,lpu_start+i] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),params['value'])
        else:
            if ('target_indicies' in params):
                for i in params['target_indicies']:
                    self.I[np.logical_and(self.t>params['start_time'], self.t<end_time),lpu_start+i] =  params['value']
            else:
                for i in np.arange(lpu_size):
                    self.I[np.logical_and(self.t>params['start_time'], self.t<end_time),lpu_start+i] =  params['value']

    def TimeVaryingArrayInput(self,params,lpu_start,lpu_size):
        tpa_values = params['TimePointArrayValue']                  #  TODO
        for tpa_value in tpa_values:
            #for every array time
            if 'rate_based_distribution' in params:
                for time, value in zip(tpa_value['array_time'],tpa_value['array_value']):
                    select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                    self.I[select,lpu_start+int(tpa_value['index'])] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),value)

            else:
                if 'array_value' in tpa_value:
                    for time, value in zip(tpa_value['array_time'],tpa_value['array_value']):
                        self.I[self.t>time,lpu_start+int(tpa_value['index'])] = value
                        print "Warning: Input does not contain input units"

                else: # Single spike input
                    for time in tpa_value['array_time']:
                        self.I[self.t==time,lpu_start+int(tpa_value['index'])] = 1
                        print "Warning: Input does not contain input units"

    def initialise_input(self,params,lpu_start,lpu_size):
        """ initialise an input in the matrix for a given input to a population  """

        itype=  params['type']

        if (itype == 'TimeVaryingArrayInput'):
            self.TimeVaryingArrayInput(params,lpu_start,lpu_size)

        if (itype == 'ConstantInput'):
            self.ConstantInput(params,lpu_start,lpu_size)

        if (itype == 'ConstantArrayInput'):
            self.ConstantArrayInput(params,lpu_start,lpu_size)

        if (itype == 'TimeVaryingInput'):
            self.TimeVaryingInput(params,lpu_start,lpu_size)

    def initialise_projections(self):
        """ initialise the projections, adding synapses to the network """
        for p in self.projections:
            # check we are using exp_syn for now
            # check we have all the variables needed for exp_syn ready
            # check the populations exist
            p_index = 0
            for s in p.synapse_list:
                # Create the Synapse
                synapse = self.standard_synapses(s['postsynapse']['url'])
                params = s['postsynapse']['parameters']
                for v in params:
                   synapse[v] = self.gen_value(params[v],p_index)
                p_index += 1;
                # Connect the edges with the synapse
                self.connect_projection(p.source, p.destination, s, synapse)

    def connect_projection(self,source,destination,synapse_info,synapse_model):
        """ connect projections based on synapse connectivity scheme"""
        #get population size
        s_size = self.populations[source].size
        s_start = self.populations[source].start
        d_size = self.populations[destination].size
        d_start = self.populations[destination].start


        if synapse_info['type'] == 'OneToOneConnection':
            try:
                for s,d in zip(np.arange(s_start,s_start+s_size),np.arange(d_start,d_start+d_size)):
                    self.network.add_edge(s, d, type='directed',attr_dict=synapse_model)
            except:
                print "One to One Connection Failed to be established"

        if synapse_info['type'] == 'AllToAllConnection':
            try:
                for s in np.arange(s_start,s_start+s_size):
                    for d in np.arange(d_start,d_start+d_size):
                        self.network.add_edge(s, d, type='directed',attr_dict=synapse_model)
            except:
                print "All to All Connection Failed to be esablished"

        if synapse_info['type'] == 'FixedProbabilityConnection':
            try:
                np.random.seed(synapse_info['seed'])
                for s in np.arange(s_start,s_start+s_size):
                    for d in np.arange(d_start,d_start+d_size):
                        if np.random.rand() <=  synapse_info['probability']:
                            self.network.add_edge(s, d, type='directed',attr_dict=synapse_model)

            except:
                print "FixedProbabilityConnection Failed to be esablished"

        if synapse_info['type'] == 'ConnectionList':
            try:
                for c in synapse_info['connections']:
                    self.network.add_edge(c[0], c[1], type='directed',attr_dict=synapse_model)

            except:
                print "Error: ConnectionList Failed to be esablished"

    def __initialise(self):
        """translate self.populations and self.inputs to gexf and h5 files"""

        # Initialise Input Array

        sim_dt = self.simulation['dt']         # in miliseconds
        sim_dur =  self.simulation['duration'] # in miliseconds

        sim_steps = int(sim_dur/sim_dt)
        self.simulation['steps'] = sim_steps
        sim_total = self.neurons

        self.t = np.arange(0, sim_dur, sim_dt)
        self.I = np.zeros((sim_steps, sim_total), dtype=np.double)

        lpu_index = 0;
        for p in self.populations:
            lpu_start = lpu_index;          # Start position for each neuron
            pop = self.populations[p]
            for n in np.arange(0,pop.size):
                self.add_neuron(pop.model,pop.params,lpu_index,n,pop.name)
                lpu_index +=1
            if pop.name in self.inputs:
                for i in self.inputs[pop.name]:
                    self.initialise_input(i,lpu_start,pop.size)

        self.initialise_projections()


        network_file = self.name
        input_file = self.name + "_input"

        self.__save_input(input_file,sim_steps,sim_total)
        self.__save_network(network_file)

        self.execute(network_file,input_file)

    def execute(self,network_file,input_file):


        file_name = 'spineml.log'

        screen = True
        logger = setup_logger(file_name=file_name, screen=screen)
        """
        coms = {}

        ### Need to integrate XSLT componants 
        con = nk_component.Condition("integrating")
        con.add_assignment("V","Vr")
        con.add_trigger("V > Vt")
        con.add_event("spike")

        con2 = nk_component.Condition("integrating")
        con2.add_assignment("V","Vr")
        con2.add_trigger("V > -0.02")
        con2.add_event("spike")

        reg = nk_component.Regime("integrating")
        reg.add_derivative("V","((I_Syn/C)+(Er - V)) / (R*C)")   
        reg.add_condition(con)                                  
        
        reg2 = nk_component.Regime("integrating")
        reg2.add_derivative("V","((I_Syn/C)+(Er - V)) / (R*C)")   
        reg2.add_condition(con2)                                  


        com = nk_component.Component("neuron_body")
        
        com.add_regime(reg,True)

        com.add_state_variable("V","mV")

        com.add_send_port("Event","spike")
        com.add_recieve_port("AnalogReduce","I_Syn","mA","+")

        com.add_parameter("C","nS")
        com.add_parameter("Vt","mV")
        com.add_parameter("Er","mV")
        com.add_parameter("Vr","mV")
        com.add_parameter("R","MOhm")


        com2 = nk_component.Component()
        
        com2.add_regime(reg2,True)

        com2.add_state_variable("V","mV")

        com2.add_send_port("Event","spike")
        com2.add_recieve_port("AnalogReduce","I_Syn","mA","+")

        com2.add_parameter("C","nS")
        com2.add_parameter("Vt","mV")
        com2.add_parameter("Er","mV")
        com2.add_parameter("Vr","mV")
        com2.add_parameter("R","MOhm")

        coms['LeakyIAF.xml'] = com

        coms['LeakyIAF.xml2'] = com2
        """
        data_dir = './'

        man = Manager()
        (n_dict, s_dict) = LPU.lpu_parser(data_dir+network_file+'.gexf.gz')


        man.add(LPU, 'spine_ml', self.simulation['dt'], n_dict, s_dict,
                input_file=data_dir + input_file+'.h5',
                output_file=data_dir +network_file+'_output.h5',
                device=0, debug=True,components=self.components)

        man.spawn()
        man.start(self.simulation['steps'])
        man.wait()

    def run(self):
        # Create Network
        self.__initialise()

        # Create Input



