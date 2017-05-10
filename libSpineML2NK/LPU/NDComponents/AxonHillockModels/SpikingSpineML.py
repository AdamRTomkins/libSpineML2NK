
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#from BaseAxonHillockModel import BaseAxonHillockModel
#from libSpineML2NK.NDComponents.AxonHillockModels import BaseAxonHillockModel
from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import BaseAxonHillockModel

from libSpineML.smlComponent import AnalogReducePortType
from libSpineML.smlComponent import AnalogSendPortType
from libSpineML.smlComponent import EventSendPortType

from libSpineML2NK import nk_utils

class SpikingSpineML(BaseAxonHillockModel):

    
    updates = []

    step = 0
   
    accesses = []
    
    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=False,SpineMLComponents=None):

        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        # TODO:            
        self.num_comps = params_dict['extern'].size   #TODO: Take from size in dict?
        self.params_dict = params_dict

        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = 'double' # params_dict['V'].dtype #  # TODO: Take from a standard?
        self.ddt = self.dt#/self.steps TODO: What?
        self.access_buffers = access_buffers

        # TODO: Send in a class, and only one!
        self.SpineMLComponent = SpineMLComponents[SpineMLComponents.keys()[0]]
    
        # TODO: WIP, assumes one SpineMLComponent
        self.params = [p.name for p in self.SpineMLComponent.ComponentClass.Parameter]

        # This sets the initial internal variables.
        # TODO: This sets all initial values equally? FIX!
        self.internals = OrderedDict([(s.name,params_dict[s.name][0].get()) for s in self.SpineMLComponent.ComponentClass.Dynamics.StateVariable])

        self.updates = []
        self.accesses = []
        self.accesses_dimensions = {}

        for p in self.SpineMLComponent.ComponentClass.Port:
            assert type(p) in [AnalogReducePortType, AnalogSendPortType,EventSendPortType]

            if type(p) == AnalogReducePortType:
                self.accesses.append(p.name)
                self.accesses_dimensions[p.name] = p.dimension
            if type(p) == AnalogSendPortType:
                self.updates.append(p.name)
            if type(p) == EventSendPortType:
                self.updates.append(p.name)
    
        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        # TODO:   ??     
        dtypes = {'dt': self.dtype}
        dtypes.update({k: self.inputs[k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[k].dtype for k in self.params})
        # Add Internals In        
        dtypes.update({"internal_"+k: self.internal_states[k].dtype for k in self.internals})
        # TODO: Generalise to Event Output Ports
        dtypes.update({k: self.dtype if not k == 'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)


    def pre_run(self, update_pointers):

        for k in update_pointers.keys():
            cuda.memcpy_dtod(int(update_pointers[k]),
                             self.params_dict[k].gpudata,
                             self.params_dict[k].nbytes)

    # TODO:
    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)
        
        for k in self.accesses:
            print "Input = " + str(self.inputs[k].get())
            
            print "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDd"
        self.update_func.prepared_async_call(
            self.update_func.grid, 
            self.update_func.block, 
            st,
            self.num_comps, 
            self.ddt, # TODO: was times 1000! 
            self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    # TODO:        
    def get_update_template(self):


        signiture = []
        for a in self.accesses:
            signiture.append("    %(type)s* g_%(a)s, // inputs\n" % {'type':self.type_dict[a], 'a':a} )        
        for s in self.params:
            signiture.append("    %(type)s* g_%(s)s, // state variables\n" % {'type':self.type_dict[s], 's':s})


        for i in self.internals.keys():
            signiture.append("    %(type)s* internal_g_%(i)s, // internals\n" % {'type':self.type_dict[i], 'i':i})

        for i,u in enumerate(self.updates):
            if i < len(self.updates)-1:
                signiture.append("    %(type)s* g_%(u)s, // outputs\n" % {'type':self.type_dict[u], 'u':u})
            else:             signiture.append("    %(type)s* g_%(u)s // outputs\n" % {'type':self.type_dict[u], 'u':u})
    
             
        signiture = ''.join(signiture)        

        declaration = []

        for u in self.updates:
            declaration.append("    %(type)s %(u)s; // updates\n" % {'type':self.type_dict[u], 'u':u})

        for a in self.accesses:
            declaration.append("    %(type)s %(a)s; // accesses\n" % {'type':self.type_dict[a], 'a':a})
        # Add internals
        for i in self.internals.keys():        
            declaration.append("    %(type)s internal_%(i)s; // internals\n" % {'type':self.type_dict[i], 'i':i})


        for s in self.params:
            declaration.append("    %(type)s %(s)s; // state variables\n" % {'type':self.type_dict[s], 's':s})

        declaration = ''.join(declaration)        

        temp_state_variable_assignment = []

        for s in self.params:
            temp_state_variable_assignment.append(s +"= g_"+s+"[i]; // state variables\n" % self.type_dict)

        for I in self.internals:
            temp_state_variable_assignment.append(I +"= internal_g_"+I+"[i]; // internals\n" % self.type_dict)

        temp_state_variable_assignment = ''.join(temp_state_variable_assignment)


        temp_output_assignment = []

        for a in self.updates:
           temp_output_assignment.append(a +"= internal_g_"+a+"[i]; //outputs\n")

        temp_output_assignment = ''.join(temp_output_assignment)

        temp_final_output_assignment = []

        for a in self.updates:
           temp_final_output_assignment.append("g_"+ a +"[i]= "+a+ "; // final outputs\n")

        for I in self.internals.keys():
           temp_final_output_assignment.append("internal_g_"+ I +"[i]= "+I+ "; // final internal  outputs\n")


        temp_final_output_assignment = ''.join(temp_final_output_assignment)

        temp_input_assignment = []

        for a in self.accesses: # Scale this to the right unit!
           #TODO
           #assert False

           unitScale = nk_utils.units(1, self.accesses_dimensions[a])
           print "Input Unit = %s, in %s" % (unitScale , self.accesses_dimensions[a])
           temp_input_assignment.append(a +"= g_"+a+"[i] * %(unitScale)s; // scaled inputs\n" % {'unitScale':unitScale})

        temp_input_assignment = ''.join(temp_input_assignment)

        ### TODO: Expand to correctly allow several regimes
       
        ## Regime Code
        for reg in self.SpineMLComponent.ComponentClass.Dynamics.Regime:
            reg_str = ""
            for td in reg.TimeDerivative:
                """ create component time derivitives """
                reg_str = reg_str + '%(par)s = %(par2)s+ (dt * (%(math)s));'  % {"par":td.variable,"par2":td.variable,"math":td.MathInline}

            
            con_str = ""
            for con in reg.OnCondition:
                """ create component conditions """
                ass_str = ""

                for ass in con.StateAssignment:
                    ass_str = ass_str + "%(par)s = %(math)s;" % {"par":ass.variable,"math":ass.MathInline}
                
                for eve in con.EventOut:
                    ass_str = ass_str +  eve.port + "= 1;"                
                con_str= "if( %(trigger)s ){ %(assigns)s}" % {"trigger":con.Trigger.MathInline,"assigns":ass_str}
        
        template = """
__global__ void update(int num_comps, %(dt)s dt, int nsteps,
               %(signiture)s)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(declaration)s

    for(int i = tid; i < num_comps; i += total_threads)
    {

        // Assign State Variables to temporary variables
        %(temp_state_variable_assignment)s

        // Assign output to temporary values
        %(temp_output_assignment)s

        // Assign inputs to temporary values
        %(temp_input_assignment)s

        
        %(reg_str)s
        %(con_str)s

        %(temp_final_output_assignment)s

    }
}
        """ % { 'dt':'double',
                'signiture':signiture,
                'declaration':declaration,
                'temp_state_variable_assignment':temp_state_variable_assignment,
                'temp_output_assignment':temp_output_assignment,
                'temp_input_assignment':temp_input_assignment,
                'reg_str':reg_str,
                'con_str':con_str,
                'temp_final_output_assignment':temp_final_output_assignment,
               }

        print template
        return template

    # TODO:
    def get_update_func(self, dtypes):
        from pycuda.compiler import SourceModule

        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        self.type_dict = type_dict

        type_dict.update({'fletter': 'f' if type_dict[type_dict.keys()[0]] == 'float' else ''})

        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)

        func = mod.get_function("update")

        prep = 'i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2)        
        func.prepare(prep)
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) / 256 + 1), 1)
        return func



if __name__ == '__main__':
    x=0
    if x == 1: 
        import argparse
        import itertools
        import networkx as nx
        from neurokernel.tools.logging import setup_logger
        import neurokernel.core_gpu as core



        from neurokernel.LPU.LPU import LPU

        from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
        from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
        from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

        import neurokernel.mpi_relaunch

        dt = 1e-4
        dur = 1.0
        steps = int(dur/dt)

        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', default=False,
                            dest='debug', action='store_true',
                            help='Write connectivity structures and inter-LPU routed data in debug folder')
        parser.add_argument('-l', '--log', default='none', type=str,
                            help='Log output to screen [file, screen, both, or none; default:none]')
        parser.add_argument('-s', '--steps', default=steps, type=int,
                            help='Number of steps [default: %s]' % steps)
        parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                            help='GPU device number [default: 0]')
        args = parser.parse_args()

        file_name = None
        screen = False
        if args.log.lower() in ['file', 'both']:
            file_name = 'neurokernel.log'
        if args.log.lower() in ['screen', 'both']:
            screen = True
        logger = setup_logger(file_name=file_name, screen=True)

        man = core.Manager()

        G = nx.MultiDiGraph()

        G.add_node('neuron0', {
                   'class': 'SpikingSpineML',
                   'name': 'LeakyIAF',
                   'resting_potential': -70.0,
                   'threshold': -45.0,
                   'capacitance': 0.07, # in mS
                   'resistance': 0.2, # in Ohm
                   'reset_potential':0,
                   })

        comp_dict, conns = LPU.graph_to_dicts(G)

        fl_input_processor = StepInputProcessor('I_Syn', ['neuron0'], 40, 0.2, 0.8)
        fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

        man.add(LPU, 'ge', dt, comp_dict, conns,
                device=args.gpu_dev, input_processors = [fl_input_processor],
                output_processors = [fl_output_processor], debug=args.debug,extra_comps=[SpikingSpineML])

        man.spawn()
        man.start(steps=args.steps)
        man.wait()

    else:
        import neurokernel.mpi_relaunch
        from libSpineML2NK import nk_executable
        e = nk_executable.Executable('./test/experiment0.xml')
        e.execute()



