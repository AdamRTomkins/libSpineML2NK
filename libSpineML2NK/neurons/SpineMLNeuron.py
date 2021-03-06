"""
WIP:    Multiple Regimes
        Allow Spiking and GPot
        Port handling
            EventSend Ports are seen as a spiking output here
            AnalogSend Ports are seen as gpot
        Time Derivitive Results

Restrictions of the model:
    Only one Input allowed
        Add more by         ## Expand this to allow for multiple input ports

    Only one output port is allowed
        Assumes Analog Send port is a gpot
        Assume Events are spikes

"""

from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from libSpineML2NK import nk_utils

from neurokernel.LPU.utils.simpleio import *
from libSpineML import smlComponent
#
# Only Applicable for spiking neurons
#

class SpineMLNeuron(BaseNeuron):

    def __init__(self, n_dict, output, dt, debug=False, LPU_id=None,component=None):
        self.num_neurons = len(n_dict['id'])
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU_id = LPU_id
        self.component = component.ComponentClass

        # For every state varible
        for parameter in self.component.Dynamics.StateVariable:
            exec("self.%(name1)s = garray.to_gpu( np.asarray( n_dict['%(name2)s'], dtype=np.float64 ))" % {'name1':parameter.name,'name2':parameter.name});
           
        self.output = output

        #for every parameter in component
        for parameter in self.component.Parameter:
            exec("self.%(name1)s = garray.to_gpu( np.asarray( n_dict['%(name2)s'], dtype=np.float64 ))" % {'name1':parameter.name,'name2':parameter.name});

        _num_dendrite_cond = np.asarray([n_dict['num_dendrites_cond'][i]
                                         for i in range(self.num_neurons)],
                                        dtype=np.int32).flatten()

        _num_dendrite = np.asarray([n_dict['num_dendrites_I'][i]
                                    for i in range(self.num_neurons)],
                                   dtype=np.int32).flatten()

        self._cum_num_dendrite = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite, dtype=np.int32))))
        self._cum_num_dendrite_cond = garray.to_gpu(np.concatenate((
                                    np.asarray([0,], dtype=np.int32),
                                    np.cumsum(_num_dendrite_cond, 
                                              dtype=np.int32))))
        self._num_dendrite = garray.to_gpu(_num_dendrite)
        self._num_dendrite_cond = garray.to_gpu(_num_dendrite_cond)
        # TODO: Remove tie to input I variable        
        self._pre = garray.to_gpu(np.asarray(n_dict['I_pre'], dtype=np.int32))
        self._cond_pre = garray.to_gpu(np.asarray(n_dict['cond_pre'],
                                                  dtype=np.int32))

        # TODO: Remove tie to V state Variable
        self._V_rev = garray.to_gpu(np.asarray(n_dict['reverse'],
                                               dtype=np.double))


        ## Expand this to allow for multiple input ports
        self.I = garray.zeros(self.num_neurons, np.double)

        self._update_I_cond = self._get_update_I_cond_func()
        self._update_I_non_cond = self._get_update_I_non_cond_func()
        self.update = self.get_gpu_kernel()



        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array",
                                     tables.Float64Atom(), (0,self.num_neurons))

            # Set up state variables for debugging
            # every varible is recorded here
            for variable in self.component.Dynamics.StateVariable:
                exec("self.%s_file = tables.openFile(self.LPU_id + '_%s.h5', mode='w')" % (variable.name,variable.name));
                exec("self.%s_file.createEArray('/','array', tables.Float64Atom(), (0,self.num_neurons))" % (variable.name));


    def create_kernel(self):
        par_str = ""
        var_str = ""    #,r,c,er,vt,vr
        var_ass_str = "" #r = R[nid];c = C[nid];er = Er[nid];vt = Vt[nid];vr = Vr[nid];

        for parameter in self.component.Parameter:
            par_str = par_str+ ', %(type)s *%(par)s_s' % {"type": dtype_to_ctype(np.float64),"par": parameter.name}
            var_str = var_str+ '%(var)s,' % {"var": parameter.name}
            var_ass_str = var_ass_str+ '%(var)s = %(par)s_s[nid];' % {"var": parameter.name,"par": parameter.name}
        
        ## Send Port Code
        sp_dec = ""
        sp_save = ""
        sp_output_type = ""
        sp_zero_str = ""
    
        ## Recieve Port Code
        rp_dec_str = ""
        rp_ass_str = ""
        rp_tmp_dec_str = ""

        for p in self.component.Port: ### HERE

            

            if type(p) is smlComponent.EventSendPortType:
                sp_dec = sp_dec + "int %(sp)s;" %  {"sp": p.name}
                sp_zero_str = sp_zero_str + "%(sp)s = 0;" % {"sp": p.name} 
                sp_output_type = "int"   
                sp_save = sp_save + p.name

            elif type(p) is smlComponent.AnalogSendPortType:              
                sp_output_type = "double"
                sp_save = sp_save + p.name 
            

            if type(p) is smlComponent.AnalogReducePortType:
                # Note Traditional Use of Units
                unitScale = nk_utils.units(1,p.dimension)
                rp_dec_str = rp_dec_str + " %(type)s *%(rp)s_s" %  {"rp": p.name,"type": dtype_to_ctype(np.float64)}
                rp_ass_str = rp_ass_str + " %(rp)s = %(rp2)s_s[nid] * %(UnitScale)f;" %  {"rp": p.name,"rp2": p.name,"UnitScale":unitScale}
                rp_tmp_dec_str = rp_tmp_dec_str + "%(rp)s\n\t" % {"rp": p.name}  
            
        ## State Variable Code
        sv_dec_str = ""
        sv_tmp_dec_str = ""
        sv_ass_str = ""
        sv_save_str = ""

        for sv in self.component.Dynamics.StateVariable:
            sv_dec_str = sv_dec_str + ",%(type)s *%(sv)s_sv\n    " % {"type": dtype_to_ctype(np.float64),"sv": sv.name} 
            sv_tmp_dec_str = sv_tmp_dec_str + "%(sv)s,\n\t" % {"type": dtype_to_ctype(np.float64),"sv": sv.name}   
            sv_ass_str = sv_ass_str+ '%(sv_tmp)s = %(sv)s_sv[nid];' % {"sv_tmp": sv.name,"sv": sv.name}          
            sv_save_str = sv_save_str+ '%(sv)s_sv[nid] = %(sv_tmp)s;' % {"sv_tmp": sv.name,"sv": sv.name}    

        ### TODO: Expand to correctly allow several regimes
       
        ## Regime Code
        for reg in self.component.Dynamics.Regime:
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
        
        

        cuda_src = """
// %(type)s and %(nneu)d must be replaced using Python string foramtting
#define NNEU %(nneu)d
__global__ void spine_ml_neuron(
    int neu_num,
    %(type)s dt,   
    %(sp_output_type)s      *output             // Output type assignment
    %(svs_dec)s,                                // State Variable Assignment
    %(rp_dec)s                                  // Set up Inputs - Assume since recieve port for I    
    %(pars)s)                                   // Parameters
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;
    %(type)s %(svs_tmp_dec)s %(vars)s %(rp_tmp_vars)s;        // State Variable declarations and variables
    %(port_dec)s                                // Dynamic output port 
    
    if( nid < neu_num ){
        // Assign state variables to temporary vars
        %(svs_ass)s         
                    
        // Assign input to recieve port
        %(rp_ass_str)s                          

        // Assign variables from shared memory
        %(vars_ass)s                            
       
        // update differential
        %(td)s                                

        // zero our output (required for spiking)
        %(sp_zero)s

        // Conditions
        %(condition)s                                
                   
        // Save State Variables
        %(svs_save)s                        

        // Assumes one output variable
        output[nid] = %(port_save)s;        

    }
    return; 
}
""" % { "type": dtype_to_ctype(np.float64),
        "pars": par_str,    
        "svs_dec": sv_dec_str,
        "vars": var_str,
        "svs_tmp_dec":sv_tmp_dec_str,
        "svs_ass": sv_ass_str,
        "svs_save": sv_save_str, 
        "vars_ass": var_ass_str,
        "nneu": self.gpu_block[0],
        "td": reg_str,
        "condition":con_str,
        "port_dec" : sp_dec,
        "port_save" : sp_save,
        "sp_output_type":sp_output_type,
        "rp_dec": rp_dec_str,
        "rp_ass_str":rp_ass_str,
        "rp_tmp_vars": rp_tmp_dec_str,
        "sp_zero":sp_zero_str
    }
        return cuda_src

    @property
    def neuron_class(self): return True

    def eval(self, st=None):
        p_str = ""
        sv_str = ""
        for parameter in self.component.Parameter:
            p_str = p_str+ ',self.%s.gpudata' % parameter.name

        for variable in self.component.Dynamics.StateVariable:
            sv_str = sv_str+ ',self.%s.gpudata' % variable.name

        ## TODO: Expand this to allow for multiple input ports
        exec("self.update.prepared_async_call(self.gpu_grid,self.gpu_block,st,self.num_neurons,self.dt,self.output %(state_vars)s, self.I.gpudata %(params)s)" % {"params":p_str,"state_vars":sv_str});

        # Do dynamically based on the output in spineML experiment  
        if self.debug:
            ### TODO: Change to allow several input ports
            self.I_file.root.array.append(self.I.get().reshape((1, -1)))

            # State Variable debugging recording 
            for variable in self.component.Dynamics.StateVariable:
                exec("self.%s_file.root.array.append(self.%s.get().reshape((1, -1)))" % (variable.name,variable.name));
            

    def get_gpu_kernel( self):
        self.gpu_block = (128, 1, 1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)

        cuda_src = self.create_kernel()

        mod = SourceModule(
                cuda_src ,
                options=["--ptxas-options=-v"])
        func = mod.get_function("spine_ml_neuron")
        p_str = ""
        sv_str = ""
        for parameter in self.component.Parameter:
            p_str = p_str + ',np.intp'

        for variable in self.component.Dynamics.StateVariable:
            sv_str = sv_str+ ',np.intp'

        ### TODO: UPDATE FOR Various Inputs
        exec("func.prepare( [ np.int32, np.float64, np.intp %(sv)s, np.intp %(p)s ])" % {"sv":sv_str,"p":p_str});

        return func
        
    def post_run(self):
        ### Specify Outputs
        if self.debug:
            self.I_file.close()
            for variable in self.component.Dynamics.StateVariable:
                exec("self.%s_file.close()" % (variable.name));
           

    @property
    def update_I_override(self): return True

    def update_I(self, synapse_state, st=None):
        self.I.fill(0)
        if self._pre.size>0:
            self._update_I_non_cond.prepared_async_call(self._grid_get_input, 
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite.gpudata, self._num_dendrite.gpudata,
                self._pre.gpudata, self.I.gpudata)
        if self._cond_pre.size>0:
            self._update_I_cond.prepared_async_call(self._grid_get_input,     
                self._block_get_input, st, int(synapse_state),
                self._cum_num_dendrite_cond.gpudata, 
                self._num_dendrite_cond.gpudata,
                self._cond_pre.gpudata, self.I.gpudata, self.V.gpudata,
                self._V_rev.gpudata)
        
    ### CONDUCTANCE BASED SYNAPSES WILL CAUSE ISSUE IF State Variable is not V
    
    # Improve to remove I and include op
    def _get_update_I_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d
        __global__ void get_input(double* synapse, int* cum_num_dendrite, 
                                  int* num_dendrite, int* pre, double* I_pre, 
                                  double* V, double* V_rev)
        {
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;
            int neuron;
            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double V_in[32];
            __shared__ double input[32][33];
            
            if(tidy == 0)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                    V_in[tidx] = V[neuron];
                }
            }else if(tidy == 1)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }
            input[tidy][tidx] = 0.0;
            __syncthreads();
            neuron = bid * N + tidy;
            if(neuron < NUM_NEURONS)
            {
                int n_den = num_den[tidy];
                int start = den_start[tidy];
                double VV = V_in[tidy];
                for(int i = tidx; i < n_den; i += N)
                {
                   input[tidy][tidx] += synapse[pre[start + i]] * 
                                        (VV - V_rev[start + i]);
                }
            }
               __syncthreads();
               if(tidy < 8)
               {
                   input[tidx][tidy] += input[tidx][tidy + 8];
                   input[tidx][tidy] += input[tidx][tidy + 16];
                   input[tidx][tidy] += input[tidx][tidy + 24];
               }
               __syncthreads();
               if(tidy < 4)
               {
                   input[tidx][tidy] += input[tidx][tidy + 4];
               }
               __syncthreads();
               if(tidy < 2)
               {
                   input[tidx][tidy] += input[tidx][tidy + 2];
               }
               __syncthreads();
               if(tidy == 0)
               {
                   input[tidx][0] += input[tidx][1];
                   neuron = bid*N + tidx;
                   if(neuron < NUM_NEURONS)
                   {
                       I_pre[neuron] -= input[tidx][0];
                    }
               }
        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons}, 
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self._block_get_input = (32,32,1)
        self._grid_get_input = ((self.num_neurons - 1) / 32 + 1, 1)
        return func

    def _get_update_I_non_cond_func(self):
        template = """
        #define N 32
        #define NUM_NEURONS %(num_neurons)d
        __global__ void get_input(double* synapse, int* cum_num_dendrite, 
                                  int* num_dendrite, int* pre, double* I_pre)
        {
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;
            int neuron;
            __shared__ int num_den[32];
            __shared__ int den_start[32];
            __shared__ double input[32][33];
            if(tidy == 0)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    num_den[tidx] = num_dendrite[neuron];
                }
            }else if(tidy == 1)
            {
                neuron = bid * N + tidx;
                if(neuron < NUM_NEURONS)
                {
                    den_start[tidx] = cum_num_dendrite[neuron];
                }
            }
            input[tidy][tidx] = 0.0;
            __syncthreads();
            neuron = bid * N + tidy;
            if(neuron < NUM_NEURONS){
               int n_den = num_den[tidy];
               int start = den_start[tidy];
               for(int i = tidx; i < n_den; i += N)
               {
                   input[tidy][tidx] += synapse[pre[start + i]];
               }
            }
            __syncthreads();
            if(tidy < 8)
            {
                input[tidx][tidy] += input[tidx][tidy + 8];
                input[tidx][tidy] += input[tidx][tidy + 16];
                input[tidx][tidy] += input[tidx][tidy + 24];
            }
            __syncthreads();
            if(tidy < 4)
            {
                input[tidx][tidy] += input[tidx][tidy + 4];
            }
            __syncthreads();
            if(tidy < 2)
            {
                input[tidx][tidy] += input[tidx][tidy + 2];
            }
            __syncthreads();
            if(tidy == 0)
            {
                input[tidx][0] += input[tidx][1];
                neuron = bid*N+tidx;
                if(neuron < NUM_NEURONS)
                {
                    I_pre[neuron] += input[tidx][0];
                }
            }
        }
        //can be improved
        """
        mod = SourceModule(template % {"num_neurons": self.num_neurons}, 
                           options = ["--ptxas-options=-v"])
        func = mod.get_function("get_input")
        func.prepare([np.intp, np.intp, np.intp, np.intp, np.intp])
        return func



def get_units(unit):
        """ resolve dimentsions """
        units = {
         'GA': 1*(10**9),'MA': 1*(10**6),'kA': 1*(10**3),'A' : 1,'cA': 1*(10**-1),'mA': 1*(10**-3),'uA': 1*(10**-6),'nA': 1*(10**-9),'pA': 1*(10**-12),'fA': 1*(10**-15),'?':1
        }
        if unit in units:
            return(units[unit])
        else:
            return 1
