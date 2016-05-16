
from libSpineML import smlNetwork
import numpy as np
import pdb

def gen_value(prop,index=0):
    """ convert a standard parameter set into a specific parameter """

    #try:
    
    if not prop.AbstractValue == None:
        #Fixed Value
        if type(prop.AbstractValue) == smlNetwork.FixedValueType:
            value = prop.AbstractValue.value

        elif type(prop.AbstractValue) == smlNetwork.ValueListType:
            values =  prop.AbstractValue.Value
            value = values[index].value
        else:
            raise TypeError('Property Abstract Value type: %s not recognized' % str(type(prop.AbstractValue)))                
        
    elif not prop.AbstractDistribution == None:    
        np.random.seed(seed=prop.AbstractDistribution.seed + index) 
        if type(prop.AbstractDistribution) == smlNetwork.UniformDistributionType:
            value = np.random.uniform(prop.AbstractDistribution.minimum,prop.AbstractDistribution.maximum)

        elif type(prop.AbstractDistribution) == smlNetwork.NormalDistributionType:
           value = np.random.normal(prop.AbstractDistribution.mean,prop.AbstractDistribution.variance)

        elif type(prop.AbstractDistribution) == smlNetwork.PoissonDistributionType:
            value = np.random.poisson(prop.AbstractDistribution.mean)
        else:
            raise TypeError('Property Abstract Distribution type: %s not recognized' % str(type(prop.AbstractDistribution)))
    else:
        raise TypeError('Property type: %s not recognized' % prop.name)
        
    #except:
    #    raise TypeError('Value type: %s not recognized' % str(type(prop)))
           
    return units(value,prop.dimension)

def units(value,unit):
    """ Return a value in standard units
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

def poisson(dt,freq,samples):
    """ return a poisson spike train """
    return  np.random.uniform(size = samples)<dt*freq

def regular(duration,steps,frequency):
    """ return a regular spike train """
    step_jump = (duration/frequency) / (duration/steps)
    a= np.zeros(steps);
    a[1::int(step_jump)]
    return a

def rate_based_distrobution(distribution, duration,steps,frequency):
    """ filter between distrobutions """

    if distribution == 'regular':
        return regular(duration, steps,frequency)
    elif distribution == 'poisson':
        return poisson(duration, steps,frequency)
    else:
        log("Error","Rate Distrobution is not correctly defined")




def TimeVaryingInput(params,lpu_start,lpu_size,time,inputs):
    """
    time should be in seconds to keep standard units
    WIP: trasnlate all times in this to seconds
    """
    tp_values = params.TimePointValue                 
    if params.target_indices is None:
        params.target_indices = np.arange(lpu_size)
    if params.start_time is None:
        params.start_time = 0
    if params.duration is None:
        params.duration = max(time)

    for tp_value in tp_values:
        
        """if params.rate_based_distribution is not None:
           # REQUIRES TESTING
            select = np.logical_and(time>=tp_value.time,time<=tp_value.time+params.duration)    
            for i in params.target_indices:
                inputs[select,lpu_start+i] = rate_based_distrobution(params.rate_based_distribution,params.duration,np.sum(select),int(tp_value.value))
        
        else:
        """
        if tp_value.value is not None:
            
            for i in params.target_indices:
                select = np.logical_and(time>=units(tp_value.time,'mS'),time<=units(tp_value.time,'mS')+params.duration)           
                inputs[select,lpu_start:lpu_start+lpu_size] =  tp_value.value

        else: # Single spike input
            for i in params.target_indices:
                inputs[time==tp_value.time,lpu_start+i] =  1
    return inputs



def ConstantArrayInput(params,lpu_start,lpu_size,time,inputs):

    if params.start_time is None:
        params.start_time = 0
    if params.duration is None:
        params.duration = max(time)

    #if 'rate_based_distribution' in params:
    # Requires Testing
    """
        for i in np.arange(params['array_size']):
            select = np.logical_and(time>=tp_value.time,time<=tp_value.time+params.duration)           
            inputs[select,lpu_start+i] = rate_based_distrobution(params.rate_based_distribution,params.duration,np.sum(select),params.array_value[i])

    else:
    """
    for i in np.arange(params.array_size):
        select = np.logical_and(time>=units(params.start_time,'mS'),time<=units(params.start_time,'mS')+params.duration)    
        inputs[select,lpu_start+i] =  params.array_value[i];

    return inputs

def ConstantInput(params,lpu_start,lpu_size,time,inputs):
    if params.target_indices is None:
        params.target_indices = np.arange(lpu_size)
    if params.start_time is None:
        params.start_time = 0
    if params.duration is None:
        params.duration = max(time)

    #if 'rate_based_distribution' in params:
    # Needs Testing
    """
        for i in params['target_indicies']:
            select = np.logical_and(self.t>params['start_time'], self.t<end_time)
            self.I[select,lpu_start+i] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),params['value'])

        """
    #    pass    
    #else:
    for i in params.target_indices:
        select = np.logical_and(time>=units(params.start_time,'mS'),time<=units(params.start_time,'mS')+params.duration)  
        inputs[select,lpu_start+i] =  params.value

    return inputs


def TimeVaryingArrayInput(params,lpu_start,lpu_size,time,inputs):

    tpa_values = params.TimePointArrayValue
             
    if params.target_indices is None:
        params.target_indices = np.arange(lpu_size)
    if params.start_time is None:
        params.start_time = 0
    if params.duration is None:
        params.duration = max(time)

    for tpa_value in tpa_values:
        #for every array time
        """
        if 'rate_based_distribution' in params:
            for time, value in zip(tpa_value['array_time'],tpa_value['array_value']):
                select = np.logical_and(self.t>params['start_time'], self.t<end_time)
                self.I[select,lpu_start+int(tpa_value['index'])] = rate_based_distrobution(params['rate_based_distribution'],params['duration'],np.sum(select),value)

        else:
        """
        
        if tpa_value.array_value is not None:
            
            for array_time, value in zip(tpa_value.array_time.split(','),tpa_value.array_value.split(',')):
               
                inputs[time>=units(float(array_time),'mS'),lpu_start+int(tpa_value.index)] = float(value)


        else: 
            for array_time in tpa_value.array_time.split(','):
                inputs[time==float(array_time),lpu_start+int(tpa_value.index)] = 1
    return inputs



