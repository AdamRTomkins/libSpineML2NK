import neurokernel.mpi_relaunch
import scipy.io as io
from libSpineML2NK import nk_executable
from libSpineML import smlExperiment
import numpy as np
import pdb
import pickle
e = nk_executable.Executable('./experiment0.xml')

exp = e.bundle.experiments[0].Experiment[0]
ai = exp.AbstractInput[0]

wild = io.loadmat('data/WildBG6Data.mat')

m_input = wild['recorded_input'][0:36000,0]
l = m_input.tolist()
l.insert(0,0)
m_input = np.array(l)


net = e.bundle.networks[0]
pop = net.Population[0]

pars = pickle.load( open( "data/wildPars.p", "rb" ) )

model     = io.loadmat("data/WildModel.mat")
theta     = np.ravel(model['wildNARXparamsBG6'])

# Overwrite default values with Python Script Values
for prop in pop.Neuron.Property:
    if prop.name in pars:
        prop.AbstractValue.value = pars[prop.name]
        print "Replaced %s" % prop.name
    else:
        # Set initial values:
        if prop.name == 'Contrast_ym_1':    
            prop.AbstractValue.value = pars['Contrast_X0_1']
            print "Replaced %s" % prop.name
        if prop.name == 'Contrast_ym_2':
            prop.AbstractValue.value = pars['Contrast_X0_2']
            print "Replaced %s" % prop.name
        if prop.name == 'Contrast_ym_3':
            prop.AbstractValue.value = pars['Contrast_X0_3']
            print "Replaced %s" % prop.name
        if prop.name == 'Mean_ym_1':
            prop.AbstractValue.value = pars['Mean_X0_1']
            print "Replaced %s" % prop.name
        if prop.name == 'Mean_ym_2':
            prop.AbstractValue.value = pars['Mean_X0_2']
            print "Replaced %s" % prop.name
        if prop.name == 'Mean_ym_3':
            prop.AbstractValue.value = pars['Mean_X0_3']
            print "Replaced %s" % prop.name
        if prop.name[0:2] == 'th':
            # Inject Narx Parameters
            num = int(prop.name[-(len(prop.name)-2): len(prop.name)])
            prop.AbstractValue.value = float(theta[num])
            print "Replaced %s" % theta[num]
   
 
    print prop.name + " is " + str(prop.AbstractValue.value) 

# Saturate Input! Nothing below 1e-10
m_input[m_input < 1e-10] =1e-10

#Rewrite Input Dynamically from mutant data
ai.TimePointValue = []      # Get rid of default input

for time, inj in enumerate(m_input):
    tp = smlExperiment.TimePointValueType(time=time,value=inj)
    ai.add_TimePointValue(tp)

exp.Simulation.duration = len(m_input)/1000.0

e.set_debug()
e.execute()

