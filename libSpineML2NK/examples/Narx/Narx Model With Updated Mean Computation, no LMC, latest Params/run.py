import neurokernel.mpi_relaunch
import scipy.io as io
from libSpineML2NK import nk_executable
from libSpineML import smlExperiment
e = nk_executable.Executable('./experiment0.xml')

exp = e.bundle.experiments[0].Experiment[0]
ai = exp.AbstractInput[0]

mutant = io.loadmat('data/MutantBG6Data.mat')
m_input = mutant['recorded_input'][4099:12000,0]

net = e.bundle.networks[0]
pop = net.Population[0]
pop.Neuron.Property

test_params = False
if test_params:
    for prop in pop.Neuron.Property:
        if prop.name == 'Am':
            prop.AbstractValue.value = 0.01
        if prop.name == 'Bm':
            prop.AbstractValue.value = -0.9979



# Saturate Input! Nothing below 1e-10
m_input[m_input < 1e-10] =1e-10

#Rewrite Input Dynamically from mutant data
ai.TimePointValue = []      # Get rid of default input

for time, inj in enumerate(m_input):
    tp = smlExperiment.TimePointValueType(time=time,value=inj)
    ai.add_TimePointValue(tp)

exp.Simulation.duration = len(m_input)/1000.0

e.execute()


