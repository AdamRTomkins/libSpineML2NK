import neurokernel.mpi_relaunch
import scipy.io as io
from libSpineML2NK import nk_executable
from libSpineML import smlExperiment
e = nk_executable.Executable('./experiment0.xml')

exp = e.bundle.experiments[0].Experiment[0]
ai = exp.AbstractInput[0]

mutant = io.loadmat('data/MutantBG6Data.mat')
m_input = mutant['recorded_input'][4100:12000,0]

# Saturate Input! Nothing below 1e-10

m_input[m_input < 1e-10] =1e-10
ai.TimePointValue = []

#Rewrite Input Dynamically

time = 0.0
for inj in m_input:
    tp = smlExperiment.TimePointValueType(time=time,value=inj)
    ai.add_TimePointValue(tp)
    time = time +1

exp.Simulation.duration = len(m_input)/1000.0

e.execute()


