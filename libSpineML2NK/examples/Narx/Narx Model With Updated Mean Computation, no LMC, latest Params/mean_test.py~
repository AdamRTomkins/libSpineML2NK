"""
A script to compare the nk_model mean generation, to a python implementation and the original data set

Python and NK match, with the original data being too high. 
I propose this is either a bad parameter set, or a different method used to calculate the mean.

"""
from matplotlib import pyplot as plt
import h5py


nk_mean = h5py.File('RealInput_mu_m.h5')['array'][:]
mutant = io.loadmat('../data/MutantBG6Data.mat')

m_input = mutant['recorded_input'][:,0]

mean_calc = m_input *0

Am = 0.0012
Bm = -0.9975

for u in (np.arange(1,len(m_input))):
    mean_calc[u] = (Am*(m_input[u]+m_input[u-1]))-(Bm*(mean_calc[u-1]))

mean_calc_1 = mean_calc
mean_calc = m_input *0

Am = 0.0010
Bm = -0.9979

for u in (np.arange(1,len(m_input))):
    mean_calc[u] = (Am*(m_input[u]+m_input[u-1]))-(Bm*(mean_calc[u-1]))


#plt.plot(m_input)
plt.plot(mean_calc_1, label = "Python Thesis Params")
plt.plot(mean_calc,label = "Python Old Params")
plt.plot(np.arange(4100,len(nk_mean)+4100),nk_mean,label = "Neurokernel")
plt.plot(mutant['Computed_Mean'][:,0],label = "Provided Data")
plt.legend()
plt.show()
