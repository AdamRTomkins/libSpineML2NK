import scipy.io as io
import h5py
from matplotlib import pyplot as plt
import numpy as np


mutant = io.loadmat('data/MutantBG6Data.mat')
nk_input = h5py.File('RealInput_I.h5')['array'][:]

# Ys
y0_mean = h5py.File('RealInput_mean_y_0.h5')['array'][:]
y1_mean = h5py.File('RealInput_mean_y_1.h5')['array'][:]
y2_mean = h5py.File('RealInput_mean_y_2.h5')['array'][:]

y0_contrast = h5py.File('RealInput_contrast_y_0.h5')['array'][:]
y1_contrast = h5py.File('RealInput_contrast_y_1.h5')['array'][:]
y2_contrast = h5py.File('RealInput_contrast_y_2.h5')['array'][:]

ax = plt.subplot(2, 1, 1)
plt.title('ys mean')
plt.plot(y0_mean, label='y0')
plt.plot(y1_mean, label='y1')
plt.plot(y2_mean, label='y2')
ax.set_yscale('log')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('ys contrast')
plt.plot(y0_contrast, label='y0')
plt.plot(y1_contrast, label='y1')
plt.plot(y2_contrast, label='y2')
plt.legend()
ax.set_yscale('log')
plt.show()

# Fs
f0_mean = h5py.File('RealInput_mean_f_0.h5')['array'][:]
f1_mean = h5py.File('RealInput_mean_f_1.h5')['array'][:]
f2_mean = h5py.File('RealInput_mean_f_2.h5')['array'][:]

f0_contrast = h5py.File('RealInput_contrast_f_0.h5')['array'][:]
f1_contrast = h5py.File('RealInput_contrast_f_1.h5')['array'][:]
f2_contrast = h5py.File('RealInput_contrast_f_2.h5')['array'][:]

ax = plt.subplot(2, 1, 1)
ax.set_yscale('log')
plt.title('fs mean')
plt.plot(f0_mean, label='f0')
plt.plot(f1_mean, label='f1')
plt.plot(f2_mean, label='f2')

plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('Fs contrast')
plt.plot(f0_contrast, label='f0')
plt.plot(f1_contrast, label='f1')
plt.plot(f2_contrast, label='f2')
plt.legend()
ax.set_yscale('log')
plt.show()


# Gs
g0_mean = h5py.File('RealInput_mean_g_0.h5')['array'][:]
g1_mean = h5py.File('RealInput_mean_g_1.h5')['array'][:]
g2_mean = h5py.File('RealInput_mean_g_2.h5')['array'][:]

g0_contrast = h5py.File('RealInput_contrast_g_0.h5')['array'][:]
g1_contrast = h5py.File('RealInput_contrast_g_1.h5')['array'][:]
g2_contrast = h5py.File('RealInput_contrast_g_2.h5')['array'][:]

ax = plt.subplot(2, 1, 1)
plt.title('gs mean')
plt.plot(g0_mean, label='g0')
plt.plot(g1_mean, label='g1')
plt.plot(g2_mean, label='g2')
ax.set_yscale('log')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('Gs contrast')
plt.plot(g0_contrast, label='g0')
plt.plot(g1_contrast, label='g1')
plt.plot(g2_contrast, label='g2')
plt.legend()
ax.set_yscale('log')
plt.show()


# Input Differences
offset = 4000
nk_input = h5py.File('RealInput_I.h5')['array'][:]
m_input = mutant['recorded_input'][offset:offset+len(nk_input)]

ax = plt.subplot(2, 1, 1)
plt.title('inputs')
plt.plot(nk_input, label='Neurokernel')
plt.plot(m_input, label='Mutant')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('difference')
plt.plot(m_input-nk_input,label='Difference')
plt.legend()
plt.show()

# Output differences
m_output = mutant['MPO_Blocks'][offset:offset+len(nk_input),0] - np.mean( mutant['MPO_Blocks'][offset:offset+len(nk_input),0])
nk_output = h5py.File('RealInput_y.h5')['array'][:] - np.mean(h5py.File('RealInput_y.h5')['array'][:])

ax = plt.subplot(1, 1, 1)
ax.set_ylim([-100,100])

plt.title('Narx Output')
plt.plot(nk_output, label='Neurokernel')
plt.plot(m_output, label='Mutant')
plt.legend()

plt.show()

#########################

# Output differences
m_output = mutant['MPO_Blocks'][5000:5000+(len(nk_input)-1000),0] 

m_output = m_output - np.mean(m_output[-4000])

nk_output = h5py.File('RealInput_y.h5')['array'][1000:]

nk_output = nk_output - np.mean(nk_output[-4000])

ax = plt.subplot(1, 1, 1)
#ax.set_ylim([-100,100])

plt.title('Narx Output')
plt.plot(nk_output , label='Neurokernel')
plt.plot(m_output - (m_output[0] - nk_output[0]), label='Mutant')
plt.legend()

plt.show()


#########################

print "Difference %s" % np.mean(m_output-nk_output)

# Gain Differences
nk_mean_g = h5py.File('RealInput_mean_g.h5')['array'][:]
m_mean_g =  mutant['Blocks_MeanGain'][offset:offset+len(nk_input)]

nk_contrast_g = h5py.File('RealInput_contrast_g.h5')['array'][:]
m_contrast_g =  mutant['Blocks_ContrastGain'][offset:offset+len(nk_input)]

ax = plt.subplot(2, 1, 1)
plt.title('Total Mean Gain')
plt.plot(nk_mean_g, label='Neurokernel')
plt.plot(m_mean_g, label='Mutant')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('Total Contrast Gain')
plt.plot(nk_contrast_g, label='Neurokernel')
plt.plot(m_contrast_g, label='Mutant')
plt.legend()
plt.show()

# Computed Means
nk_mean = h5py.File('RealInput_mu_m.h5')['array'][:]
m_mean =  mutant['Computed_Mean'][offset:offset+len(nk_input),0]

nk_contrast = h5py.File('RealInput_mu_c.h5')['array'][:]
m_contrast =  mutant['Computed_Contrast'][offset:offset+len(nk_input),0]

ax = plt.subplot(2, 1, 1)
plt.title('Computed Mean')
plt.plot(nk_mean, label='Neurokernel')
plt.plot(m_mean, label='Mutant')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.title('Computed Contrast')
plt.plot(nk_contrast, label='Neurokernel')
plt.plot(m_contrast, label='Mutant')
plt.legend()
plt.show()

























