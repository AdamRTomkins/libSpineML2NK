# run and validate model

from Simulator_twoGains import *
from matplotlib import pyplot as plt
import h5py
import numpy as np
from scipy import io


Params    = scipy.io.loadmat("intpoint_mutant2.mat")
SortedPar = sortGainParams(Params,'x_mutant2')
SortedPar['Mean_beta_3'] = 2000
SortedPar['Gralbeta_G']  =  1000
model     = scipy.io.loadmat("MutantModel.mat")
inoutData = scipy.io.loadmat("MutantInputOutput.mat" )    
U_input = inoutData['Mutant_Input']
theta     = np.ravel(model['mutantNARXparamsBG6'])

model_range = np.arange(0,36000)

(SimOut,data) = simModel2Gains(theta,SortedPar,np.ravel(U_input),ScalingFactor = 1.0)

mutant = io.loadmat('../data/MutantBG6Data.mat')


# Output

O = h5py.File('../RealInput_y.h5')['array'][4200:36000]
O = O - O[-1]
ax = plt.subplot(1, 1, 1)
plt.title('Input')
plt.plot(O , label='Neurokernel' )
mpo = mutant['MPO_Blocks'][4200:36000]
mpo = mpo -mpo[-1]

plt.plot(mpo , label='MPO')
plt.legend()
plt.show()


# Input
I = h5py.File('../RealInput_I.h5')['array'][:]
ax = plt.subplot(1, 1, 1)
plt.title('Input')
plt.plot(I , label='Neurokernel Input' )
plt.plot(mutant['recorded_input'] , label='Recorded Input')
plt.legend()
plt.show()

# Mean and Contrast
mu = h5py.File('../RealInput_mu.h5')['array'][:]
nu = h5py.File('../RealInput_nu.h5')['array'][:]


ax = plt.subplot(2, 1, 1)
plt.title('Input Mean')
plt.plot(mu , label='mu' )
plt.plot(mutant['Computed_Mean'] , label='mu python')
plt.legend()
ax = plt.subplot(2, 1, 2)
plt.plot(nu , label='nu' )
plt.plot(mutant['Computed_Contrast'] , label='nu python')
plt.legend()

plt.show()






my1 = h5py.File('../RealInput_Mean_y_1.h5')['array'][:]
my2 = h5py.File('../RealInput_Mean_y_2.h5')['array'][:]
my3 = h5py.File('../RealInput_Mean_y_3.h5')['array'][:]

mf1 = h5py.File('../RealInput_Mean_f_1.h5')['array'][:]
mf2 = h5py.File('../RealInput_Mean_f_2.h5')['array'][:]
mf3 = h5py.File('../RealInput_Mean_f_3.h5')['array'][:]

mg1 = h5py.File('../RealInput_Mean_g_1.h5')['array'][:]
mg2 = h5py.File('../RealInput_Mean_g_2.h5')['array'][:]
mg3 = h5py.File('../RealInput_Mean_g_3.h5')['array'][:]


mg = h5py.File('../RealInput_Mean_g.h5')['array'][:]


ax = plt.subplot(2, 2, 1)
ax.set_yscale('log')
plt.title('Mean Ys')
plt.plot(my1 , label='Mean Y_1' )
plt.plot(data['Mean_y_1'][model_range],'-',label='Mean Y_1 Python')
plt.plot(my2 , label='Mean Y_2')
plt.plot(data['Mean_y_2'][model_range],'-',label='Mean Y_2 Python')
plt.plot(my3 , label='Mean Y_3')
plt.plot(data['Mean_y_3'][model_range],'-',label='Mean Y_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 2)
ax.set_yscale('log')
plt.title('Mean Fs')
plt.plot(mf1 , label='Mean F_1')
plt.plot(data['Mean_f_1'][model_range],'-',label='Mean F_1 Python')
plt.plot(mf2 , label='Mean F_2')
plt.plot(data['Mean_f_2'][model_range],'-',label='Mean F_2 Python')
plt.plot(mf3 , label='Mean F_3')
plt.plot(data['Mean_f_3'][model_range],'-',label='Mean F_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 3)
ax.set_yscale('log')
plt.title('Mean Gs')
plt.plot(mf1 , label='Mean G_1')
plt.plot(data['Mean_g_1'][model_range],'-',label='Mean G_1 Python')
plt.plot(mf2 , label='Mean G_2')
plt.plot(data['Mean_g_2'][model_range],'-',label='Mean G_2 Python')
plt.plot(mf3 , label='Mean G_3')
plt.plot(data['Mean_g_3'][model_range],'-',label='Mean G_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 4)
ax.set_yscale('log')
plt.title('Mean G Total')
plt.plot(mg , label='Mean G')
plt.plot(data['Mean_g'][model_range],'-',label='Mean G Python')
plt.legend()

plt.show()

cy1 = h5py.File('../RealInput_Contrast_y_1.h5')['array'][:]
cy2 = h5py.File('../RealInput_Contrast_y_2.h5')['array'][:]
cy3 = h5py.File('../RealInput_Contrast_y_3.h5')['array'][:]

cf1 = h5py.File('../RealInput_Contrast_f_1.h5')['array'][:]
cf2 = h5py.File('../RealInput_Contrast_f_2.h5')['array'][:]
cf3 = h5py.File('../RealInput_Contrast_f_3.h5')['array'][:]

cg1 = h5py.File('../RealInput_Contrast_g_1.h5')['array'][:]
cg2 = h5py.File('../RealInput_Contrast_g_2.h5')['array'][:]
cg3 = h5py.File('../RealInput_Contrast_g_3.h5')['array'][:]

cg = h5py.File('../RealInput_Contrast_g.h5')['array'][:]


ax = plt.subplot(2, 2, 1)
ax.set_yscale('log')
plt.title('Contrast Ys')
plt.plot(cy1 , label='Contrast Y_1')
plt.plot(data['Contrast_y_1'][model_range],'-',label='Contrast Y_1 Python')
plt.plot(cy2 , label='Contrast Y_2')
plt.plot(data['Contrast_y_2'][model_range],'-',label='Contrast Y_2 Python')
plt.plot(cy3 , label='Contrast Y_3')
plt.plot(data['Contrast_y_3'][model_range],'-',label='Contrast Y_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 2)
ax.set_yscale('log')
plt.title('Contrast Fs')
plt.plot(cf1 , label='Contrast F_1')
plt.plot(data['Contrast_f_1'][model_range],'-',label='Contrast F_1 Python')
plt.plot(cf2 , label='Contrast F_2')
plt.plot(data['Contrast_f_2'][model_range],'-',label='Contrast F_2 Python')
plt.plot(cf3 , label='Contrast F_3')
plt.plot(data['Contrast_f_3'][model_range],'-',label='Contrast F_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 3)
ax.set_yscale('log')
plt.title('Contrast Gs')
plt.plot(cf1 , label='Contrast G_1')
plt.plot(data['Contrast_g_1'][model_range],'-',label='Contrast G_1 Python')
plt.plot(cf2 , label='Contrast G_2')
plt.plot(data['Contrast_g_2'][model_range],'-',label='Contrast G_2 Python')
plt.plot(cf3 , label='Contrast G_3')
plt.plot(data['Contrast_g_3'][model_range],'-',label='Contrast G_3 Python')
plt.legend()

ax = plt.subplot(2, 2, 4)
ax.set_yscale('log')
plt.title('Contrast G Total')
plt.plot(cg , label='Contrast G')
plt.plot(data['Contrast_g'][model_range],'-',label='Contrast G Python')
plt.legend()

plt.show()

