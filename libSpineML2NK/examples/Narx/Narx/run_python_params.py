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

mutant = io.loadmat('data/MutantBG6Data.mat')

m_input = mutant['recorded_input'][0:36000,0]
l = m_input.tolist()
l.insert(0,0)
m_input = np.array(l)


net = e.bundle.networks[0]
pop = net.Population[0]
pop.Neuron.Property 


pars = {
 'Contrast_K_1': 1.0000931884041385e-06,
 'Contrast_K_2': 0.017438565232583827,
 'Contrast_K_3': 0.00053921972167124226,

 'Contrast_X0_1': 0.78950438106528775,
 'Contrast_X0_2': 0.33112848144438961,
 'Contrast_X0_3': 7.7300101281352198,

 'Contrast_alpha_1': -2.2093228258778383,
 'Contrast_alpha_2': -1.3444700959816109,
 'Contrast_alpha_3': -1.794266702113672,
 'Contrast_beta_1': 6999.9905331388572,
 'Contrast_beta_2': 496.14805024208408,
 'Contrast_beta_3': 3100.0004971668282,
 "Contrast_beta_G']": 16.405345948449689,

 'Contrast_tau_1': 1.0401071826364103,

 'Contrast_A_1':0.0012003566315597568,
 'Contrast_B_1':  -0.9975992867368805,



 'Contrast_tau_2': 0.49999997136200214,
 'Contrast_A_2': 0.0024937657285116362,
 'Contrast_B_2': -0.9950124685429768,


 'Contrast_tau_3': 1.7319623248428329,
 'Contrast_A_3':  0.0007212041952871236,
 'Contrast_B_3': -0.9985575916094257,

 'Gralbeta_G': 1000,

 'Mean_K_1': 0.0073778950460808802,
 'Mean_K_2': 0.022266684710357632,
 'Mean_K_3': 0.0066009688542815071,
 'Mean_X0_1': 0.00014552091408427518,
 'Mean_X0_2': 0.08958786615222121,
 'Mean_X0_3': 0.1033351339097275,
 'Mean_alpha_1': -1.6952757942013319,
 'Mean_alpha_2': -1.2384563460468387,
 'Mean_alpha_3': -1.1660506692817525,
 'Mean_beta_1': 56.70165385724556,
 'Mean_beta_2': 416.19002756957138,
 'Mean_beta_3': 2000,

 'Mean_beta_G': 15.581626909473325,

 'Mean_tau_1': 8.2361520984230552,
 'Mean_A_1': 0.00015174687177639374,
 'Mean_B_1': -0.9996965062564472,

 'Mean_tau_2': 0.60169556672600977,
 'Mean_A_2': 0.002073155636233452,
 'Mean_B_2': -0.9958536887275331,


 'Mean_tau_3': 3.707965580158564,
 'Mean_A_3':  0.0003369984766284639,
 'Mean_B_3': -0.9993260030467431,

 'Am': 0.0012484394506866417,
 'Bm' : -0.9975031210986267
  
}


model     = io.loadmat("../data/MutantModel.mat")
theta     = np.ravel(model['mutantNARXparamsBG6'])

props = {}
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
   
    props[prop.name] = prop.AbstractValue.value
 
    print prop.name + " is " + str(prop.AbstractValue.value) 

pickle.dump( props, open( "properties.p", "wb" ) )

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

