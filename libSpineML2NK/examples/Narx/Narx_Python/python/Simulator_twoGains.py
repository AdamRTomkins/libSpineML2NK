from config import *
import scipy
import scipy.io
import scipy.signal
import numpy as np
import pickle
from numpy import *
import numpy
from matplotlib.pyplot import  * #plot


def printGainLoops(L1,L2,L3,Name = 'noname'):
    plt.figure(Name+ " Loops")

    subplot(321)
    plot(L1,'b')

    subplot(322)
    plot(L1,'b');yscale('log')
    
    subplot(323)
    plot(L2,'g');
    
    subplot(324)
    plot(L2,'g');yscale('log')
    
    subplot(325)
    plot(L3,'k')
    
    subplot(326)
    plot(L3,'k');yscale('log')
    
    plt.figure(Name+ " Loops Sum")
    
    
    
    subplot(221)
    plot(L1,'b');yscale('log')
    plot(L2,'g');yscale('log')
    plot(L3,'k');yscale('log')
    
    
    subplot(222)
    plot(L1,'b')
    plot(L2,'g')
    plot(L3,'k')
    
    subplot(223)
    plot(L1+L2+L3,'b');yscale('log')

    
    
    subplot(224)
    plot(L1+L2+L3,'b')

#
# Get AM and BM out of here
# ADAM
def MeanContrastComputation(inData):
    kT    = arange(size(inData))*(1.0/400.)
    tau   = 1.0 
    L     = scipy.signal.lti(1.0,[tau,1.0])
    (L_T,L_yout,L_xout) = scipy.signal.lsim(L,
                                            inData,
                                            kT,
                                            X0 = 0.0,
                                            interp=0)# X0 - initial condition , iterp 1 - ZOH
    Contrast = inData - L_yout   
    return(L_yout,Contrast)

def SimLoop(fs,inData,LoopParams):
    kT                  = arange(size(inData))*(1.0/fs)

    L                   = scipy.signal.lti(1.0,[LoopParams['tau'],1.0])
    
    


    (L_T,L_yout,L_xout) = scipy.signal.lsim(L,
                                            inData,
                                            kT,
                                            X0 = LoopParams['X0'],
                                            interp=0)# X0 - initial condition , iterp 1 - ZOH
    # calculate y                                            
    u_tilde = numpy.array(L_yout)

    # calulate f
    g        = LoopParams['K']*(u_tilde**LoopParams['alpha'])
    

    Art = 1000 #
    g   = g*Art
    
    g_prime  = LoopParams['beta']+(g-LoopParams['beta'])/(1+exp((g-LoopParams['beta'])))
    g_prime = g_prime/Art;
    return (g_prime, u_tilde,g/Art,g_prime)

def GainController(inData,GainParams,loop):
    R = np.zeros(len(inData))    
    loop_save = {}
    for L in GainParams:
        print "%(loop)s loop,  Parameter %(par)s" % {"loop":loop,"par":L}
        (L_temp,y,f,g) = SimLoop(400.0,inData,GainParams[L])
        R = R + L_temp
        loop_save[loop +"_y_" + L[-1]] = y
        loop_save[loop +"_f_" + L[-1]] = f
        loop_save[loop +"_g_" + L[-1]] = g
    
    loop_save[loop +"_g"] = R
    
    return (R,loop_save)

def simModel2Gains(theta,     #narx              
                   GainParams,     #gains              
                   inData,          #input
                   ScalingFactor):  # 1 for wild 1000 mutant
    
    #Scale input
    for i in xrange(len(inData)):
        if(inData[i]<=10.0**-10.0):
            inData[i]=10.0**-10.0
    
    #resolution = 10./ScalingFactor
    scaled_inData = ScalingFactor*inData
    
    #Compute Mean and Contrast
    (Mean,Contrast) = MeanContrastComputation(scaled_inData)
    
    #Feed the Mean and Variance to theis controllers    
    MeanParams = {}
    
    MeanParams['L1'] = {'X0'   : GainParams['Mean_X0_1'],
                                 'tau'  : GainParams['Mean_tau_1'],
                                 'K'    : GainParams['Mean_K_1'],
                                 'alpha': GainParams['Mean_alpha_1'],
                                 'beta' : GainParams['Mean_beta_1']}

    MeanParams['L2'] = {'X0'   : GainParams['Mean_X0_2'],
                                 'tau'  : GainParams['Mean_tau_2'],
                                 'K'    : GainParams['Mean_K_2'],
                                 'alpha': GainParams['Mean_alpha_2'],
                                 'beta' : GainParams['Mean_beta_2']}
                                 
    MeanParams['L3'] = {'X0'   : GainParams['Mean_X0_3'],
                                 'tau'  : GainParams['Mean_tau_3'],
                                 'K'    : GainParams['Mean_K_3'],
                                 'alpha': GainParams['Mean_alpha_3'],
                                 'beta' : GainParams['Mean_beta_3']}
                                 
    ContrastParams = {}
    
    ContrastParams['L1'] = {'X0'   : GainParams['Contrast_X0_1'],
                                     'tau'  : GainParams['Contrast_tau_1'],
                                     'K'    : GainParams['Contrast_K_1'],
                                     'alpha': GainParams['Contrast_alpha_1'],
                                     'beta' : GainParams['Contrast_beta_1']}

    ContrastParams['L2'] = {'X0'   : GainParams['Contrast_X0_2'],
                                     'tau'  : GainParams['Contrast_tau_2'],
                                     'K'    : GainParams['Contrast_K_2'],
                                     'alpha': GainParams['Contrast_alpha_2'],
                                     'beta' : GainParams['Contrast_beta_2']}
                                 
    ContrastParams['L3'] = {'X0'   : GainParams['Contrast_X0_3'],
                                     'tau'  : GainParams['Contrast_tau_3'],
                                     'K'    : GainParams['Contrast_K_3'],
                                     'alpha': GainParams['Contrast_alpha_3'],
                                     'beta' : GainParams['Contrast_beta_3']}
                      
    (Adapted_Mean,mean_dict)     = GainController(scaled_inData,MeanParams,"Mean")
    (Adapted_Contrast,cont_dict) = GainController(scaled_inData,ContrastParams,'Contrast')    

    loop_data = {}
    loop_data.update(mean_dict)
    loop_data.update(cont_dict)

    
#    pickle.dump({'Blocks_MeanGain':Adapted_Mean,'Blocks_ContrastGain':Adapted_Contrast},open("Wild_BlockGains.p","wb"))
#    pickle.dump({'Blocks_MeanGain':Adapted_Mean,'Blocks_ContrastGain':Adapted_Contrast},open("Mutant_BlockGains.p","wb"))
    
    #Compute the raw input
    Adapted_inData   = Mean*Adapted_Mean + Contrast*Adapted_Contrast
    
    #Applied Gral Saturation to raw input
    satParam   = GainParams['Gralbeta_G'];
    
    Art = 1000
    Adapted_inData   = Adapted_inData*Art
    
    NARX_input = satParam+(Adapted_inData-satParam)/(1+exp((Adapted_inData-satParam)))
    NARX_input = NARX_input/Art;  
    
    
    #Applied saturated raw input to NARX model
    
    NARX_output = simNarxPhotoreceptor(theta,NARX_input)
    
    return (NARX_output,loop_data)
    
def simNarxPhotoreceptor(theta,NARX_input):
    """
    The strucure, which is fix, is the following:
            y[k-1] = theta[1-1]*y[k-01-1]+
                 theta[2-1]*y[k-03-1]+
                 theta[3-1]*NARX_input[k-05-1]*NARX_input[k-04-1]+
                 theta[4-1]+
                 theta[5-1]*NARX_input[k-06-1]+
                 theta[6-1]*NARX_input[k-04-1]*y[k-06-1]+
                 theta[7-1]*NARX_input[k-07-1]+
                 theta[8-1]*NARX_input[k-07-1]*NARX_input[k-06-1]+
                 theta[9-1]*y[k-04-1]+
                 theta[10-1]*y[k-05-1]+
                 theta[11-1]*NARX_input[k-04-1]*y[k-05-1]+
                 theta[12-1]*NARX_input[k-04-1]*y[k-02-1]+
                 theta[13-1]*NARX_input[k-07-1]*NARX_input[k-03-1]+
                 theta[14-1]*NARX_input[k-05-1]+
                 theta[15-1]*NARX_input[k-04-1]    
    """
    N=size(NARX_input)              #number of data samples
    MaxLag=7                        #maximum number of lags in the model
    y=numpy.zeros(size(NARX_input))
    for k in range(MaxLag+1,N-1):
        y[k-1]=theta[1-1]*y[k-01-1]+theta[2-1]*y[k-03-1]+theta[3-1]*NARX_input[k-05-1]*NARX_input[k-04-1]+theta[4-1]+theta[5-1]*NARX_input[k-06-1]+theta[6-1]*NARX_input[k-04-1]*y[k-06-1]+theta[7-1]*NARX_input[k-07-1]+ theta[8-1]*NARX_input[k-07-1]*NARX_input[k-06-1]+ theta[9-1]*y[k-04-1]+ theta[10-1]*y[k-05-1]+ theta[11-1]*NARX_input[k-04-1]*y[k-05-1]+ theta[12-1]*NARX_input[k-04-1]*y[k-02-1]+ theta[13-1]*NARX_input[k-07-1]*NARX_input[k-03-1]+ theta[14-1]*NARX_input[k-05-1]    + theta[15-1]*NARX_input[k-04-1]    
    return y
    
def sortGainParams(Params,VarName):
    vals = np.ravel(Params[VarName])
      
    P_str = str(Params['orderparam'])
    P_STR = P_str[4:].split()
        
    SortedParams = {}
    for i in xrange(len(vals)):
        SortedParams[P_STR[i]] = vals[i]            
    return SortedParams


if __name__ == "__main__":    
    
    wild_or_mutant = 'mutant'
    ImageProcessing = 'n'
    
    if(wild_or_mutant=='wild'):
        Params    = scipy.io.loadmat("C:\\Users\\uos\\Google Drive\\PHD_2014\\PR_Jan\\intpoint_wild8.mat")    
        SortedPar = sortGainParams(Params,'x_wild8')        
        model     = scipy.io.loadmat("C:\\Users\\uos\\Google Drive\\PHD_2014\\PR_Jan\\WildModel.mat")
        inoutData = scipy.io.loadmat("C:\\Users\\uos\\Google Drive\\PHD_2014\\PR_Jan\\WildInputOutput.mat")
        theta = np.ravel(model['wildNARXparamsBG6'])
        structure = model['wildNARXstructure']
        U_input = inoutData['Wild_Input']
        pickle.dump({'Blocks_Params':SortedPar},open("Wild_BlockParams.p","wb"))
    else:
        Params    = scipy.io.loadmat("intpoint_mutant2.mat")
        SortedPar = sortGainParams(Params,'x_mutant2')
        SortedPar['Mean_beta_3'] = 2000
        SortedPar['Gralbeta_G']  =  1000
        model     = scipy.io.loadmat("MutantModel.mat")
        inoutData = scipy.io.loadmat("MutantInputOutput.mat" )    
        U_input = inoutData['Mutant_Input']
        theta     = np.ravel(model['mutantNARXparamsBG6'])
        pickle.dump({'Blocks_Params':SortedPar},open("Mutant_BlockParams.p","wb"))


        
        
    (SimOut,loop_data) = simModel2Gains(theta,SortedPar,np.ravel(U_input),ScalingFactor = 1.0)
    plot(SimOut-60)
    #    xlim(4000,76000)
    ylim(-100,100)

    show()    
#    if(wild_or_mutant=='wild'):
#        pickle.dump(SimOut-60,open("Wild_Blocks_MPO.p","wb"))
#    else:
#        pickle.dump(SimOut-60,open("Mutant_Blocks_MPO.p","wb"))
    

    #Image processing part
    if(ImageProcessing == 'y'):
        
        #Load and show the original picture
        
        imageDir  = "C:\\Users\uos\\Dropbox\\PHD\\MyThesis\\PythonProgramms\\ImageProcessing\\"
        im = Image.open(imageDir +"SLC.jpg")
        print im.format, im.size, im.mode
        im.show()
        
        #Decompose the RGB coponenets 
        r, g, b    = im.split()
        
        #Merge the info into one single channel and show the output picture
        im2 = Image.merge("RGB", (r, r, r))
        im2.show()

        #Prepare the picture to be rasterised
        R = np.array(r)        
        
        
        if(wild_or_mutant == 'wild'):
            myScalingFactor = 0.05/255.
        elif(wild_or_mutant == 'mutant'):            
            myScalingFactor = 0.05/255.
            #myScalingFactor = 0.05/255.
        
        
        
        yImageSimulator = simModel2Gains(theta,SortedPar,np.ravel(R),ScalingFactor = myScalingFactor)
        
        Temp0                 = yImageSimulator+np.abs(np.min(yImageSimulator))
        yImageSimulatorScaled = Temp0*(255.0/((np.max(yImageSimulator))-np.min(yImageSimulator)))
                
        
        ImSimulator  = Image.fromarray(np.reshape(uint8(yImageSimulatorScaled), np.shape(R)))
        ImSimulator.show()
        
        #input scaling        
        if(wild_or_mutant == 'wild'):
            b = -0.058;
            #R_Scaled        = (0.158/255.0)*R+b
            R_Scaled        = (0.05/255.0)*R
        elif(wild_or_mutant == 'mutant'):
            #b = -0.132;
            R_Scaled        = (0.05/255.0)*R
        
        yImageNARX      = simNarxPhotoreceptor(theta,np.ravel(R_Scaled))
        
        #Output Scaling        
        Temp1 =yImageNARX+np.abs(np.min(yImageNARX))
        Temp1 =Temp1*(255.0/((np.max(yImageNARX))-np.min(yImageNARX)))


#        yImageNARX      = polyNARMAX.simModelFL(structure,theta,np.ravel(R_Scaled))        
        ImNARX       = Image.fromarray(np.reshape(uint8(Temp1)     , np.shape(R)))    
        ImNARX.show()
    
