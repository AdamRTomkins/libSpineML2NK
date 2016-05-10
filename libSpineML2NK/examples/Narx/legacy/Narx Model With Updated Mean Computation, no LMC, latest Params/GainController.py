# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 08:45:18 2014

@author: uos
"""
import numpy
import scipy
from scipy import io
import sys
class GainController(object):
    #Windows
    #Params    = scipy.io.loadmat("C:\\Users\\uos\\Dropbox\\PHD\\Year1\\Programs\\Model\\FullModel")
    #model     = scipy.io.loadmat("C:\\Users\\uos\\Google Drive\\PHD_2014\\PR_Jan\\WildModel.mat")
    #inoutData = scipy.io.loadmat("C:\\Users\\uos\\Google Drive\\PHD_2014\\PR_Jan\\WildInputOutput.mat")    
    
    if sys.platform == 'linux2':
        dirData   = "/home/carlos/PythonProgramms/"     
    # name for the mac 
    elif sys.platform == 'darwin':
        dirData = "/Users/carlos/Dropbox/PHD/MyThesis/PythonProgramms/"
    else:
        dirData   = "C:\\Users\\uos\\Dropbox\\PHD\\MyThesis\\PythonProgramms\\"
        
    Params    = scipy.io.loadmat( dirData + "FullModel.mat"       )
    model     = scipy.io.loadmat( dirData + "WildModel.mat"       )
    inoutData = scipy.io.loadmat( dirData + "WildInputOutput.mat" )

    #Linux
    """
    Params    = scipy.io.loadmat("/home/carlos/PythonProgramms/FullModel.mat")
    model     = scipy.io.loadmat("/home/carlos/PythonProgramms/WildModel.mat")
    inoutData = scipy.io.loadmat("/home/carlos/PythonProgramms/WildInputOutput.mat")
    """
    def sortOneGainParams(GainParams):
        SortedParams = {}
    
        SortedParams = {'tau_1'      : GainParams['tau_1'][0][0],
                        'K_1'        : GainParams['K_1'][0][0],
                        'alpha_1'    : GainParams['alpha_1'][0][0],
                        'beta_1'     : GainParams['beta_1'][0][0],
                        'tau_2'      : GainParams['tau_2'][0][0],
                        'K_2'        : GainParams['K_2'][0][0],
                        'alpha_2'    : GainParams['alpha_2'][0][0],
                        'beta_2'     : GainParams['beta_2'][0][0],
                        'tau_3'      : GainParams['tau_3'][0][0],
                        'K_3'        : GainParams['K_3'][0][0],
                        'alpha_3'    : GainParams['alpha_3'][0][0],
                        'beta_3'     : GainParams['beta_3'][0][0],
                        'Gralbeta_G' : GainParams['gam'][0][0],}     
    
        return SortedParams

    SortedPar = sortOneGainParams(Params)   
    theta     = numpy.ravel(Params['theta'])
    structure = model['wildNARXstructure']
    U_input   = numpy.ravel(inoutData['Wild_Input']) 
    
    
    @classmethod
    def SimLoop(fs,inData,LoopParams):
        kT                  = numpy.arange(numpy.size(inData))*(1.0/fs)
        L                   = scipy.signal.lti(1.0,[LoopParams['tau'],1.0])
        (L_T,L_yout,L_xout) = scipy.signal.lsim(L,
                                            inData,
                                            kT,
                                            X0 = LoopParams['X0'],
                                            interp=0)# X0 - initial condition , iterp 1 - ZOH
        
                                            
        u_tilde = numpy.array(L_yout)
        g        = LoopParams['K']*(u_tilde**LoopParams['alpha'])
       
        Art = 1000
        g   = g*Art
       
        g_prime  = LoopParams['beta']+(g-LoopParams['beta'])/(1+numpy.exp((g-LoopParams['beta'])))
        g_prime = g_prime/Art;
       
        return g_prime
    
    @classmethod
    def DigitalSimLoop(cls,fs,inData,LoopParams):
        """
        The artifact - ART - is not necessary
        for the wild type, however it was kept for
        simplicity and convinience.
        """
        U      = inData
        N      = numpy.size(U)	
	
        ALPHA  = 1.0 + 2.0*fs*LoopParams['tau']
        BETA   = 1.0 - 2.0*fs*LoopParams['tau']
    
        A      = 1.0/ALPHA
        B      = BETA/ALPHA

        N      = numpy.size(U)      		#number of data samples
        MaxLag = 1                        		#maximum number of lags in the model
        H      = numpy.zeros(numpy.size(U))
    
        for k in xrange(MaxLag+1,N-1):
            H[k-1] = A*( U[k-01-0] + U[k-01-1] ) - B*H[k-01-1]


        u_tilde = numpy.array(H)
        g       = LoopParams['K']*(u_tilde**LoopParams['alpha'])
    
        #check for overflow
        for k in xrange(numpy.size(g)):
            if numpy.isinf(g[k]) == True:
                g[k] = sys.float_info.max
            
    
    
        g_prime  = LoopParams['beta']+(g-LoopParams['beta'])/(1+numpy.exp((g-LoopParams['beta'])))    
    
        return g_prime
        
    @classmethod
    def computeGainController(cls,inData,GainParams):
        R = numpy.zeros(len(inData))    
        for L in GainParams:
            L_temp = cls.DigitalSimLoop(400.0,inData,GainParams[L])
            R = R + L_temp
        return R
    
    @classmethod
    def simModel1Gain(cls,
                      theta         =theta,
		                  GainParams    =SortedPar,
		                  inData        =U_input,
		                  ScalingFactor =1.0,
                      gain          =None):

        #Scale input
        for i in xrange(len(inData)):
            if(inData[i]<=10.0**-10.0):
                inData[i]=10.0**-10.0
    
        #resolution = 10./ScalingFactor
        scaled_inData = ScalingFactor*inData    
    
        #Feed the Mean and Variance to theis controllers    
        WildParams = {}
    
        WildParams['L1'] = {'tau'  : GainParams['tau_1'],
                            'K'    : GainParams['K_1'],
                            'alpha': GainParams['alpha_1'],
                            'beta' : GainParams['beta_1']}

        WildParams['L2'] = {'tau'  : GainParams['tau_2'],
                        'K'    : GainParams['K_2'],
                        'alpha': GainParams['alpha_2'],
                        'beta' : GainParams['beta_2']}
                                 
        WildParams['L3'] = {'tau'  : GainParams['tau_3'],
                            'K'    : GainParams['K_3'],
                            'alpha': GainParams['alpha_3'],
                            'beta' : GainParams['beta_3']}    
    
	
        Adapted_U = cls.computeGainController(scaled_inData,WildParams)
    
        U_tilde   = Adapted_U*scaled_inData
	
        #Applied Gral Saturation to raw input
        satParam   = GainParams['Gralbeta_G'];
    
    
        NARX_input = satParam+(U_tilde-satParam)/(1+numpy.exp((U_tilde-satParam)))
    
    
        #Applied saturated raw input to NARX model    
        NARX_output = cls.simNarxPhotoreceptor(theta,NARX_input)
        
        if gain: 
          res = Adapted_U,NARX_output
        else:
          res = NARX_output
        return res 
    
    @classmethod
    def simGainController(cls,theta,
		  GainParams,
		  inData,
		  ScalingFactor):

        #Scale input
        for i in xrange(len(inData)):
            if(inData[i]<=10.0**-10.0):
                inData[i]=10.0**-10.0
    
        #resolution = 10./ScalingFactor
        scaled_inData = ScalingFactor*inData    
    
        #Feed the Mean and Variance to theis controllers    
        WildParams = {}
    
        WildParams['L1'] = {'tau'  : GainParams['tau_1'],
                            'K'    : GainParams['K_1'],
                            'alpha': GainParams['alpha_1'],
                            'beta' : GainParams['beta_1']}

        WildParams['L2'] = {'tau'  : GainParams['tau_2'],
                        'K'    : GainParams['K_2'],
                        'alpha': GainParams['alpha_2'],
                        'beta' : GainParams['beta_2']}
                                 
        WildParams['L3'] = {'tau'  : GainParams['tau_3'],
                            'K'    : GainParams['K_3'],
                            'alpha': GainParams['alpha_3'],
                            'beta' : GainParams['beta_3']}    
    
	
        Adapted_U = cls.computeGainController(scaled_inData,WildParams)
    
        U_tilde   = Adapted_U*scaled_inData
	
        #Applied Gral Saturation to raw input
        satParam   = GainParams['Gralbeta_G'];
    
    
        NARX_input = satParam+(U_tilde-satParam)/(1+numpy.exp((U_tilde-satParam)))
        
    
        return NARX_input
            
    
    @classmethod
    def simNarxPhotoreceptor(cls,theta,NARX_input):
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
        N=numpy.size(NARX_input)              #number of data samples
        MaxLag=7                        #maximum number of lags in the model
        y=numpy.zeros(numpy.size(NARX_input))
        for k in range(MaxLag+1,N-1):
            y[k-1]=theta[1-1]*y[k-01-1]+theta[2-1]*y[k-03-1]+theta[3-1]*NARX_input[k-05-1]*NARX_input[k-04-1]+theta[4-1]+theta[5-1]*NARX_input[k-06-1]+theta[6-1]*NARX_input[k-04-1]*y[k-06-1]+theta[7-1]*NARX_input[k-07-1]+ theta[8-1]*NARX_input[k-07-1]*NARX_input[k-06-1]+ theta[9-1]*y[k-04-1]+ theta[10-1]*y[k-05-1]+ theta[11-1]*NARX_input[k-04-1]*y[k-05-1]+ theta[12-1]*NARX_input[k-04-1]*y[k-02-1]+ theta[13-1]*NARX_input[k-07-1]*NARX_input[k-03-1]+ theta[14-1]*NARX_input[k-05-1]    + theta[15-1]*NARX_input[k-04-1]    
        return y


    @classmethod
    def SatFunc(x,rho):
        return rho+(x-rho)/(1+numpy.exp((x-rho)))
    
    @classmethod
    def StaticNonlinearTransf(cls,x,K,alpha,beta):
        g = K*(x**alpha)
    
        #check for overflow
        if numpy.isinf(g) == True:
            g = sys.float_info.max    
    
        return cls.SatFunc(g,beta)    
    
    @classmethod
    def simNarxPhotoreceptor_ALL(cls,U,theta,LoopsPar):    
    
        fs = 400    
    
        #Unpacking the parameters of the model

        #Linear systems
        tau_0 = LoopsPar['tau_1']
        tau_1 = LoopsPar['tau_2']
        tau_2 = LoopsPar['tau_3']
        
        #Computing linear system parameters for digital implementation
        A = numpy.zeros(3)
        B = numpy.zeros(3)    
        
        #1/alpha    
        A[0] = 1.0 / ( 1.0 + 2.0*fs*tau_0 )
        A[1] = 1.0 / ( 1.0 + 2.0*fs*tau_1 )
        A[2] = 1.0 / ( 1.0 + 2.0*fs*tau_2 )
        
        #beta/alpha
        B[0] = ( 1.0 - 2.0*fs*tau_0 ) / ( 1.0 + 2.0*fs*tau_0 )
        B[1] = ( 1.0 - 2.0*fs*tau_1 ) / ( 1.0 + 2.0*fs*tau_1 )
        B[2] = ( 1.0 - 2.0*fs*tau_2 ) / ( 1.0 + 2.0*fs*tau_2 )
        
        #Nonlinearities
        k_0     = LoopsPar['K_1']
        k_1     = LoopsPar['K_2']
        k_2     = LoopsPar['K_3']
        
        alpha_0 = LoopsPar['alpha_1']
        alpha_1 = LoopsPar['alpha_2']
        alpha_2 = LoopsPar['alpha_3']
        
        beta_0  = LoopsPar['beta_1']
        beta_1  = LoopsPar['beta_2']
        beta_2  = LoopsPar['beta_3']
        
        #General Saturation 
        Gralbeta_G = LoopsPar['Gralbeta_G']
    
        #number of data samples    
        N=numpy.size(U)              
    
    
        #For batch processing    
        
        h0 = numpy.zeros(numpy.size(U))
        h1 = numpy.zeros(numpy.size(U))
        h2 = numpy.zeros(numpy.size(U))
    
        H0 = numpy.zeros(numpy.size(U))
        H1 = numpy.zeros(numpy.size(U))
        H2 = numpy.zeros(numpy.size(U))    
        H  = numpy.zeros(numpy.size(U))    
    
        U_tilde    = numpy.zeros(numpy.size(U))   
        NARX_input = numpy.zeros(numpy.size(U))
        y          = numpy.zeros(numpy.size(U))
       
        #for k in range(MaxLag+1,N-1):
        for k in xrange(N-1):
           
           #Data scalation
           if(U[k]<=10.0**-10.0):
               U[k]=10.0**-10.0
        
           if   k==0:
               #Linear systems
               h0[k] = A[0]*U[k]
               h1[k] = A[1]*U[k]
               h2[k] = A[2]*U[k]
               
               #Static nonlinearities
               H0[k] = cls.StaticNonlinearTransf( h0[k], k_0, alpha_0, beta_0 )
               H1[k] = cls.StaticNonlinearTransf( h1[k], k_1, alpha_1, beta_1 )
               H2[k] = cls.StaticNonlinearTransf( h2[k], k_2, alpha_2, beta_2 )
            
               #Adapted U            
               H[k]          = H0[k]+H1[k]+H2[k]            
               U_tilde[k]    = U[k]*H[k]            
               NARX_input[k] = cls.SatFunc(U_tilde[k],Gralbeta_G)
            
               #NARX
               y[k]=theta[4-1]
               y_max = y[k]
               y_min = y[k]
                
           elif k >= 1:
               
               #Linear systems
               h0[k] = A[0]*(U[k]+U[k-1]) - B[0]*h0[k-1]
               h1[k] = A[1]*(U[k]+U[k-1]) - B[1]*h1[k-1]
               h2[k] = A[2]*(U[k]+U[k-1]) - B[2]*h2[k-1]
            
               #Static nonlinearities
               H0[k] = cls.StaticNonlinearTransf( h0[k], k_0, alpha_0, beta_0 )
               H1[k] = cls.StaticNonlinearTransf( h1[k], k_1, alpha_1, beta_1 )
               H2[k] = cls.StaticNonlinearTransf( h2[k], k_2, alpha_2, beta_2 )
            
               #Adapted U            
               H[k]          = H0[k]+H1[k]+H2[k]
               U_tilde[k]    = U[k]*H[k]
               NARX_input[k] = cls.SatFunc(U_tilde[k],Gralbeta_G)
            
            
               if k==1:                
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]            
        
               elif k==2:                
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]
        
               elif k==3:
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]+theta[2-1]*y[k-03]
        
               elif k==4:
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]+theta[2-1]*y[k-03]+theta[9-1]*y[k-04]+theta[12-1]*NARX_input[k-04]*y[k-02]+theta[15-1]*NARX_input[k-04] 
        
               elif k==5:
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]+theta[2-1]*y[k-03]+theta[9-1]*y[k-04]+theta[12-1]*NARX_input[k-04]*y[k-02]+theta[15-1]*NARX_input[k-04]+theta[3-1]*NARX_input[k-05]*NARX_input[k-04]+theta[10-1]*y[k-05]+theta[11-1]*NARX_input[k-04]*y[k-05]+theta[14-1]*NARX_input[k-05]
        
               elif k==6:
                   #NARX
                   y[k]= theta[4-1] + theta[1-1]*y[k-01]+theta[2-1]*y[k-03]+theta[9-1]*y[k-04]+theta[12-1]*NARX_input[k-04]*y[k-02]+theta[15-1]*NARX_input[k-04]+theta[3-1]*NARX_input[k-05]*NARX_input[k-04]+theta[10-1]*y[k-05]+theta[11-1]*NARX_input[k-04]*y[k-05]+theta[14-1]*NARX_input[k-05]+theta[5-1]*NARX_input[k-06]+theta[6-1]*NARX_input[k-04]*y[k-06]
        
               elif k>=7:                
                   y[k]=theta[1-1]*y[k-01]+theta[2-1]*y[k-03]+theta[3-1]*NARX_input[k-05]*NARX_input[k-04]+theta[4-1]+theta[5-1]*NARX_input[k-06]+theta[6-1]*NARX_input[k-04]*y[k-06]+theta[7-1]*NARX_input[k-07]+ theta[8-1]*NARX_input[k-07]*NARX_input[k-06]+ theta[9-1]*y[k-04]+ theta[10-1]*y[k-05]+ theta[11-1]*NARX_input[k-04]*y[k-05]+ theta[12-1]*NARX_input[k-04]*y[k-02]+ theta[13-1]*NARX_input[k-07]*NARX_input[k-03]+ theta[14-1]*NARX_input[k-05] + theta[15-1]*NARX_input[k-04]
            
               #Get max val
               if y[k] > y_max :
                   y_max = y[k]
            
               #Get max val
               if y[k] < y_min :
                   y_min = y[k]

        return (y,y_min,y_max)
    
    
    

    
