def tau_AB(tau):
    fs = 400.0
    ALPHA  = 1.0 + 2.0*fs*tau
    BETA   = 1.0 - 2.0*fs*tau
    A      = 1.0/ALPHA
    B      = BETA/ALPHA  
    return (A,B)
