import numpy as np
from matplotlib import pyplot as plt
import scipy 
from scipy import io

def _1st_digital_linear_filter(x,
                               x_1,
                               y_1,
                               fs,
                               Tau,
                               Kss,
                               **kwargs):
    """
    x_1: 1-step lagged input
    y_1: 1-step lagged output
    """
    alpha = 1.+2.*fs*Tau;
    beta = 1.-2.*fs*Tau;

    A = Kss/alpha;
    B = beta/alpha;

    y = A * (x + x_1) - B * y_1;

    return y

def filter_mean_contrast(x,
                         x_0 = 0.,
                         mean_0 = 0.,
                         ):
    """
    """
    u = np.ravel(x)
    N = len(u)
    u_mean = np.zeros(N)
    u_mean[0] = mean_0

    
    for i in xrange(1,N-1):
        u_mean[i+1] = _1st_digital_linear_filter(u[i],
                                                 u[i-1],
                                                 u_mean[i-1],
                                                 400.,
                                                 1.,
                                                 1.
                                                 )

    u_contrast = u - u_mean
    return u_mean, u_contrast


if __name__ == '__main__':
    import sys
    wdata = '/Users/carlos/Dev/phd/src/Data/WildBG6Data.mat'
    wild = scipy.io.loadmat(wdata)
    x = wild['recorded_input']
    x_mean, x_contrast = filter_mean_contrast(x[4500:12000])

