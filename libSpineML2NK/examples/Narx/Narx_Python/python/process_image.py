import argparse
import sys
import itertools
import networkx as nx
import pdb
import argparse
import itertools
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import io
import cv2
import pickle 


file = 'leaf.jpg'
im = cv2.imread(file,cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32)

# scale the input signals between 0 and 0.2 or as passed in args.scale
im  = ((1.0*im) / 256) * args.scale

params['x'] = im.shape[0] 
    params['y'] = 1
    params['frames'] = im.shape[1]
    params['total'] = im.shape[0]
    pickle.dump( params, open( args.output+".p", "wb" ) )
    
    I = im.astype(np.double)
    with h5py.File(args.output+'.h5', 'w') as f:
        f.create_dataset('array', (params['frames'], params['total']),
                     dtype=np.double,
                     data=I)

    gen_network(params['total'],args.output,im[1,:])
