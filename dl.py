import keras
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import os, sys

from settings import DATA_DIR
from feature import mat_to_data


file_names = os.listdir(DATA_DIR + '/train_3/')

## extracting data
f = mat_to_data(DATA_DIR+"/train_3/"+file_names[0])
fs = f['iEEGsamplingRate'][0,0]
eegData = f['data']
[nt, nc] = eegData.shape
print((nt, nc))
subsampLen = np.floor(fs * 60)
numSamps = int(np.floor(nt / subsampLen));      # Num of 1-min samples
sampIdx = range(0,nt,numSamps)
print eegData.shape


rng = np.random.RandomState(23455)
