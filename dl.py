from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy as np
import os, sys

from settings import DATA_DIR
from feature import mat_to_data


np.random.seed(1337)
file_names = os.listdir(DATA_DIR + '/train_3/')

## extracting data
f, fs, eegData = {}, {}, {}
input_data = []
print len(file_names)
for file_name in file_names:
    f = mat_to_data(DATA_DIR+"/train_3/"+file_name)
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    subsampLen = np.floor(fs * 60)
    numSamps = int(np.floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,nt,numSamps)
    print file_name
    input_data.append((eegData, file_name.split('.')[0]))
    break

## definitions
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 1)


rng = np.random.RandomState(23455)

# model
model = Sequential()
model.add(Convolution1D(nb_filters, 3, border_mode='valid', input_shape=(100,)))
model.add(Activation('relu'))

