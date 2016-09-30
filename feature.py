import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
#import pyeeg
# pyeeg is the one that has very good fractal dimensions
# computation but not installed here

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x

def calculate_features(file_name):
    f = mat_to_data(file_name)
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    print((nt, nc))
    subsampLen = floor(fs * 60)
    numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,nt,numSamps)
    #print(sampIdx)
    feat = [] # Feature Vector
    for i in range(1, numSamps):
        print('processing file {} epoch {}'.format(file_name,i))
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]

        # compute Shannon's entropy, spectral edge and correlation matrix
        D = np.absolute(np.fft.fft(eegData))
        D[0,:]=0                                # set the DC component to zero
        D /= D.sum()                      # Normalize each channel
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt/fs*lvl).astype('int')+1
        # segments corresponding to frequency bands

        dspect = np.zeros((len(lvl)-1,nc))
        for j in range(len(lvl)-1):
            dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:])

        # Find the shannon's entropy
        spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)))

        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5

        topfreq = int(round(nt/sfreq*tfreq))+1
        A = np.cumsum(D[:topfreq,:])
        B = A - (A.max()*ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1)/(topfreq-1)*tfreq

        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=eegData)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)

        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)

        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(floor(nt/2))
        no_levels = int(floor(log(ldat,2)))
        seg = floor(ldat/pow(2, no_levels-1))

        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels,nc))
        for j in range(no_levels-1,-1,-1):
            dspect[j,:] = 2*np.sum(D[int(floor(ldat/2))+1:ldat,:])
            ldat = int(floor(ldat/2))

        # Find the Shannon's entropy
        spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)))

        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)

        # Fractal dimensions
        no_channels = nc
        #fd = np.zeros((2,no_channels))
        #for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(eegData[:,j])
        #    fd[1,j] = pyeeg.hfd(eegData[:,j],3)
        #    fd[2,j] = pyeeg.hurst(eegData[:,j])

        #[mobility[j], complexity[j]] = pyeeg.hjorth(eegData[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(eegData)

        # Mobility
        mobility = np.divide(np.std(np.diff(eegData)), np.std(eegData))

        # Complexity
        complexity = np.divide(np.divide(np.diff(np.diff(eegData)),
                                         np.std(np.diff(eegData))), mobility)
        # Statistical properties
        # Skewness
        sk = skew(eegData)

        # Kurtosis
        kurt = kurtosis(eegData)

        # compile all the features
        feat = np.concatenate((feat,
                               spentropy.ravel(),
                               spedge.ravel(),
                               lxchannels.ravel(),
                               lxfreqbands.ravel(),
                               spentropyDyd.ravel(),
                               lxchannels.ravel(),
                               #fd.ravel(),
                               activity.ravel(),
                               mobility.ravel(),
                               #complexity.ravel(),
                               sk.ravel(),
                               kurt.ravel()
                                ))

    return feat


file_names = os.listdir('./train_1/')
f = open("features1.txt", "w")
for name in file_names:
    label = name.split('.')[0].split('_')[2]
    f.write(label + ' ')
    feat = calculate_features("./train_1/"+name)
    n = 85                           # only care about the first minute
    for i in range(n):
        f.write(str(i)+":"+str(feat[i])+" ")
    f.write("\n")
f.close()
