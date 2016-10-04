import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import pywt

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0,0] for n in names}
    return ndata

def DWT(file_name):
    f = mat_to_data(file_name)
    fs =f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape    
    #print (nt,nc)
    feat = [] #Feature Vector (electrode_16{band_5{feature_4}})
    for i in range(nc):        
        print('processing file {} channel{}'.format(file_name,i))
        #do we need filters before?
        #cD1:200-400Hz cD2:100-200Hz cD3:50-100Hz
        #cD4:25-50Hz cD5:12.5-25Hz cD6:6.25-12.5Hz cD7:3.125-6.25Hz cA7:0-3.125Hz
        #use cD4-cA7
        #MAV_eeg = np.mean(eegData[:,i])
        #RMS_eeg = np.sqrt(np.mean(np.square(eegData[:,i])))
        #STD_eeg = np.std(eegData[:,i])
        #print(MAV_eeg)
        #print(RMS_eeg)
        #print(STD_eeg)
        #raw_input()
        cA7,cD7,cD6,cD5,cD4,cD3,cD2,cD1 = pywt.wavedec(eegData[:,i],'db2',level=7) 
        bands = [cD4,cD5,cD6,cD7,cA7]
        #calculate entropy, energy, std_deviation of each band
        #add them into feat
        MAV = np.mean(cD3)
        for band in bands:
            #absolute mean values of adjacent band
            ad_MAV = MAV
            #mean of absolute values
            MAV = np.mean(band)
            #root mean square
            RMS = np.sqrt(np.mean(np.square(band)))
            #standard deviation
            STD = np.std(band)
            #Average Amplitute Change
            total = 0
            for j in range(band.size - 1):
                total = total + abs(band[j+1]-band[j])
            AAC = total / (band.size-1)
            #ratio of the absolute mean values of ajacent sub-bands
            RMA = ad_MAV/MAV
            feat.extend([MAV,RMS,STD,AAC,RMA])
    return feat

file_names = os.listdir('./../train_1/')
f = open("DWT_5*5*16_feature_for_svm","w")
for name in file_names:
    label = name.split('.')[0].split('_')[2]
    #print label
    if (label == "0"): 
        label = "-1"
    else:
        label = "+1"
    f.write(label + ' ')
    DWT_feat = DWT("./../train_1/"+name)
    n = len(DWT_feat)
    for i in range(n):
        f.write(str(i)+":"+str(DWT_feat[i])+" ")
        #print (str(i)+":"+str(DWT_feat[i])+" ")
        #raw_input()
    f.write("\n")
f.close()
