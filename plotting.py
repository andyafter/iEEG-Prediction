from feature import mat_to_data
from settings import DATA_DIR
import matplotlib.pyplot as plt
from numpy import floor


file_name = DATA_DIR + '/train_3/3_111_1.mat'
f = mat_to_data(file_name)
fs = f['iEEGsamplingRate'][0,0]
eegData = f['data']
[nt, nc] = eegData.shape
print((nt, nc))
subsampLen = floor(fs * 60)
numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples
sampIdx = range(0,nt,numSamps)
