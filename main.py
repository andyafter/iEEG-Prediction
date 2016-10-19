import sys, os
from feature import calculate_features, mat_to_data
from settings import DATA_DIR

import numpy as np

file_names = os.listdir(DATA_DIR + '/train_3/')
f = open("./output/features3_1.txt", "w")
for name in file_names:
    label = name.split('.')[0].split('_')[2]
    f.write(label + ' ')
    feat = calculate_features(DATA_DIR + "/train_3/"+name)
    n = 85                           # only care about the first minute
    for i in range(n):
        f.write(str(i)+":"+str(feat[i])+" ")
    f.write("\n")
#f = mat_to_data(DATA_DIR + '/train_3/3_140_1.mat')
f.close()
#f1 = mat_to_data(DATA_DIR + '/train_3/3_141_1.mat')
#f2 = mat_to_data(DATA_DIR + '/train_3/3_1792_0.mat')
#f3 = mat_to_data(DATA_DIR + '/train_3/3_233_0.mat')
#f4 = mat_to_data(DATA_DIR + '/train_3/3_1496_0.mat')

