import sys, os
from feature import calculate_features
from settings import DATA_DIR

file_names = os.listdir(DATA_DIR + '/train_3/')
#f = open("features1.txt", "w")
print file_names[145]
for name in file_names:
    break
    label = name.split('.')[0].split('_')[2]
    f.write(label + ' ')
    feat = calculate_features(DATA_DIR + "/train_3/"+name)
    n = 85                           # only care about the first minute
    for i in range(n):
        f.write(str(i)+":"+str(feat[i])+" ")
    f.write("\n")
#f.close()

