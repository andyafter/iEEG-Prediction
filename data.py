from random import random

f = open("features1.txt",'r')
f2 = open("fout1.txt",'w')

inter, pos = 0, 0
while True:
    l = f.readline()
    if l =='':
        break
    if 'nan' in l:
        continue
    if l[0] == '1':
        inter += 1
        f2.write(l)
    elif l[0] == '0':
        if random()<0.122:
            f2.write(l)
        pos += 1

f.close()
f2.close()
