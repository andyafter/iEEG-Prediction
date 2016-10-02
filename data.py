f = open("features1.txt",'r')
f2 = open("fout1.txt",'w')

while True:
    l = f.readline()
    if l =='':
        break
    if 'nan' in l:
        continue
    f2.write(l)

f.close()
f2.close()
