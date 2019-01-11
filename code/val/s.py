f=open('suppose_label_val.txt','r')
label_val=[]
for k in f.readlines():
    label_val.append(int(k))
f.close()
la=[0,0,0,0,0,0,0,0,0,0]
for index in range(500):
    labels=label_val[index*20:index*20+20]
    for k in labels:
        la[k]+=1
print la
