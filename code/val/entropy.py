f=open('entropy.txt','r')
g=open('suppose_label_val.txt','r')
ent=[]
label=[]
intt=[]
for k in f.readlines():
    ent.append(float(k))
index=0
for k in g.readlines():
    label.append(float(k))
    if float(k)>=6:
        intt.append(index)
    index+=1
print intt
base=ent[0]
for k in intt:
    print (ent[k]-base)
