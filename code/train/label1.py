f=open('label2.txt','r')
g=open('max_label.txt','w')
h=open('suppose_label.txt','w')
q=open('true_label_train.txt','r')
true_label=[]
for k in q.readlines():
    true_label.append(int(k))
maxx_po=[]
index=0
for i in f.readlines():
    if i[0]=='i':
        content=i
        g.write(content)
    elif i[1]=='9':
        lines=i.split('\t')
        maxx=0
        label=-1
        maxx_po.append(float(lines[true_label[index]+1]))
        index+=1
        content=str(label)
        g.write(content)
        g.write('\n')
g.close()
f.close()
f=open('label2.txt','r')
lines=[]
p=f.readlines()
print len(p)
ass_label=[]
cal=[0,0,0,0,0,0,0,0,0,0]
for i in range(50000):
    flag=0
    index=i*11+1
    for j in range(10):
        lines=p[j+index].strip().split('\t')[1:]
        if float(lines[true_label[i]])>0.5 and float(lines[true_label[i]])>maxx_po[i]*0.99:
            ass_label.append(j)
            flag=1
            cal[j]+=1
            break
    if flag==0:
        ass_label.append(9)
        cal[9]+=1
for i in ass_label:
    h.write(str(i))
    h.write('\n')
h.close()
print cal
        
    

f.close()
