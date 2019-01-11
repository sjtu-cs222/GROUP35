import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import numpy as np
from resnet import ResNet18
import random
import time

device = torch.device('cuda:1')


f=open('suppose_label_val.txt','r')
label_val=[]
for k in f.readlines():
    label_val.append(int(k))
label_val=np.array(label_val)
f.close()
shuff=[]
dataa=[]
for i in range(len(label_val)):
    if label_val[i]==0:
        shuff.append(i)
    else:
        dataa.append(i)
random.shuffle(shuff)
for i in range(16712):
    dataa.append(shuff[i])
random.shuffle(dataa)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') 
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)") 
args = parser.parse_args()


EPOCH = 135   
pre_epoch = 0  
BATCH_SIZE = 128   
LR = 0.1        

ppp=open("train_index.txt",'w')
for k in dataa:
    ppp.write(str(k))
    ppp.write('\n')
ppp.close()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=False, num_workers=2)   
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = ResNet18().to(device)

train_inputs=[]
train_labels=[]
all_inputs=[]
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    for k in range(len(inputs)):
        all_inputs.append(inputs[k])

for i in range(1500):
    batch=torch.FloatTensor(20, 3,32,32).zero_()
    batch_label=[]
    for j in range(20):
        batch[j]=(all_inputs[dataa[i*20+j]])
        batch_label.append(label_val[dataa[i*20+j]])
    
    batch_label=torch.from_numpy(np.array(batch_label))
    train_inputs.append(batch)
    train_labels.append(batch_label)
train=[]
cal=[0,0,0,0,0,0,0,0,0,0]
for i in range(1500):
    a=[]
    a.append(train_inputs[i])
    a.append(train_labels[i])
    for k in train_labels[i]:
        cal[k.item()]+=1
    train.append(a)
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

net.load_state_dict(torch.load('./11/res_model/net_030.pth'))

if __name__ == "__main__":
    best_acc = 85  
    print("Start Training, Resnet-18!")  
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                index=0
                random.shuffle(train)
                for i in range(len(train)):
                    
                    length = len(trainloader)
                    inputs=train[i][0]
                    labels=train[i][1]
                
                    index+=1
                    #labels=torch.from_numpy(labels)
                    
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    #print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                     #     % (epoch + 1, (i + 1 + epoch * 400), sum_loss / (i + 1), 100. * correct / total))
                    #f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                    #      % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    #f2.write('\n')
                    #f2.flush()
                print("epoch",sum_loss)
                torch.save(net.state_dict(), './11/res_model/net_%03d.pth' % (epoch + 1+31))
                """
                #predict in the test set
                if (1):
                    correct = 0
                    total = 0
                    index=0
                    ll=[0,0,0,0,0,0,0,0,0,0]
                    la=[0,0,0,0,0,0,0,0,0,0]
                    g=open('predict.txt','w')
                    starttime=time.time()
                    for data in testloader:
                        #print index
                        net.eval()
                        images, labels = data
                        labels=label_val[index*20:index*20+20]
                        index+=1
                        labels=torch.from_numpy(labels)
                        images=images.to(device)
                        labels =  labels.to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        for k in predicted:
                            #print(str(int(k)))
                            g.write(str(int(k)))
                            g.write('\n')

                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                        for k in predicted:
                            #print (k)
                            ll[k]+=1
                        for k in labels:
                            la[k]+=1
                    print(correct) 
                    #endtime=time.time()
                    #print (endtime-starttime)
                    #g.close()
                    print (ll)
                    print (la)
                        #print ("acc",100.*correct/total)
                    print("Training Finished, TotalEPOCH=%d" % EPOCH)
                    print(index)"""
                    