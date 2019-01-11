import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main():


    # Check the save_dir exists or not

    model1 = torch.nn.DataParallel(resnet.__dict__['resnet20']())
    model1.cuda()

    # optionally resume from a checkpoint
    """if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))"""
    checkpoint = torch.load('./pretrained_models/resnet20.th')
    model1.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True
    
    

    # optionally resume from a checkpoint
    """if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))"""
    checkpoint = torch.load('./pretrained_models/resnet20.th')
    model1.load_state_dict(checkpoint['state_dict'])
    model2 = torch.nn.DataParallel(resnet.__dict__['resnet110']())
    model2.cuda()
    checkpoint = torch.load('./pretrained_models/resnet110.th')
    model2.load_state_dict(checkpoint['state_dict'])
    cudnn.benchmark = True
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='../../data', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss().cuda()

    validate(val_loader, model1,model2, criterion)

def validate(val_loader, model1,model2, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    f=open('suppose_label.txt','r')
    switch=[]
    for i in f.readlines():
        switch.append(int(i))
    f.close()
    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    starttime=time.time()
    index=0
    adv1=[]
    adv2=[]
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output2=model2(input_var)
        output1=model1(input_var)
        #output = model(input_var)
        """if switch[index]>2:
            output=model2(input_var)
        else:
            output=model1(input_var)"""
        """a=random.randint(1,10)
        if a>5:
            output=model2(input_var)
        else:
            output=model1(input_var)
        loss = criterion(output, target_var)
        index+=1"""

        output1 = output1.float()
        output2= output2.float()
        loss = criterion(output1, target_var)
        loss = loss.float()

        # measure accuracy and record loss
        """prec1 = accuracy(output1.data, target)[0]
        prec2 = accuracy(output2.data, target)[0]
        a=prec1[0].cpu().item()
        b=prec2[0].cpu().item()
        if float(a)>float(b):
            adv1.append(index)
        elif float(a)<float(b):
            adv2.append(index)
        index+=1"""
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
    #f1=open('adv1.txt','w')
    #f2=open('adv2.txt','w')
    endtime=time.time()
    #print(' * Prec@1 {top1.avg:.3f}'
    #      .format(top1=top1))
    print(endtime-starttime)
    """print("adv1")
    for k in adv1:
        f1.write(str(k))
        f1.write('\t')
        f1.write(str(switch[k]))
        f1.write('\n')
        print(k,switch[k])
    print("adv2")
    for k in adv2:
        print(k,switch[k])
        f2.write(str(k))
        f2.write('\t')
        f2.write(str(switch[k]))
        f2.write('\n')
    f1.close()
    f2.close()
    print(len(adv1))
    print(len(adv2))"""
    return 0

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
# 110:93.68 160.41
# mulit :92.1 47.377
# random:92.68 104.3
#20:91.73 37
