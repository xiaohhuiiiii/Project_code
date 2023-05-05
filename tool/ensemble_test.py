import os
from sklearn import datasets
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as TF
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
import sys
sys.path.append('/home/2021/xiaohui/Storage/Project_code')
from Dataset import get_pre_dataset
from utils import AverageMeter, ProgressMeter, Metrics, accuracy
import pandas as pd
from os.path import join
import torch.nn.functional as F
from model.bilinear import BiCNN_vgg16
import random



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(model, val_loader, criterion, val_param, mode='val'):
    top1 = AverageMeter('Acc', ':6.2f')
    if mode == 'test':
        progress = ProgressMeter(
            len(val_loader),
            [top1],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(val_loader),
            [ top1],
            prefix='Validate: ')
    metrics = Metrics()

    # switch to evaluation mode
    model.eval()
    total = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            metrics.store(output, target)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))

            top1.update(acc1[0].item(), images.size(0))

        print(' * Acc {:.3f}'
              .format(top1.avg))
        if mode == 'test':
            recall, precision, F1_score, auc_score = metrics.cal_metrics(1)
            print('recall: {}   precision: {}   F1_score: {}   auc: {}'.format(str(recall), str(precision), str(F1_score), str(auc_score)))


    return top1.avg


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(1998)
    transforms1 = TF.Compose([
        TF.Resize(224),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transforms2 = TF.Compose([
        TF.Resize(224),
        TF.RandomHorizontalFlip(p=1.0),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transforms3 = TF.Compose([
        TF.Resize(224),
        TF.RandomRotation(45),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    transforms4 = TF.Compose([
        TF.Resize(224),
        TF.RandomHorizontalFlip(p=1.0), 
        TF.RandomRotation(45),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])