import os
from sklearn import datasets
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as tf
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import sys
sys.path.append('/home/2021/xiaohui/Storage/Project_code')
from Dataset import get_dataset, get_pre_dataset
from torch.utils.data import DataLoader
from utils import AverageMeter, ProgressMeter, Metrics, accuracy
import pandas as pd
from os.path import join
import torch.nn.functional as F
from model.bilinear import BiCNN_vgg, BiCNN_rn
from torchsummary import summary
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
from torchstat import stat
# os.environ['TORCH_HOME'] = '/data16/xiaohui/torch_model'
os.environ["WANDB_API_KEY"] = "fd1d49cf1af38d81a6f219ce6364b6dce49b1490"
os.environ["WANDB_MODE"] = "offline"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(name, backbone=None):
    if name == 'res18':
        model = models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif name == 'res34':
        model = models.resnet34(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif name == 'res50':
        model = models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif name == 'res101':
        model = models.resnet101(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif name == 'next50':
        model = models.resnext50_32x4d(pretrained=False)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, 2)
    elif name == 'vgg16':
        model = models.vgg16_bn(pretrained=False)
        in_features = model.classifier[0].in_features
        model.classifier = torch.nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 2),
        )
    elif name == 'vgg13':
        model = models.vgg13_bn(pretrained=False)
        in_features = model.classifier[0].in_features
        model.classifier = torch.nn.Sequential(
        nn.Linear(in_features, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 2),
        )
    elif name == 'dense121':
        model = models.densenet121(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
    elif name == 'dense161':
        model = models.densenet161(pretrained=False)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
    elif name == 'BiCNN_vgg':
        model = BiCNN_vgg(pretrain=False, backbone=backbone)
    elif name == 'BiCNN_rn':
        model = BiCNN_rn(backbone=backbone, pretrain=False)
    else:
        raise Exception('model error')
    return model

if __name__ == '__main__':
    model1 = get_model('BiCNN_vgg', 'vgg16')
    model2 = get_model('vgg16')
    stat(model1, (3, 224, 224))
    stat(model2, (3, 224, 224))
    # summary(model1, (3, 224, 224), device='cpu')
    # summary(model2, (3, 224, 224), device='cpu')
