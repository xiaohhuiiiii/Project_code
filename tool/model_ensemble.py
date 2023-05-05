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
sys.path.append('./Project_code')
from Dataset import get_dataset, get_pre_dataset
from torch.utils.data import DataLoader
from utils import AverageMeter, ProgressMeter, Metrics, accuracy
import pandas as pd
from os.path import join
import torch.nn.functional as F
from model.bilinear import BiCNN_vgg, BiCNN_rn
import random
from sklearn.metrics import roc_auc_score, auc, roc_curve

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_pre(model, test_loader, name_list):

    # switch to evaluation mode
    model.eval()
    total_pre = []
    total_label = []
    total_name = []
    with torch.no_grad():
        for i, (images, target, name) in enumerate(test_loader):

            images = images.to(device)

            # compute output
            output = model(images)
            pre = F.softmax(output, dim=1)
            total_pre.extend(pre.cpu().data.tolist())
            total_label.extend(target.data.tolist())
            total_name.extend(list(name))
    if name_list != None:
        for i in range(len(total_name)):
            assert total_name[i] == name_list[i]

    return total_pre, total_label, total_name

def ensemble_vote(model_path, test_loader, get_csv, backbone_list, group):
    model_list = os.listdir(model_path)
    for i in tqdm(range(len(model_list))):
        model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(model_path, model_list[i]))
        model.load_state_dict(state_dict['state_dict'])
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        label = np.argmax(np.array(pre), axis=1)
        if i == 0:
            pre_label = label
        else:
            pre_label += label
    pre_label[pre_label < 2.5] = 0
    pre_label[pre_label > 2.5] = 1
    pre_label = np.array(pre_label)
    total_label = np.array(total_label)
    pre_csv = pd.DataFrame({'name': name_list, 'true_label': total_label, 'pre_label': pre_label})
    pre_correct = list(0 for i in range(2))
    pre_total = list(0 for i in range(2))
    all_label = list(0 for i in range(2))
    for index in range(len(pre_label)):
        pre_total[pre_label[index]] += 1
        all_label[total_label[index]] += 1
        if pre_label[index] == total_label[index]:
            pre_correct[pre_label[index]] += 1
    recall = pre_correct[1] / all_label[1]
    specificity = pre_correct[0] / all_label[0]
    acc = (pre_correct[0] + pre_correct[1]) / len(total_label)
    if get_csv:
        pre_csv.to_csv('./Project_code/pre_csv.csv')
    return acc, recall, specificity

def ensemble_mean(model_path, test_loader, get_csv, group, backbone_list):
    model_list = os.listdir(model_path)
    for i in tqdm(range(len(model_list))):
        if backbone_list[str(group)] == 'vgg16':
            model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        else:
            model = BiCNN_rn(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(model_path, model_list[i]))
        model.load_state_dict(state_dict['state_dict'])
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        if i == 0:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
    pre_total /= 5
    total_label = np.array(total_label)
    pre_label = np.argmax(pre_total, axis=1)
    pre_total = np.array(pre_total)
    if get_csv is not None and get_csv == 'prob':
        pre_csv = pd.DataFrame({'name': name_list, 'true_label': total_label, 'pre_prob_0': pre_total[:, 0], 'pre_prob_1': pre_total[:, 1]})
    elif get_csv is not None and get_csv == 'label':
        pre_csv = pd.DataFrame({'name': name_list, 'true_label': total_label, 'pre_label': pre_label})
    auc_score = roc_auc_score(total_label, pre_total[:, 1])
    pre_correct = list(0 for i in range(2))
    pre_total = list(0 for i in range(2))
    all_label = list(0 for i in range(2))
    for index in range(len(pre_label)):
        pre_total[pre_label[index]] += 1
        all_label[total_label[index]] += 1
        if pre_label[index] == total_label[index]:
            pre_correct[pre_label[index]] += 1
    recall = pre_correct[1] / all_label[1]
    specificity = pre_correct[0] / all_label[0]
    acc = (pre_correct[0] + pre_correct[1]) / len(total_label)
    if get_csv is not None:
        pre_csv.to_csv('./pre_list/pre__mean_{}.csv'.format(group))
    return acc, recall, specificity, auc_score


if __name__ == '__main__':
    backbone_list = {'1': 'res18', '2': 'res34', '3': 'res50', '4': 'res101', '5': 'next50', '6': 'vgg16'}
    mode = 'mean'
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_csv = './label_5fold/Test.csv'
    test_label = pd.read_csv(test_csv).values.tolist()
    test_label = test_label[1:]
    test_data = get_dataset('./project_data_crop_new', test_label, val_transforms, get_name=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    if mode == 'mean':
        for i in range(6, 7):
            acc, recall, specificity, auc_score = ensemble_mean('./checkpoint/BiCNN_select_5fold/{}'.format(i), test_loader, get_csv=None, group=i, backbone_list=backbone_list)
            print('Acc: {:.3f}, recall: {:.3f}, specificity: {:.3f}, Auc: {:.3f}'.format(acc, recall, specificity, auc_score))
    elif mode == 'vote':
        acc, recall, specificity = ensemble_vote('./checkpoint/BiCNN_select_5fold/6', test_loader, get_csv=True)
        print('Acc: {:.3f}, recall: {:.3f}, specificity: {:.3f}'.format(acc, recall, specificity))