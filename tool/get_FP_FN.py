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
import pandas as pd
from os.path import join
import torch.nn.functional as F
from model.bilinear import BiCNN_vgg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pre(model, test_loader, name_list):

    # switch to evaluation mode
    model.eval()
    total_pre = []
    total_label = []
    total_name = []
    with torch.no_grad():
        print('test', len(test_loader))
        for i, (images, target, name) in enumerate(test_loader):
            print(333)
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


def val_model(transforms, type, pre_type):
    test_label = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label_5fold/Test.csv', header=None)
    test_label = test_label.values.tolist()
    test_label = test_label[1:]
    test_dataset = get_pre_dataset('/home/2021/xiaohui/Storage/project_data_crop_new', test_label, transforms=transforms, type=int(type))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    print(len(test_dataset))
    for i in tqdm(range(1, 6)):
        model = BiCNN_vgg(backbone='vgg16').to(device)
        stat_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/new_checkpoint/BiCNN_5fold/1/best/{}fold_best.pth.tar'.format(i))
        model.load_state_dict(stat_dict['state_dict'])
        model = model.to(device)
        name_list = None
        print(111)
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        print(222)
        if i == 1:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
    pre_total /= 5
    total_label = np.array(total_label)
    print(total_label)
    pre_label = np.argmax(pre_total, axis=1)
    pre_total = np.array(pre_total)
    data_list = []
    for index, pre in enumerate(pre_label):
        index = int(index)
        pre = int(pre)
        if pre_type == 'FP' and pre == 1:
            data_list.append(name_list[index].split('/')[-1].split('.')[0].split('_')[-1])
        elif pre_type == 'FN' and pre == 0:
            data_list.append(name_list[index].split('/')[-1].split('.')[0].split('_')[-1])
        elif pre_type == 'TP' and pre == 1:
            data_list.append(name_list[index].split('/')[-1].split('.')[0].split('_')[-1])
        elif pre_type == 'TN' and pre == 0:
            data_list.append(name_list[index].split('/')[-1].split('.')[0].split('_')[-1])
    return data_list


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transforms = TF.Compose([
        TF.Resize(224),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    FP_list = val_model(transforms, type=0, pre_type='FP')
    FN_list = val_model(transforms, type=1, pre_type='FN')
    TP_list = val_model(transforms, type=1, pre_type='TP')
    TN_list = val_model(transforms, type=0, pre_type='TN')
    total_FP_list = np.array(FP_list)
    total_FN_list = np.array(FN_list)
    total_TN_list = np.array(TN_list)
    total_TP_list = np.array(TP_list)
    dataframe_FP = pd.DataFrame({'FP': total_FP_list})
    dataframe_FP.to_csv('/home/2021/xiaohui/Storage/Project_code/visualization/FP_list.csv', index=False)
    dataframe_FN = pd.DataFrame({'FN': total_FN_list})
    dataframe_FN.to_csv('/home/2021/xiaohui/Storage/Project_code/visualization/FN_list.csv', index=False)
    dataframe_TP = pd.DataFrame({'TP': total_TP_list})
    dataframe_TP.to_csv('/home/2021/xiaohui/Storage/Project_code/visualization/TP_list.csv', index=False)
    dataframe_TN = pd.DataFrame({'TN': total_TN_list})
    dataframe_TN.to_csv('/home/2021/xiaohui/Storage/Project_code/visualization/TN_list.csv', index=False)