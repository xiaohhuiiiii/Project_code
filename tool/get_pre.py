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
from Dataset import get_dataset
import pandas as pd
from os.path import join
import torch.nn.functional as F
from model.bilinear import BiCNN

def val_model(transforms, pre_list, label_list, name_list, num):
    model = BiCNN()
    stat_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/clean/2/pre_{}fold.pth.tar'.format(num))
    model.load_state_dict(stat_dict['state_dict'])
    model = model.to(device)


    test_label = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Test_fold_{}.csv'.format(num), header=None)
    test_label = test_label.values.tolist()
    test_label = test_label[1:]
    test_dataset = get_dataset('/home/2021/xiaohui/Storage/project_data_crop', test_label, transforms=transforms, get_name=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    total = []
    for step, (img, label, img_name) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        predict = model(img)
        total.extend(predict.cpu().data.numpy())
        name_list.extend(img_name)
        label = label.data.numpy()
        label_list.extend(label)
    total = torch.tensor(total)
    pre = torch.argmax(total, dim=1)
    pre = pre.numpy()
    pre_list.extend(pre)
    return pre_list, label_list, name_list

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transforms = TF.Compose([
        TF.Resize(224),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    pre_list = []
    label_list = []
    name_list = []
    for i in tqdm(range(1, 6)):
        pre_list, label_list, name_list = val_model(transforms, pre_list, label_list, name_list, i)
    dataframe = pd.DataFrame({'name': name_list, 'true_label': label_list, 'pre_label': pre_list})
    dataframe.to_csv('/home/2021/xiaohui/Storage/Project_code/pre_list.csv', index=False)