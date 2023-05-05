import os
from sklearn import datasets
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as TF
import numpy as np
import torch.nn as nn
from tqdm import tqdm 
from Dataset import get_pre_dataset
import glob
import shutil
import pandas as pd
from os.path import join

def val_model(transforms, type, pre_type):
    model = models.densenet121(pretrained=False)
    in_features =  model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    stat_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/model_select_pre/7/1fold.pth.tar')
    model.load_state_dict(stat_dict['state_dict'])
    model = model.to(device)


    test_label = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Test.csv', header=None)
    test_label = test_label.values.tolist()
    test_label = test_label[1:]
    test_dataset = get_pre_dataset('/home/2021/xiaohui/Storage/project_data_crop', test_label, transforms=transforms, type=int(type))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    path_list = []
    total = []
    for step, (img, label, get_path) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        predict = model(img)
        total.extend(predict.cpu().data.numpy())
        path_list.extend(get_path)
    total = torch.tensor(total)
    topk, index = total.topk(5, 0, True, True)
    print(topk)
    index = index.numpy()
    for i, j in index:
        i = int(i)
        j = int(j)
        img_dir_0 = path_list[i]
        img_dir_1 = path_list[j]
        if type == 0:
            if pre_type == 'FP':
                shutil.copy(img_dir_1, join('/home/2021/xiaohui/Storage/Project_code/topk', 'FP', img_dir_1.split('/')[-1]))
            elif pre_type == 'TN':
                shutil.copy(img_dir_0, join('/home/2021/xiaohui/Storage/Project_code/topk', 'TN', img_dir_0.split('/')[-1]))
            else:
                raise Exception('False')
        elif type == 1:
            if pre_type == 'FN':
                shutil.copy(img_dir_0, join('/home/2021/xiaohui/Storage/Project_code/topk', 'FN', img_dir_0.split('/')[-1]))
            elif pre_type == 'TP':
                shutil.copy(img_dir_1, join('/home/2021/xiaohui/Storage/Project_code/topk', 'TP', img_dir_1.split('/')[-1]))
            else:
                raise Exception('False')
        else:
            raise Exception('Do not have this type')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transforms = TF.Compose([
        TF.Resize(224),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    val_model(transforms, type=0, pre_type='TN')
    val_model(transforms, type=0, pre_type='FP')
    val_model(transforms, type=1, pre_type='TP')
    val_model(transforms, type=1, pre_type='FN')