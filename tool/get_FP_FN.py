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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def val_model(transforms, type, pre_type, num):
    model = models.densenet121(pretrained=False)
    in_features =  model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    stat_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/select_pre/7/pre_{}fold.pth.tar'.format(num))
    model.load_state_dict(stat_dict['state_dict'])
    model = model.to(device)


    test_label = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Test_fold_{}.csv'.format(num), header=None)
    test_label = test_label.values.tolist()
    test_label = test_label[1:]
    test_dataset = get_pre_dataset('/home/2021/xiaohui/Storage/project_data_crop', test_label, transforms=transforms, type=int(type))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    path_list = []
    total = []
    data_list = []
    for step, (img, label, get_path) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        predict = model(img)
        total.extend(predict.cpu().data.numpy())
        path_list.extend(get_path)
    total = torch.tensor(total)
    P_list = F.softmax(total, dim=1)
    P_list = P_list.numpy()
    pre_list = torch.argmax(total, dim=1)
    pre_list = pre_list.numpy()
    for index, pre in enumerate(pre_list):
        index = int(index)
        pre = int(pre)
        if pre_type == 'FP' and pre == 1:
            data_list.append([path_list[index].split('/')[-1].split('.')[0].split('_')[-1], P_list[index][1], num])
        elif pre_type == 'FN' and pre == 0:
            data_list.append([path_list[index].split('/')[-1].split('.')[0].split('_')[-1], P_list[index][0], num])
        elif pre_type == 'TP' and pre == 1:
            data_list.append([path_list[index].split('/')[-1].split('.')[0].split('_')[-1], P_list[index][1], num])
        elif pre_type == 'TN' and pre == 0:
            data_list.append([path_list[index].split('/')[-1].split('.')[0].split('_')[-1], P_list[index][0], num])
    return data_list


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transforms = TF.Compose([
        TF.Resize(224),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    total_FP_list = []
    total_FN_list = []
    total_TP_list = []
    total_TN_list = []
    for i in range(1, 6):
        FP_list = val_model(transforms, type=0, pre_type='FP', num=i)
        FN_list = val_model(transforms, type=1, pre_type='FN', num=i)
        TP_list = val_model(transforms, type=1, pre_type='TP', num=i)
        TN_list = val_model(transforms, type=0, pre_type='TN', num=i)
        for fp in FP_list:
            total_FP_list.append(fp)
        for fn in FN_list:
            total_FN_list.append(fn)
        for tp in TP_list:
            total_TP_list.append(tp)
        for tn in TN_list:
            total_TN_list.append(tn)
    total_FP_list = np.array(total_FP_list)
    total_FN_list = np.array(total_FN_list)
    total_TN_list = np.array(total_TN_list)
    total_TP_list = np.array(total_TP_list)
    dataframe_FP = pd.DataFrame({'FP': total_FP_list[:, 0], 'P': total_FP_list[:, 1], 'fold': total_FP_list[:, 2]})
    dataframe_FP.to_csv('/home/2021/xiaohui/Storage/Project_code/FP_list.csv', index=False)
    dataframe_FN = pd.DataFrame({'FN': total_FN_list[:, 0], 'P': total_FN_list[:, 1], 'fold': total_FN_list[:, 2]})
    dataframe_FN.to_csv('/home/2021/xiaohui/Storage/Project_code/FN_list.csv', index=False)
    dataframe_TP = pd.DataFrame({'TP': total_TP_list[:, 0], 'P': total_TP_list[:, 1], 'fold': total_TP_list[:, 2]})
    dataframe_TP.to_csv('/home/2021/xiaohui/Storage/Project_code/TP_list.csv', index=False)
    dataframe_TN = pd.DataFrame({'TN': total_TN_list[:, 0], 'P': total_TN_list[:, 1], 'fold': total_TN_list[:, 2]})
    dataframe_TN.to_csv('/home/2021/xiaohui/Storage/Project_code/TN_list.csv', index=False)