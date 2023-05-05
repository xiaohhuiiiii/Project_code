from cProfile import label
import enum
from tokenize import group
import torch
import sys
sys.path.append('/home/2021/xiaohui/Storage/Project_code')
from model.bilinear import BiCNN_vgg, BiCNN_rn
import os
from os.path import join
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
import numpy as np
from Dataset import get_dataset
import pandas as pd
import torchvision.transforms as tf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import random

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
    if name_list is not None:
        for i in range(len(total_name)):
            assert total_name[i] == name_list[i]
    return total_pre, total_label, total_name


def plt_ROC_fold(root, test_loader, name, save_path, group, backbone_list, img_name):
    model_list = os.listdir(root)
    Font = {'size': 10, 'family': 'Times New Roman'}
    color_list = ['r','k','b','y','c','g','m']
    plt.figure(figsize=(6, 6))
    for fold, model_name in enumerate(tqdm(model_list)):
        model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(root, model_name))
        model.load_state_dict(state_dict['state_dict'])
        model.eval()
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        if fold == 0:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
        total_label = np.array(total_label)
        fpr, tpr, _ = roc_curve(total_label, pre_total[:, 1])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label='{}fold: {}'.format(fold + 1, auc_score), color=color_list[fold])
    plt.scatter([0.26], [0.64], cmap='y', label='dental student')
    plt.scatter([0.44], [0.90], cmap='b', label='junior dentist')
    plt.scatter([0.08], [0.53], cmap='g', label='senior dentist')
    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.title(name.upper())
    plt.savefig(join(save_path, img_name))

def plt_ROC_mean(root, test_loader, name, save_path, group, backbone_list, avarage=False):
    Font = {'size': 12, 'family': 'Times New Roman'}
    model_list = os.listdir(root)
    plt.figure(figsize=(8, 8), dpi=300)
    for i in tqdm(range(len(model_list))):
        if backbone_list[str(group)] == 'vgg16':
            model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        else:
            model = BiCNN_rn(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(root, model_list[i]))
        model.load_state_dict(state_dict['state_dict'])
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        if i == 0:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
    pre_total /= 5
    total_label = np.array(total_label)
    fpr, tpr, _ = roc_curve(total_label, pre_total[:, 1])
    auc_score, ci = bootstrap_roc(total_label, pre_total[:, 1])
    plt.plot(fpr, tpr, 'b', label='AUROC: {:.2f}({:.2f},{:.2f})'.format(auc_score, ci[0], ci[1]), color='r')
    if avarage:
        plt.scatter([0.30], [0.73], c='g', label='Dental student', marker='^')
        plt.scatter([0.31], [0.87], c='g', label='Dental student with BCNN', marker='^')
        plt.scatter([0.32], [0.84], c='b', label='Junior dentist')
        plt.scatter([0.32], [0.89], c='b', label='Junior dentist with BCNN', marker='^')
        plt.scatter([0.13], [0.62], c='k', label='Senior dentist')
        plt.scatter([0.15], [0.80], c='k', label='Senior dentist with BCNN', marker='^')
        img_name = 'new_ROC_mean.png'
    else:
        plt.scatter([0.26], [0.64], c='#990033', label='Dental student1', marker='^', s=50)
        plt.scatter([0.28], [0.85], c='#990033', marker='^', s=50)
        plt.scatter([0.33], [0.81], c='#FF9999', label='Dental student2', marker='^', s=50)
        plt.scatter([0.34], [0.88], c='#FF9999', marker='^', s=50)
        plt.scatter([0.44], [0.90], c='#003300', label='Junior dentist1', marker='.', s=200)
        plt.scatter([0.44], [0.92], c='#003300', marker='.', s=200)
        plt.scatter([0.20], [0.77], c='#99CC66', label='Junior dentist2', marker='.', s=200)
        plt.scatter([0.21], [0.85], c='#99CC66', marker='.', s=200)
        plt.scatter([0.08], [0.53], c='#003399', label='Senior dentist1', marker='s', s=50)
        plt.scatter([0.12], [0.78], c='#003399', marker='s', s=50)
        plt.scatter([0.17], [0.71], c='#0099CC', label='Senior dentist2', marker='s', s=50)
        plt.scatter([0.19], [0.81], c='#0099CC', marker='s', s=50)
        img_name = 'new_ROC.png'

    plt.legend(loc='lower right', prop=Font)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    plt.tick_params(labelsize=15)
    plt.title(name.upper())
    plt.savefig(join(save_path, img_name))

def plt_PR_fold(root, test_loader, name, save_path, group, backbone_list, img_name):
    model_list = os.listdir(root)
    Font = {'size': 10}
    Font2 = {'size': 10}
    color_list = ['r','k','b','y','c','g','m']
    plt.figure(figsize=(6, 6))
    for fold, model_name in enumerate(tqdm(model_list)):
        model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(root, model_name))
        model.load_state_dict(state_dict['state_dict'])
        model.eval()
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        if fold == 0:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
        total_label = np.array(total_label)
        precision, recall, _ = precision_recall_curve(total_label, pre_total[:, 1])
        area = average_precision_score(total_label, pre_total[:, 1])
        plt.plot(recall, precision, 'b', label='{}fold: {}'.format(fold, area), color=color_list[fold])
    plt.legend(loc='lower right', prop=Font)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', Font)
    plt.xlabel('Recall', Font)
    plt.tick_params(labelsize=15)
    plt.title(name.upper())
    plt.savefig(join(save_path, img_name))

def plt_PR_mean(root, test_loader, name, save_path, group, backbone_list, avarage=False):
    Font = {'size': 12, 'family': 'Times New Roman'}
    mm = str('Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r').split(', ')
    model_list = os.listdir(root)
    plt.figure(figsize=(8, 8), dpi=300)
    for i in tqdm(range(len(model_list))):
        if backbone_list[str(group)] == 'vgg16':
            model = BiCNN_vgg(backbone=backbone_list[str(group)]).to(device)
        else:
            model = BiCNN_rn(backbone=backbone_list[str(group)]).to(device)
        state_dict = torch.load(join(root, model_list[i]))
        model.load_state_dict(state_dict['state_dict'])
        name_list = None
        pre, total_label, name_list = get_pre(model, test_loader, name_list)
        if i == 0:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
    pre_total /= 5
    total_label = np.array(total_label)
    precision, recall, _ = precision_recall_curve(total_label, pre_total[:, 1])
    area, ci = bootstrap_prc(total_label, pre_total[:, 1])
    plt.plot(recall, precision, 'b', label='AUPRC: {:.2f}({:.2f},{:.2f})'.format(area, ci[0], ci[1]), color='r')

    if avarage:
        plt.scatter([0.73], [0.51], c='g', label='Dental student')
        plt.scatter([0.87], [0.54], c='g', label='Dental student with BCNN', marker='^')
        plt.scatter([0.84], [0.54], c='b', label='Junior dentist')
        plt.scatter([0.89], [0.55], c='b', label='Junior dentist with BCNN', marker='^')
        plt.scatter([0.62], [0.69], c='k', label='Senior dentist')
        plt.scatter([0.80], [0.69], c='k', label='Senior dentist with BCNN', marker='^')
        img_name = 'new_PR_mean.png'
    else:
        plt.scatter([0.64], [0.51], c='#990033', label='Dental student1', marker='^', s=50)
        plt.scatter([0.85], [0.56], c='#990033', marker='^', s=50)
        plt.scatter([0.81], [0.51], c='#FF9999', label='Dental student2', marker='^', s=50)
        plt.scatter([0.88], [0.52], c='#FF9999', marker='^', s=50)
        plt.scatter([0.90], [0.47], c='#003300', label='Junior dentist1', marker='.', s=200)
        plt.scatter([0.92], [0.47], c='#003300', marker='.', s=200)
        plt.scatter([0.77], [0.62], c='#99CC66', label='Junior dentist2', marker='.', s=200)
        plt.scatter([0.85], [0.63], c='#99CC66', marker='.', s=200)
        plt.scatter([0.53], [0.73], c='#003399', label='Senior dentist1', marker='s', s=50)
        plt.scatter([0.78], [0.73], c='#003399', marker='s', s=50)
        plt.scatter([0.71], [0.64], c='#0099CC', label='Senior dentist2', marker='s', s=50)
        plt.scatter([0.81], [0.65], c='#0099CC', marker='s', s=50)
        img_name = 'new_PR.png'

    plt.legend(loc='lower left', prop=Font)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision', Font)
    plt.xlabel('Recall', Font)
    plt.tick_params(labelsize=15)
    plt.title(name.upper())
    plt.savefig(join(save_path, img_name))

def bootstrap_roc(y, pred, bootstraps = 1000, fold_size = 400):
    print(len(y))
    statistics = np.zeros(bootstraps)

    df = pd.DataFrame(columns=['y', 'pred'])
    # df.
    df.loc[:, 'y'] = y
    df.loc[:, 'pred'] = pred
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    prevalence = len(df_pos) / len(df)
    for i in range(bootstraps):
        pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
        neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
        score = roc_auc_score(y_sample, pred_sample)
        statistics[i] = score
    result = np.quantile(statistics, [.025, 0.975])
    return statistics.mean(), result

def bootstrap_prc(y, pred, bootstraps = 1000, fold_size = 400):
    statistics = np.zeros(bootstraps)

    df = pd.DataFrame(columns=['y', 'pred'])
    # df.
    df.loc[:, 'y'] = y
    df.loc[:, 'pred'] = pred
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]
    prevalence = len(df_pos) / len(df)
    for i in range(bootstraps):
        pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
        neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

        y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
        pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
        prec, rec, _ = precision_recall_curve(y_sample, pred_sample)
        score = auc(rec, prec)
        statistics[i] = score
    result = np.quantile(statistics, [.025, 0.975])
    return statistics.mean(), result

if __name__ == '__main__':
    set_seed(1998)
    backbone_list = {'1': 'vgg16', '2': 'res34', '3': 'res50', '4': 'res101', '5': 'next50', '6': 'vgg16'}

    root = '/home/2021/xiaohui/Storage/Project_code/new_checkpoint/BiCNN_5fold/1/best'
    save_path = '/home/2021/xiaohui/Storage/Project_code/look/'
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_csv = '/home/2021/xiaohui/Storage/Project_code/label_5fold/Test.csv'
    test_label = pd.read_csv(test_csv).values.tolist()
    test_label = test_label[1:]
    test_data = get_dataset('/home/2021/xiaohui/Storage/project_data_crop_new', test_label, val_transforms, get_name=True)
    test_loader = DataLoader(test_data, batch_size=32)
    plt_ROC_mean(root, test_loader, 'BCNN_vgg16', save_path, group=1, backbone_list=backbone_list, avarage=False)
    plt_PR_mean(root, test_loader, 'BCNN_vgg16', save_path, group=1, backbone_list=backbone_list, avarage=False)
