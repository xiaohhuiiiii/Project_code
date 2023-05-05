from difflib import diff_bytes
import enum
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
import wandb
import random
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve
import argparse
# os.environ['TORCH_HOME'] = '/data16/xiaohui/torch_model'
os.environ["WANDB_API_KEY"] = "fd1d49cf1af38d81a6f219ce6364b6dce49b1490"
os.environ["WANDB_MODE"] = "offline"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


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
    return statistics

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
    return statistics

def bootstrap_acc(y, pred, bootstraps = 1000, fold_size = 400):
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
        count = 0
        for j in range(len(y_sample)):
            if pred_sample[j] == y_sample[j]:
                count += 1
        statistics[i] = count / len(y_sample)
    return statistics

def bootstrap_sen(y, pred, bootstraps = 1000, fold_size = 400):
    statistics_sens = np.zeros(bootstraps)
    statistics_spec = np.zeros(bootstraps)
    statistics_prec = np.zeros(bootstraps)
    statistics_f1 = np.zeros(bootstraps)

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
        pre_correct = list(0 for i in range(2))
        pre_all = list(0 for i in range(2))
        all_label = list(0 for i in range(2))
        for j in range(len(y_sample)):
            pre_all[pred_sample[j]] += 1
            all_label[y_sample[j]] += 1
            if pred_sample[j] == y_sample[j]:
                pre_correct[y_sample[j]] += 1
        statistics_sens[i] = pre_correct[1] / all_label[1]
        statistics_spec[i] = pre_correct[0] / all_label[0]
        if pre_all[1] == 0:
            statistics_prec[i] = 1
        else:
            statistics_prec[i] = pre_correct[1] / pre_all[1]
        if (statistics_sens[i] + statistics_prec[i]) == 0:
            statistics_f1[i] = 1
        else:
            statistics_f1[i] = 2 * (statistics_sens[i] * statistics_prec[i]) / (statistics_sens[i] + statistics_prec[i])
    return statistics_sens, statistics_spec, statistics_prec, statistics_f1

def bootstrap_val(y, pred, pre_label, bootstraps = 1000, fold_size = 400):
    result_ci = {}
    statistics_acc = np.zeros(bootstraps)
    statistics_roc = np.zeros(bootstraps)
    statistics_prc = np.zeros(bootstraps)
    statistics_sens = np.zeros(bootstraps)
    statistics_spec = np.zeros(bootstraps)
    statistics_prec = np.zeros(bootstraps)
    statistics_f1 = np.zeros(bootstraps)

    for i in range(bootstraps):
        for fold in range(5):
            df = pd.DataFrame(columns=['y', 'pred', 'pre_label'])
            df.loc[:, 'y'] = y[fold]
            df.loc[:, 'pred'] = pred[fold]
            df.loc[:, 'pre_label'] = pre_label[fold]
            df_pos = df[df.y == 1]
            df_neg = df[df.y == 0]
            prevalence = len(df_pos) / len(df)

            pos_sample = df_pos.sample(n = int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n = int(fold_size * (1-prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            pre_label_sample = np.concatenate([pos_sample.pre_label.values, neg_sample.pre_label.values])
            pre_correct = list(0 for i in range(2))
            pre_all = list(0 for i in range(2))
            all_label = list(0 for i in range(2))
            count = 0
            prec, rec, _ = precision_recall_curve(y_sample, pred_sample)
            for j in range(len(y_sample)):
                pre_all[pre_label_sample[j]] += 1
                all_label[y_sample[j]] += 1
                if pre_label_sample[j] == y_sample[j]:
                    count += 1
                    pre_correct[y_sample[j]] += 1
            statistics_roc[i] += roc_auc_score(y_sample, pred_sample)
            statistics_prc[i] += auc(rec, prec)
            statistics_acc[i] += count / len(y_sample)
            statistics_sens[i] += pre_correct[1] / all_label[1]
            this_sens = pre_correct[1] / all_label[1]
            statistics_spec[i] += pre_correct[0] / all_label[0]
            if pre_all[1] == 0:
                statistics_prec[i] += 1
                this_prec = 1
            else:
                statistics_prec[i] += pre_correct[1] / pre_all[1]
                this_prec = pre_correct[1] / pre_all[1]
            if (this_sens + this_prec) == 0:
                statistics_f1[i] += 1
            else:
                statistics_f1[i] += 2 * (this_sens * this_prec) / (this_sens + this_prec)
        statistics_acc[i] /= 5
        statistics_f1[i] /= 5
        statistics_prc[i] /= 5
        statistics_roc[i] /= 5
        statistics_prec[i] /= 5
        statistics_sens[i] /= 5
        statistics_spec[i] /= 5


    result_ci['acc'] = statistics_acc
    result_ci['roc'] = statistics_roc
    result_ci['prc'] = statistics_prc
    result_ci['recall'] = statistics_sens
    result_ci['specifity'] = statistics_spec
    result_ci['precision'] = statistics_prec
    result_ci['f1'] = statistics_f1

    return result_ci

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
    total_pre = np.array(total_pre, dtype='float64')
    total_label = np.array(total_label)

    return total_pre, total_label, total_name

def get_model(name, backbone):
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
    model = model.to(device)
    return model

def ensemble_val(model_path, model_name, backbone_name, val_path, args):

    all_y = []
    all_pred = []
    all_pre_label = []

    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    for i in tqdm(range(1, args.fold + 1)):
        model = get_model(model_name, backbone_name)
        if args.type == 'best':
            state_dict = torch.load(join(model_path, '{}fold_best.pth.tar'.format(i)))
        elif args.type == 'last':
            state_dict = torch.load(join(model_path, '{}fold.pth.tar'.format(i)))
        model.load_state_dict(state_dict['state_dict'])

        val_csv = join(val_path, 'Val_fold_{}.csv'.format(i))
        val_label = pd.read_csv(val_csv, header=None).values.tolist()
        val_label = val_label[1:]
        val_data = get_dataset('/home/2021/xiaohui/Storage/project_data_crop_new', val_label, val_transforms, get_name=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

        name_list = None
        pre, total_label, name_list = get_pre(model, val_loader, name_list)
        pre_label = np.argmax(pre, axis=1)

        all_y.append(total_label)
        all_pred.append(pre[:, 1])
        all_pre_label.append(pre_label)
    result_ci = bootstrap_val(all_y, all_pred, all_pre_label)
    
    return result_ci
    

def ensemble_test(model_path, model_name, backbone_name, val_path, args):
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    val_csv = join(val_path, 'Test.csv')
    val_label = pd.read_csv(val_csv, header=None).values.tolist()
    val_label = val_label[1:]
    val_data = get_dataset('/home/2021/xiaohui/Storage//project_data_crop_new', val_label, val_transforms, get_name=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    for i in tqdm(range(1, args.fold + 1)):
        model = get_model(model_name, backbone_name)
        if args.type == 'best':
            state_dict = torch.load(join(model_path, '{}fold_best.pth.tar'.format(i)))
        elif args.type == 'last':
            state_dict = torch.load(join(model_path, '{}fold.pth.tar'.format(i)))
        model.load_state_dict(state_dict['state_dict'])

        name_list = None
        pre, total_label, name_list = get_pre(model, val_loader, name_list)
        if i == 1:
            pre_total = np.array(pre, dtype='float64')
        else:
            pre_total += np.array(pre, dtype='float64')
    pre_total /= 5
    total_label = np.array(total_label)
    pre_label = np.argmax(pre_total, axis=1)
    pre_total = np.array(pre_total)

    sens_score, spec_score, prec_score, f1_score = bootstrap_sen(total_label, pre_label)
    
    
    acc_score = bootstrap_acc(total_label, pre_label)
    
    auroc_score = bootstrap_roc(total_label, pre_total[:, 1])
    
    auprc_score = bootstrap_prc(total_label, pre_total[:, 1])

    return acc_score, prec_score, sens_score, spec_score, f1_score, auroc_score, auprc_score

if __name__ == '__main__':
    set_seed(1998)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help='To validate the test set or val set')
    parser.add_argument('-m', '--model_list', type=str, required=True, nargs='+')
    parser.add_argument('-b', '--backbone_list', default=None, type=str, nargs='+')

    parser.add_argument('-f', '--fold', type=int, default=5)
    parser.add_argument('--lossweight', default=None, type=str, nargs='+')
    parser.add_argument('--root', default='/home/2021/xiaohui/Storage/Project_code/new_checkpoint/model_5fold', type=str)
    parser.add_argument('--type', default='best', type=str)

    args = parser.parse_args()
    for num, i in enumerate(range(4, 9)):
        model_path = join(args.root, str(i), args.type)
        if num >= len(args.model_list):
            model_name = args.model_list[-1]
        else:
            model_name = args.model_list[num]
        if args.backbone_list is not None:
            if num >= len(args.backbone_list):
                backbone_name = args.backbone_list[-1]
            else:
                backbone_name = args.backbone_list[num]
        else:
            backbone_name = None
        print('model:{}, backbone:{}'.format(model_name, backbone_name))
        val_path = '/home/2021/xiaohui/Storage/Project_code/label_{}fold'.format(args.fold)
        args.model = model_name
        args.backbone = backbone_name
        if args.lossweight is not None:
            args.weight = args.lossweight[num]

        if args.test:
            print('Using Test...')
            acc, precision, recall, specifity, f1, auroc, auprc = ensemble_test(model_path, model_name, backbone_name, val_path, args)
            data = {'Acc' : acc, 'Precision' : precision, 'Sensitivity' : recall, 'Specifity' : specifity, 'F1_score' : f1, 'Auroc' : auroc, 'Auprc' : auprc}
        else:
            print('Using Val...')
            result = ensemble_val(model_path, model_name, backbone_name, val_path, args)
            data = {'Acc' : result['acc'], 'Precision' : result['precision'], 'Sensitivity' : result['recall'], 'Specifity' : result['specifity'], 'F1_score' : result['f1'], 'Auroc' : result['roc'], 'Auprc' : result['prc']}
        dataframe = pd.DataFrame(data)
        if backbone_name is not None:
            dataframe.to_csv('/home/2021/xiaohui/Storage/Project_code/bootstrap_val/{}_{}.csv'.format(model_name, backbone_name))
        else:
            if(i > 7):
                dataframe.to_csv('/home/2021/xiaohui/Storage/Project_code/bootstrap_val/{}_{}.csv'.format(model_name, 'scrach'))
            else:
                dataframe.to_csv('/home/2021/xiaohui/Storage/Project_code/bootstrap_val/{}_{}.csv'.format(model_name, 'finetune'))
        