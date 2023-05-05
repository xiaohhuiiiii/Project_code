from http.client import PRECONDITION_FAILED
import os
from socket import SocketIO
import tarfile
import numpy as np
import torch
import shutil
import cv2
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc, roc_curve

# 保存指标
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# 计算acc
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

# 计算recall、precision、f1score、AUC
class Metrics():
    def __init__(self):
        self.pre_correct = list(0 for i in range(2))
        self.class_total = list(0 for i in range(2))
        self.pre_total = list(0 for i in range(2))
        self.predictions = []
        self.labels = []
        self.score = []

    def store(self, outputs, targets):
        targets = targets.cpu().data
        predition = outputs.argmax(1).cpu().data
        score = F.softmax(outputs, dim=1)
        score = score.cpu().data
        for index in range(len(targets)):
            self.class_total[targets[index]] += 1
            self.pre_total[predition[index]] += 1
            if predition[index] == targets[index]:
                self.pre_correct[targets[index]] += 1
        self.predictions.extend(predition.tolist())
        self.labels.extend(targets.tolist())
        self.score.extend(score.tolist())

    def cal_metrics(self):
        if self.class_total[1] == 0:
            recall = 0
        else:
            recall = self.pre_correct[1] / self.class_total[1]
        if self.class_total[0] == 0:
            specificity = 0
        else:
            specificity = self.pre_correct[0] / self.class_total[0]
        if self.pre_total[1] == 0:
            precision = 0
        else:
            precision = self.pre_correct[1] / self.pre_total[1]
        self.score = np.array(self.score)
        auc_score = roc_auc_score(self.labels, self.score[:, 1])
        self.score = list(self.score)
        if (precision + recall) == 0:
            F1_score = 0
        else:
            F1_score = 2 * (precision * recall) / (precision + recall)
        return recall, specificity, precision, F1_score, auc_score
        
    def reset(self):
        self.pre_correct = list(0 for i in range(2))
        self.class_total = list(0 for i in range(2))
        self.pre_total = list(0 for i in range(2))
        self.predictions = []
        self.labels = []

# 打印信息
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class manual_Compose:
    def __init__(self, transforms, labels):
        self.transforms = transforms
        self.labels = labels

    def __call__(self, img):
        for i, t in enumerate(self.transforms):
            if self.labels[i] == 0:
                img = t(img)
            elif self.labels[i] == 1:
                img = np.array(img)
                img = t(image=img)
                img = img["image"]
                img = Image.fromarray(img)
            else:
                raise Exception('transform error')
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


