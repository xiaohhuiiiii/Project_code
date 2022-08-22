import os
import tarfile
import numpy as np
import torch
import shutil
import cv2
import matplotlib.pyplot as plt
from PIL import ImageEnhance, Image
from sklearn.metrics import roc_auc_score, auc, roc_curve
envpath = '/data16/xiaohui/anaconda3/envs/machine/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

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

    def store(self, outputs, targets):
        targets = targets.cpu().data
        predition = outputs.argmax(1).cpu().data

        for index in range(len(targets)):
            self.class_total[targets[index]] += 1
            self.pre_total[predition[index]] += 1
            if predition[index] == targets[index]:
                self.pre_correct[targets[index]] += 1
        self.predictions.extend(predition.tolist())
        self.labels.extend(targets.tolist())

    def cal_metrics(self, label):
        print(self.pre_total)
        recall = self.pre_correct[label] / self.class_total[label]
        precision = self.pre_correct[label] / self.pre_total[label]
        F1_score = 2 *  (recall * precision) / (precision + recall)
        fpr, tpr, thred = roc_curve(self.labels, self.predictions, pos_label=label)
        auc_score = auc(fpr, tpr)
        return recall, precision, F1_score, auc_score
        
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

# 直方图均衡
def equalizehist(img_path):
    image = cv2.imread(img_path, 0)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    clahe = clahe.apply(image)
    cv2.imwrite('/data16/xiaohui/Project_code/look/equa_img3.jpg', clahe)

# 对比度
def enhance_Contrast(img_path):
    img = Image.open(img_path)
    enhance = ImageEnhance.Contrast(img)
    enhance_img = enhance.enhance(1)
    enhance_img.save('/data16/xiaohui/Project_code/look/enhance.jpg')

# 锐化
def Sharpening(img_path):
    src = cv2.imread(img_path, 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    dst = cv2.filter2D(src, -1, kernel)
    cv2.imwrite('/data16/xiaohui/Project_code/look/Sharpen.tif', dst)

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


