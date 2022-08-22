import pandas as pd
import torch
import torch.nn as nn
import time
import timm
from torchvision.models import densenet121
from utils import AverageMeter, ProgressMeter, Metrics, accuracy
from Dataset import get_dataset
import torchvision.transforms as tf
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
import config

device = torch.device('cuda')
def validate(model, val_loader, criterion, val_param, mode='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    if mode == 'test':
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Test: ')
    else:
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1],
            prefix='Validate: ')
    metrics = Metrics()

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            metrics.store(output, target)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % val_param['print_freq'] == 0:
                progress.display(i)

        print(' * Acc {top1.avg:.3f}'
              .format(top1=top1))
        if mode == 'test':
            recall, precision, F1_score, auc_score = metrics.cal_metrics(1)
            print('recall: {}   precision: {}   F1_score: {}   auc: {}'.format(str(recall), str(precision), str(F1_score), str(auc_score)))


    return top1.avg

if __name__ == '__main__':
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    test_label = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Test.csv', header=None)
    test_label = test_label.values.tolist()
    test_label = test_label[1:]
    test_data = get_dataset('/home/2021/xiaohui/Storage/project_data_crop', test_label, val_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)
    model = densenet121(pretrained=False)
    in_features =  model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    stat_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/model_select_pre/7/1fold.pth.tar')
    model.load_state_dict(stat_dict['state_dict'])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    val_param = config.val_param
    validate(model, test_loader, criterion, val_param, mode='test')