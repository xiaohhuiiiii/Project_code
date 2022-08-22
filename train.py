from cmath import inf
from operator import mod
from statistics import mode
import torch
import torchvision
import torchvision.transforms as tf
import torchvision.models as models
from torch.utils.data import DataLoader
from Dataset import  get_dataset
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import config
from utils import AverageMeter, accuracy, ProgressMeter, Metrics
import time
import wandb
import os
from os.path import join
import shutil
import torch.nn as nn
import pandas as pd
from model.densenet import densenet23, densenet41
from transform import get_transform
from focal_loss import Focal_loss
from model.bilinear import BiCNN, BCNN_all, BiCNN_new
import torch.optim.lr_scheduler as lr_scheduler

# os.environ['TORCH_HOME'] = '/data16/xiaohui/torch_model'
torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    # define transforms and param
    train_param = config.train_param
    val_param = config.val_param
    train_transforms = get_transform(train_param)
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    # get datasets
    for fold in range(1, 6):
        label_csv = ('/home/2021/xiaohui/Storage/Project_code/label/Train_fold_{}.csv'.format(str(fold)),
                    '/home/2021/xiaohui/Storage/Project_code/label/Test_fold_{}.csv'.format(str(fold)))
        train_label = pd.read_csv(label_csv[0], header=None)
        train_label = train_label.values.tolist()
        val_label = pd.read_csv(label_csv[1], header=None)
        val_label = val_label.values.tolist()
        train_label = train_label[1:]
        val_label = val_label[1:]
        train_data = get_dataset('/home/2021/xiaohui/Storage/project_data_crop', train_label, train_transforms)
        val_data = get_dataset('/home/2021/xiaohui/Storage/project_data_crop', val_label, val_transforms)
        train_loader = DataLoader(train_data, batch_size=train_param['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=val_param['batch_size'], shuffle=False, num_workers=2)
        # select model
        if train_param['model'] == 'res18':
            if train_param['pretrain'] == True:
                model = models.resnet18(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet18(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif train_param['model'] == 'res34':
            if train_param['pretrain'] == True:
                model = models.resnet34(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet34(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif train_param['model'] == 'res50':
            if train_param['pretrain'] == True:
                model = models.resnet50(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet50(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif train_param['model'] == 'next50':
            if train_param['pretrain'] == True:
                model = models.resnext50_32x4d(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnext50_32x4d(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif train_param['model'] == 'vgg16':
            if train_param['pretrain'] == True:
                model = models.vgg16_bn(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
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
        elif train_param['model'] == 'dense23':
            model = densenet23(num_classes=2)
        elif train_param['model'] == 'dense41':
            model = densenet41(num_classes=2)
        elif train_param['model'] == 'dense121':
            if train_param['pretrain'] == True:
                model = models.densenet121(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.densenet121(pretrained=False)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2)
        elif train_param['model'] == 'dense161':
            if train_param['pretrain'] == True:
                model = models.densenet161(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.densenet161(pretrained=False)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2)
        elif train_param['model'] == 'incept':
            if train_param['pretrain'] == True:
                model = timm.create_model('inception_v3', pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = timm.create_model('inception_v3', pretrained=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 2)
        elif train_param['model'] == 'BCNN_all':
            model = BCNN_all()
        elif train_param['model'] == 'BCNN':
            model = BiCNN()
        elif train_param['model'] == 'BCNN_new':
            model = BiCNN_new()
        else:
            raise Exception('model error')
        model = model.to(device)

        save_path = join('/home/2021/xiaohui/Storage/Project_code/checkpoint', config.project_name, str(train_param['group']))
        optimizer = torch.optim.SGD(model.parameters(), lr=train_param['lr'], weight_decay=train_param['weight_decay'])
        if train_param['schedule'] == 'milestone':
            Scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_param['milestone'], gamma=0.1)
        elif train_param['schedule'] == 'cosine':
            Scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-9)
        if train_param['loss'] == 'CE':
            if train_param['loss_weight'] is not None:
                weight = torch.tensor(train_param['loss_weight']).to(device)
                criterion = torch.nn.CrossEntropyLoss(weight=weight)
            elif train_param['loss_weight'] is None:
                criterion = torch.nn.CrossEntropyLoss()
        elif train_param['loss'] == 'FL':
            criterion = Focal_loss()
        if train_param['pretrain'] == True:
            model_name = 'pre_{}fold.pth.tar'.format(fold)
            best_name = 'pre_{}fold_best.pth.tar'.format(fold)
        else:
            model_name = '{}fold.pth.tar'.format(fold)
            best_name = '{}fold_best.pth.tar'.format(fold)


        # train start
        best_acc = 0

        if config.use_wandb:
            wandb.init(project=config.project_name)
            wandb.config.update(config.train_param)
            if train_param['pretrain']:
                wandb.run.name = 'pre_' + train_param['model'] + '_{}fold'.format(str(fold))
            else:
                wandb.run.name = train_param['model'] + '_{}fold'.format(str(fold))
            wandb.watch(model, criterion, log='all', log_freq=10)

        for epoch in range(train_param['epoch']):
            model.train()
            train(model, train_loader, optimizer, criterion, epoch, train_param)

            acc1 = validate(model, val_loader, criterion, val_param, epoch)

            is_best = acc1 > best_acc
            best_acc = max(acc1, best_acc)
            Scheduler.step()
            if not os.path.exists(join(save_path, )):
                os.makedirs(save_path)
        if config.if_save:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }, join(save_path, model_name))
        
        # 测试
        validate(model, val_loader, criterion, val_param, epoch, mode='test')
        wandb.run.finish()
def train(model, train_loader, optimizer, criterion, epoch, train_param):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix='Epoch: [{}]'.format(epoch)
    )
    num_batches_per_epoch = len(train_loader)
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1 = accuracy(output, target, topk=(1, ))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % train_param['print_freq'] == 0:
            progress.display(i)

            if config.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg,
                    'epoch': epoch
                })

    return losses.avg, top1.avg

def validate(model, val_loader, criterion, val_param, epoch, mode='val'):
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

        if mode == 'val':
            if config.use_wandb:
                wandb.log({
                    'val_loss': losses.avg,
                    'val_acc': top1.avg,
                    'epoch': epoch
                })
        elif mode == 'test':
            if config.use_wandb:
                wandb.log({
                    'test_acc': top1.avg,
                    'recall': recall, 
                    'precision': precision, 
                    'F1_score': F1_score, 
                    'auc_score': auc_score
                })

    return top1.avg

if __name__ == '__main__':
    main()











