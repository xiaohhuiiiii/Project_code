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
from utils import AverageMeter, accuracy, ProgressMeter, Metrics
import time
import wandb
import os
from os.path import join
import argparse
import torch.nn as nn
import pandas as pd
from model.densenet import densenet23, densenet41
from transform import get_transform
from focal_loss import Focal_loss
from model.bilinear import BiCNN_rn, BiCNN_vgg
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np


torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    set_seed(1998)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default=5, type=int, help='k fold')
    parser.add_argument('-b', '--backbone', default=None, type=str, help='select backbone')
    parser.add_argument('-m', '--model', default='BiCNN_rn', type=str, help='select model')
    parser.add_argument('-g', '--group', default=1, type=int)
    parser.add_argument('--pretrain', default=False, type=bool, help='If use pretrained model')
    parser.add_argument('--loss_weight', default=None, type=float, nargs='+', help='If use loss_weight')
    ###################################
    parser.add_argument('--project_name', type=str, default='new_BCNNvgg', help='The name of wandb project')
    parser.add_argument('--use_wandb', default=False, type=bool, help='If use wandb')
    parser.add_argument('--if_save', default=True, type=bool, help='If save model')
    ####################################
    parser.add_argument('--cosstep', default='epoch', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epoch', default=80, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--milestone', default=[30, 60])
    parser.add_argument('--schedule', default='milestone', type=str, help='select learning_rate scheduler')
    #####################################
    args = parser.parse_args()

    num_fold = args.fold
    train_transforms = get_transform(args)
    val_transforms = tf.Compose([tf.Resize(224), 
                                tf.ToTensor(), 
                                tf.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    # get datasets
    test_csv = './label_{}fold/Test.csv'.format(num_fold)
    test_label = pd.read_csv(test_csv).values.tolist()
    test_label = test_label[1:]
    test_data = get_dataset('data_root', test_label, val_transforms) # data_root -> where data is
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    for fold in range(1, num_fold + 1):
        label_csv = ('./label_{}fold/Train_fold_{}.csv'.format(str(num_fold), str(fold)),
                    './label_{}fold/Val_fold_{}.csv'.format(str(num_fold), str(fold)))
        train_label = pd.read_csv(label_csv[0], header=None)
        train_label = train_label.values.tolist()
        val_label = pd.read_csv(label_csv[1], header=None)
        val_label = val_label.values.tolist()
        train_label = train_label[1:]
        val_label = val_label[1:]
        train_data = get_dataset('data_root', train_label, train_transforms)
        val_data = get_dataset('data_root', val_label, val_transforms)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # select model
        if args.model == 'res18':
            if bool(args.pretrain) == True:
                model = models.resnet18(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet18(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif args.model == 'res34':
            if bool(args.pretrain) == True:
                model = models.resnet34(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet34(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif args.model == 'res50':
            if bool(args.pretrain) == True:
                model = models.resnet50(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet50(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif args.model == 'res101':
            if bool(args.pretrain) == True:
                model = models.resnet101(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnet101(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif args.model == 'next50':
            if bool(args.pretrain) == True:
                model = models.resnext50_32x4d(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.resnext50_32x4d(pretrained=False)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, 2)
        elif args.model == 'vgg16':
            if bool(args.pretrain) == True:
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
        elif args.model == 'dense121':
            if bool(args.pretrain) == True:
                model = models.densenet121(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.densenet121(pretrained=False)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2)
        elif args.model == 'dense161':
            if bool(args.pretrain) == True:
                model = models.densenet161(pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = models.densenet161(pretrained=False)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2)
        elif args.model == 'incept':
            if bool(args.pretrain) == True:
                model = timm.create_model('inception_v3', pretrained=True)
                # for param in model.parameters():
                #     param.requires_grad = False
            else:
                model = timm.create_model('inception_v3', pretrained=False)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, 2)
        elif args.model == 'BiCNN_vgg':
            model = BiCNN_vgg(pretrain=True, backbone=args.backbone)
        elif args.model == 'BiCNN_rn':
            model = BiCNN_rn(backbone=args.backbone, pretrain=bool(args.pretrain))
        else:
            raise Exception('model error')
        model = model.to(device)

        save_path = join('./new_checkpoint', args.project_name, str(args.group))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.schedule == 'milestone':
            Scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.1)
        elif args.schedule == 'cosine':
            Scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-9)
        if args.loss_weight is not None:
            weight = torch.tensor(args.loss_weight).to(device)
            criterion = torch.nn.CrossEntropyLoss(weight=weight)
        elif args.loss_weight is None:
            criterion = torch.nn.CrossEntropyLoss()
        model_name = '{}fold.pth.tar'.format(fold)
        best_name = '{}fold_best.pth.tar'.format(fold)


        # train start
        best_acc = 0
        if args.use_wandb:
            if not os.path.exists('./wandb/{}'.format(args.project_name)):
                os.makedirs('./wandb/{}'.format(args.project_name))
        if args.use_wandb:
            wandb.init(project=args.project_name, 
                        entity=args.entity, 
                        dir='./wandb/{}'.format(args.project_name))
            wandb.config.update(args)
            if bool(args.pretrain):
                wandb.run.name = 'pre_' + args.model + '_{}fold'.format(str(fold))
            else:
                wandb.run.name = args.model + '_{}fold'.format(str(fold))
            wandb.watch(model, criterion, log='all', log_freq=10)

        for epoch in range(args.epoch):
            model.train()
            train(model, train_loader, optimizer, criterion, epoch, args)

            acc1 = validate(model, val_loader, criterion, args, epoch)

            is_best = acc1 > best_acc
            best_acc = max(acc1, best_acc)
            Scheduler.step()
            if not os.path.exists(save_path):
                os.makedirs(join(save_path, 'last'))
                os.makedirs(join(save_path, 'best'))
            if args.if_save and is_best:
                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc
                    }, join(save_path, 'best', best_name))
        if args.if_save:
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }, join(save_path, 'last', model_name))
        
        # 测试
        validate(model, test_loader, criterion, args, epoch, mode='test')
        if args.use_wandb:
            wandb.run.finish()
def train(model, train_loader, optimizer, criterion, epoch, args):
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

        if i % args.print_freq == 0:
            progress.display(i)

            if args.use_wandb:
                wandb.log({
                    'training_loss': losses.avg,
                    'training_acc': top1.avg,
                    'epoch': epoch
                })

    return losses.avg, top1.avg

def validate(model, val_loader, criterion, args, epoch, mode='val'):
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

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc {top1.avg:.3f}'.format(top1=top1))
        if mode == 'test':
            recall, specificity, precision, F1_score, auc_score = metrics.cal_metrics()
            print('sensitivity: {}   specificity: {}   precision: {}   F1_score: {}   auc: {}'.format(str(recall), str(specificity), str(precision), str(F1_score), str(auc_score)))

        if mode == 'val':
            if args.use_wandb:
                wandb.log({
                    'val_loss': losses.avg,
                    'val_acc': top1.avg,
                    'epoch': epoch
                })
        elif mode == 'test':
            if args.use_wandb:
                wandb.log({
                    'test_acc': top1.avg,
                    'sensitivity': recall, 
                    'specificity': specificity, 
                    'precision': precision, 
                    'F1_score': F1_score, 
                    'auc_score': auc_score
                })

    return top1.avg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    main()







