from cv2 import trace
import torch
import torchvision.transforms as T
import albumentations
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import manual_Compose

def get_transform(param):
    labels = []
    transforms = [T.Resize(224)]
    labels.append(0)
    if param['Sharpeness'] != 0:
        transforms.append(T.RandomAdjustSharpness(param['Sharpeness'], p=0.5))
        labels.append(0)
    if param['ColorJitter'] != 0:
        transforms.append(T.ColorJitter(contrast=param['ColorJitter']))
        labels.append(0)
    if param['equalhist'] != 0:
        transforms.append(albumentations.CLAHE(clip_limit=param['equalhist'], tile_grid_size=(4, 4), p=1.0))
        labels.append(1)
    transforms.append(T.RandomHorizontalFlip())
    labels.append(0)
    transforms.append(T.RandomRotation(45))
    labels.append(0)
    transforms.append(T.ToTensor())
    labels.append(0)
    transforms.append(T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    labels.append(0)
    Compose = manual_Compose(transforms, labels)
    return Compose

