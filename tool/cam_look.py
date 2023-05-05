import imp
import os
from os.path import join
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from torchvision.models import densenet121
import torchvision
import cv2
import sys

from model.bilinear import BiCNN_vgg16

# checkpoint = "checkpoint/MSI_classification_param_best.ckpt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def cam(path, net, targets):
    img_path = path
    # 构建图像数据集的  batch和原图
    rgb_img = Image.open(img_path).convert("RGB")
    # img_a = np.array(rgb_img)
    input_tensor = transform(rgb_img)
        # torch.from_numpy(img_a).reshape()# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    input_tensor = input_tensor.unsqueeze(0)
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=net, target_layers=targets, use_cuda="0")

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    rgb_img = np.float32(rgb_img.resize((187, 187)))/255

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    
    grayscale_cam = np.float32(Image.fromarray(grayscale_cam).resize((187, 187)))
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # save_path = os.path.join(path)
    cv2.imwrite(join('./Project_code/look', 'New_Cam.jpg'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    # print(save_path)

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD)
        ])
    net = BiCNN_vgg16()
    state_dict = torch.load('./checkpoint/BiCNN_select_5fold/6/4fold.pth.tar')
    net.load_state_dict(state_dict['state_dict'])
    target_layers = [net.features]
    net.eval()
    root = '/home/2021/xiaohui/Storage/project_data_crop/065/cropped_image065,21.tif'
    cam(root, net, target_layers)