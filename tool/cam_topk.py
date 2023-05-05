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
sys.path.append('./Project_code/')
from model.bilinear import BiCNN

# checkpoint = "checkpoint/MSI_classification_param_best.ckpt"

def cam(path, net, targets, type):
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
    cv2.imwrite(join('./Project_code/cam_topk', type, img_path.split('/')[-1].split('.')[0] + '.jpg'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    # print(save_path)

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD)
        ])
    for pre in ['TP', 'TN', 'FP', 'FN']:
        label_csv = pd.read_csv(join('./Project_code', '{}_list.csv'.format(pre)))
        label_csv = label_csv.values.tolist()
        if not os.path.exists('./Project_code/cam_topk/{}'.format(pre)):
            os.makedirs('./Project_code/cam_topk/{}'.format(pre))
        for id in tqdm(range(10)):
            fold = int(label_csv[id][2])
            model = BiCNN()
            state_dict = torch.load('./Project_code/checkpoint/bilinear/2/pre_{}fold.pth.tar'.format(fold))
            model.load_state_dict(state_dict['state_dict'])
            target_layers = [model.features]
            patient_id = label_csv[id][0].split(',')[0][5:]
            teeth_id = label_csv[id][0].split(',')[1]
            img_path = join('/home/2021/xiaohui/Storage/project_data_crop', patient_id, 'cropped_image{},{}.tif'.format(patient_id, teeth_id))
            cam(img_path, model, target_layers, pre)
