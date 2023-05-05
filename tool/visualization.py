import imp
import os
from os.path import join
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from torchvision.models import resnet50, vgg16_bn
from PIL import Image
import numpy as np
from tqdm import tqdm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision
import cv2
import sys
# sys.path.append('/home/2021/xiaohui/Storage/Project_code/')
from model.bilinear import BiCNN_vgg
import matplotlib.cm as cm


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    for i, image_path in enumerate(image_paths):
        print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image



def cam(path, net, targets, target_class=None):
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
    if target_class is not None:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])
    else:
        grayscale_cam = cam(input_tensor=input_tensor)
    rgb_img = np.float32(rgb_img.resize((224, 224)))/255

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    
    grayscale_cam = np.float32(Image.fromarray(grayscale_cam).resize((224, 224)))
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # save_path = os.path.join(path)
    return grayscale_cam, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(join('/home/2021/xiaohui/Storage/Project_code/look', 'New_Cam.jpg'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiCNN_vgg(backbone='vgg16').to(device)
    # model = resnet50()
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, 2)
    # model = model.to(device)
    # state_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/model_select_5fold/11/pre_1fold.pth.tar')
    state_dict = torch.load('./new_checkpoint/BiCNN_5fold/1/best/5fold_best.pth.tar')
    model.load_state_dict(state_dict['state_dict'])

    transform = T.Compose(
        [   T.Resize(224), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    csv_root = './visualization'
    dir = 'BiCNN_4_Positive'
    for type in ['TP', 'TN', 'FP', 'FN']:
        img_list = []
        csv_file = pd.read_csv(join(csv_root, '{}_list.csv'.format(type)), header=None).values.tolist()
        csv_file = csv_file[1:]
        for img in csv_file:
            img = img[0]
            id = img.split(',')[0][5:]
            teeth = img.split(',')[1]
            path = './project_data_crop_new/{}/cropped_image{},{}.tif'.format(id, id, teeth)
            img_list.append(path)
        images, raw_images = load_images(img_list)
        images = torch.stack(images).to(device)
        # target_layer = [model.layer4]
        target_layer = [model.features[-4]]
        num_fold = int((len(images) / 32) + 1)
        for fold in range(num_fold):
            left = fold * 32
            if fold == num_fold - 1:
                right = len(images)
            else:
                right = (fold * 32) + 32

            images_split = images[left:right]
            raw_images_split = raw_images[left:right]
            img_list_split = img_list[left:right]

            # bp = BackPropagation(model=model)
            # probs, ids = bp.forward(images_split)  # sorted
            # bp.remove_hook()
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
            print(images_split.shape)
            output_dir = './visualization/{}/{}'.format(dir, type)
            if not os.path.exists('./visualization/{}/{}'.format(dir, type)):
                for sub_dir in ['cam', 'gcam', 'gbp']:
                    os.makedirs('./visualization/{}/{}/{}'.format(dir, type, sub_dir))

            for j in tqdm(range(len(images_split))):
                img_name = img_list_split[j].split('/')[-1].split('.')[0]
                cam_map, gradcam = cam(img_list_split[j], model, target_layer, 1)
                gb = gb_model(images_split[j].unsqueeze(0), target_category=None)
                cam_mask = cv2.merge([cam_map, cam_map, cam_map])
                cam_gb = deprocess_image(cam_mask * gb)
                gb = deprocess_image(gb)
                cv2.imwrite('./visualization/{}/{}/{}/{}.png'.format(dir, type, 'cam', img_name), gradcam)
                cv2.imwrite('./visualization/{}/{}/{}/{}.png'.format(dir, type, 'gcam', img_name), cam_gb)
                cv2.imwrite('./visualization/{}/{}/{}/{}.png'.format(dir, type, 'gbp', img_name), gb)
