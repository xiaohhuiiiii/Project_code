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
sys.path.append('/home/2021/xiaohui/Storage/Project_code/')
from model.bilinear import BiCNN_vgg
import matplotlib.cm as cm


def scale_cam_image(img, max_ele, target_size=None):
    img = img - np.min(img)
    img = img / (1e-7 + max_ele)
    if target_size is not None:
        img = cv2.resize(img, target_size)
    result = np.float32(img)

    return result

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



def cam(path, net, targets, transform, target_class=None):
    img_path = path
    # 构建图像数据集的  batch和原图
    rgb_img = Image.open(img_path).convert("RGB")
    # img_a = np.array(rgb_img)
    input_tensor = transform(rgb_img)
        # torch.from_numpy(img_a).reshape()# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    input_tensor = input_tensor.unsqueeze(0)
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=net, target_layers=targets, use_cuda="0", no_norm=True)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    if target_class is not None:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(int(target_class))])
    else:
        grayscale_cam = cam(input_tensor=input_tensor)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    
    grayscale_cam = np.float32(Image.fromarray(grayscale_cam).resize((224, 224)))
    # save_path = os.path.join(path)
    return grayscale_cam
    # cv2.imwrite(join('/home/2021/xiaohui/Storage/Project_code/look', 'New_Cam.jpg'), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

def cal_all_activations(csv_path, net, targets, transform, target_class=None):
    all_activations_mean = {}
    for type in ['TP', 'TN', 'FP', 'FN']:
        activations = []
        img_list = []
        csv_file = pd.read_csv(join(csv_path, '{}_list.csv'.format(type)), header=None).values.tolist()
        csv_file = csv_file[1:]
        for img in csv_file:
            img = img[0]
            id = img.split(',')[0][5:]
            teeth = img.split(',')[1]
            path = '/home/2021/xiaohui/Storage/project_data_crop_new/{}/cropped_image{},{}.tif'.format(id, id, teeth)
            img_list.append(path)
        images, raw_images = load_images(img_list)
        images = torch.stack(images).to(device)
        # target_layer = [model.layer4]
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

            for j in tqdm(range(len(images_split))):
                cam_map = cam(img_list_split[j], net, targets, transform, target_class)
                norm_map = scale_cam_image(cam_map, np.max(cam_map))
                norm_map *= 255
                all_elem = cam_map[norm_map > 160]
                if all_elem.sum() == 0:
                    continue
                mean_elem = all_elem.sum() / len(all_elem)

                activations.append(mean_elem)
        activations = np.array(activations)
        activations = activations.sum() / len(activations)
        if type == 'TN':
                    print(activations)
                    
        all_activations_mean[type] = activations
    return all_activations_mean

def show_img(path0, path1, net, targets, transform, target_class=None):
    img0 = Image.open(path0).convert("RGB")
    img1 = Image.open(path1).convert("RGB")
    cam_map0 = cam(path0, net, targets, transform, target_class)
    cam_map1 = cam(path1, net, targets, transform, target_class)
    if np.max(cam_map1) > np.max(cam_map0):
        maxe = np.max(cam_map1)
    else:
        maxe = np.max(cam_map0)
    norm_map1 = scale_cam_image(cam_map1, maxe)
    norm_map0 = scale_cam_image(cam_map0, maxe)
    heatmap0 = cv2.applyColorMap(np.uint8(255 * norm_map0), cv2.COLORMAP_JET)
    heatmap1 = cv2.applyColorMap(np.uint8(255 * norm_map1), cv2.COLORMAP_JET)
    heatmap0 = np.float32(heatmap0) / 255
    heatmap1 = np.float32(heatmap1) / 255
    rgb_img0 = np.float32(img0.resize((224, 224)))/255
    rgb_img1 = np.float32(img1.resize((224, 224)))/255


    cam0 = heatmap0 + rgb_img0
    cam1 = heatmap1 + rgb_img1
    cam0 = cam0 / np.max(cam0)
    cam1 = cam1 / np.max(cam1)
    cam0 = np.uint8(255 * cam0)
    cam1 = np.uint8(255 * cam1)
    img_name0 = path0.split('/')[-1].split('.')[0]
    img_name1 = path1.split('/')[-1].split('.')[0]
    root = '/home/2021/xiaohui/Storage/Project_code/map_for_T'
    for i in range(10):
        if not os.path.exists(join(root, str(i))):
            os.makedirs(join(root, str(i)))
            dir = str(i)
            break
    cv2.imwrite(join(root, dir, img_name0 + '.png'), cam0)
    cv2.imwrite(join(root, dir, img_name1 + '.png'), cam1)
    


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose(
        [   T.Resize(224), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ################################cal
    all_activation = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for i in range(1,6):
        model = BiCNN_vgg(backbone='vgg16').to(device)
        targets = [model.features[-4]]
        state_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/new_checkpoint/BiCNN_5fold/1/best/{}fold_best.pth.tar'.format(i))
        model.load_state_dict(state_dict['state_dict'])
        csv_root = '/home/2021/xiaohui/Storage/Project_code/visualization'
        activation = cal_all_activations(csv_root, model, targets, transform, 0)
        for type in ['TP', 'TN', 'FP', 'FN']:
            all_activation[type] += activation[type]
        
        print(all_activation)
    ################################show
    # path1 = join('/home/2021/xiaohui/Storage/project_data_crop_new/{}/{}.tif'.format('720', 'cropped_image720,22'))
    # path0 = join('/home/2021/xiaohui/Storage/project_data_crop_new/{}/{}.tif'.format('265', 'cropped_image265,11'))
    # show_img(path0, path1, model, targets, transform, 1)
    