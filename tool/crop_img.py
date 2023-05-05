import imp
import cv2
import numpy as np
import os
from os.path import join
from tqdm import tqdm

root = '/data16/xiaohui/project_data'
dir_list = os.listdir(root)

for dir in tqdm(dir_list):
    img_list = os.listdir(join(root, dir, 'crop2'))
    if not os.path.exists(join('/data16/xiaohui/project_data_crop', dir)):
        os.makedirs(join('/data16/xiaohui/project_data_crop', dir))
    for img in img_list:
        image = cv2.imread(join(root, dir, 'crop2', img), 0)
        image = image[:, :187]
        cv2.imwrite(join('/data16/xiaohui/project_data_crop', dir, img), image)