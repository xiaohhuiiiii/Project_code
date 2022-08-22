import os
from os.path import join
import shutil

def move_image():
    root = '/data/xiaohui/唇侧骨壁AI/0 原始数据 ai_images_1_1036(important)'
    to_path = '/data/xiaohui/唇侧骨壁data'
    dir_list = os.listdir(root)
    for dir in dir_list:
        img_list = os.listdir(join(root, dir))
        for img in img_list:
            if img.startswith('cropped_image_a'):
                shutil.move(join(root, dir, img), join(to_path, dir, 'crop1', img[:15+len(dir)+3] + '.tif'))
            print(img)
        img_list = os.listdir(join(root, dir))
        for img in img_list:
            if img.startswith('cropped_image'):
                shutil.move(join(root, dir, img), join(to_path, dir, 'crop2', img[:13+len(dir)+3] + '.tif'))
            print(img)
        img_list = os.listdir(join(root, dir))
        for img in img_list:
            shutil.move(join(root, dir, img), join(to_path, dir, 'ori', img[:len(dir)+3] + '.tif'))
            print(img)


if __name__ == '__main__':
    root = '/data/xiaohui/唇侧骨壁data'
    dir_list = os.listdir(root)
    for dir in dir_list:
        subdir_list = os.listdir(join(root, dir))
        for subdir in subdir_list:
            num = len(os.listdir(join(root, dir, subdir)))
            if num != 4 and num != 0:
                print(join(root, dir, subdir))