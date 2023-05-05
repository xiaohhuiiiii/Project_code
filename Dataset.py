import torch
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
from PIL import Image

class get_dataset(Dataset):
    def __init__(self, root, label_list, transforms=None, get_name=False):
        self.root = root
        self.transforms =transforms
        self.label_list = label_list
        self.get_name = get_name
        self.data_list = []
        self.data_name = []
        for data in self.label_list:
            if len(data) == 3:
                if len(str(int(data[0]))) == 1:
                    num = '00' + str(data[0])
                elif len(str(int(data[0]))) == 2:
                    num = '0' + str(data[0])
                else:
                    num = str(int(data[0]))
                image_path = join(self.root, num,
                                'cropped_image{},{}.tif'.format(num, str(int(data[1]))))
                label = int(data[2])
                img_name = 'image{},{}'.format(num, str(int(data[1])))
                self.data_list.append((image_path, label))
                self.data_name.append(img_name)
            elif len(data) == 4:
                if len(str(int(data[1]))) == 1:
                    num = '00' + str(int(data[1]))
                elif len(str(int(data[1]))) == 2:
                    num = '0' + str(int(data[1]))
                else:
                    num = str(int(data[1]))
                image_path = join(self.root, num,
                                'cropped_image{},{}.tif'.format(num, str(int(data[3]))))
                label = int(data[2])
                img_name = 'image{},{}'.format(num, str(int(data[3])))
                self.data_list.append((image_path, label))
                self.data_name.append(img_name)
            else:
                raise Exception('path_error')

    def __getitem__(self, index):
        image = Image.open(self.data_list[index][0]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.data_list[index][1]
        name = self.data_name[index]

        if self.get_name:
            return image, label, name
        else:
            return image, label

    def __len__(self):
        return len(self.data_list)

class get_pre_dataset(Dataset):
    def __init__(self, root, label_list, transforms=None, type=0):
        self.root = root
        self.transforms =transforms
        self.label_list = label_list
        self.data_list = []
        for data in self.label_list:
            if int(data[2]) == type:
                if len(data) == 3:
                    if len(str(data[0])) == 1:
                        num = '00' + str(data[0])
                    elif len(str(data[0])) == 2:
                        num = '0' + str(data[0])
                    else:
                        num = str(data[0])
                    image_path = join(self.root, num,
                                    'cropped_image{},{}.tif'.format(num, str(data[1])))
                    label = data[2]
                    self.data_list.append((image_path, label))
                elif len(data) == 4:
                    if len(str(int(data[1]))) == 1:
                        num = '00' + str(int(data[1]))
                    elif len(str(int(data[1]))) == 2:
                        num = '0' + str(int(data[1]))
                    else:
                        num = str(int(data[1]))
                    image_path = join(self.root, num,
                                    'cropped_image{},{}.tif'.format(num, str(int(data[3]))))
                    label = int(data[2])
                    self.data_list.append((image_path, label))
                else:
                    raise Exception('path_error')

    def __getitem__(self, index):
        image = Image.open(self.data_list[index][0]).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.data_list[index][1]
        path = self.data_list[index][0]
        return image, label, path

    def __len__(self):
        return len(self.data_list)

