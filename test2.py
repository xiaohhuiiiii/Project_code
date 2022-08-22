from builtins import Exception
from doctest import Example
import torchvision.models as models
import pandas as pd
import numpy as np

train = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Train_fold_1.csv')
train = train.values.tolist()
train = np.array(train)
test = pd.read_csv('/home/2021/xiaohui/Storage/Project_code/label/Test_fold_1.csv')
test = test.values.tolist()
test = np.array(test)
count0 = 0
count1 = 0
for row1 in train:
    if row1[2] == 0:
        count0 += 1
    elif row1[2] == 1:
        count1 += 1
    else:
        raise Exception('error')
for row2 in test:
    if row2[2] == 0:
        count0 += 1
    elif row2[2] == 1:
        count1 += 1
    else:
        raise Exception('error')
print(count0)
print(count1)