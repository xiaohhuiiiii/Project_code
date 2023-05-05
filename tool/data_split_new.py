'''
    用于数据的五折验证划分
'''
from sklearn.model_selection import  StratifiedGroupKFold
import pandas as pd
import random
'''yzj  五折交叉验证测试'''

random.seed(1998)
if __name__ == '__main__':
    csv_file ='/home/2021/xiaohui/Storage/Project_code/label_5fold/Train.csv'
    frame = pd.read_csv(csv_file, encoding='utf-8', header=None)
    # X = frame[0]
    # y = frame[1]
    # img_name = frame[2]
    # img_folder = frame[3]
    X = frame[1]
    y = frame[2]
    groups = frame[1]
    img_name = frame[3]
    print(X.shape,y.shape)   #---> (3705,) (3705,)
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True)
    i=0
    for train_index,test_index in skf.split(X,y, groups):
        #print('i=:',i,'train_index:',train_index.shape,'test_index:',test_index.shape)
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        img_train,img_test = img_name[train_index],img_name[test_index]
        i+=1
        train = pd.concat([X_train,y_train,img_train],axis=1)
        test =pd.concat([X_test,y_test,img_test],axis=1)
        #print(train)
        train.to_csv('/home/2021/xiaohui/Storage/Project_code/label_10fold/Train_fold_{}.csv'.format(str(i)))
        test.to_csv('/home/2021/xiaohui/Storage/Project_code/label_10fold/Val_fold_{}.csv'.format(str(i)))