cd /home/2021/xiaohui/Storage/Project_code
CUDA_VISIBLE_DEVICES=7 python train.py -m BiCNN_vgg -b vgg16 --pretrain True -g 1 --loss_weight 1.0 1.5