cd /home/2021/xiaohui/Storage/Project_code/tool
#################Test
# CUDA_VISIBLE_DEVICES=6 python validate_csv.py -m res18 res34 res50 res101 next50 dense121 vgg16 --root '/home/2021/xiaohui/Storage/Project_code/new_checkpoint/model_5fold'
# CUDA_VISIBLE_DEVICES=6 python validate_csv.py -m BiCNN_vgg -b vgg16 vgg19 vgg13 --root '/home/2021/xiaohui/Storage/Project_code/new_checkpoint/BiCNN_5fold'
CUDA_VISIBLE_DEVICES=6 python validate_csv.py -m BiCNN_rn -b res18 res34 res50 next50 res101 --root '/home/2021/xiaohui/Storage/Project_code/new_checkpoint/BiCNN_5fold'