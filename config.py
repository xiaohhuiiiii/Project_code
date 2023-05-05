train_param = {
    'group': 1, 
    'fold': 5,
    'loss': 'CE', 
    'batch_size': 32,
    'lr': 0.05,
    'weight_decay': 1e-4,
    'pretrain': True,
    'epoch': 100,
    'milestone': [40, 80],
    'print_freq': 10, 
    'model': 'BiCNN_vgg16', 
    'backbone': None, 
    'Sharpeness': 0, 
    'ColorJitter': 0, 
    'equalhist': 0, 
    'loss_weight': None, 
    'schedule': 'milestone'
}

val_param = {
    'batch_size': 32,
    'print_freq': 11,
}

use_wandb = True
if_save = True
project_name = 'BiCNN_minibatch'
entity = 'xiaohuiii'