import torch
import torch.nn as nn
import numpy
from PIL import Image
from torchvision import transforms as TF
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/2021/xiaohui/Storage/Project_code')
from model.bilinear import BiCNN_vgg16

class Guided_backprop():
    def __init__(self, model):
        self.model = model
        self.image_reconstruction = None
        self.activation_maps = []
        self.model.eval()
        self.register_hooks()
    
    def register_hooks(self):
        def first_layer_hook_fn(model, grad_in, grad_out):
            self.image_reconstruction = grad_in[0]
        
        def forward_hook_fn(module, input, output):
            self.activation_maps.append(output)
        
        def backward_hook_fn(module, grad_in, grad_out):
            grad = self.activation_maps.pop()
            grad[grad > 0] = 1
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            new_grad_in = positive_grad_out * grad
            return (new_grad_in,)
    
        modules = list(self.model.features.named_children())

        for name, module in modules:
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
        
        first_layer = modules[0][1]
        first_layer.register_backward_hook(first_layer_hook_fn)

    def visualize(self, input_image, target_class=None):
        model_output = self.model(input_image)
        self.model.zero_grad()
        pred_class = model_output.argmax().item()
        grad_target_map = torch.zeros(model_output.shape, dtype=torch.float)

        if target_class is not None:
            grad_target_map[0][target_class] = 1
        else:
            grad_target_map[0][pred_class] = 1

        model_output.backward(grad_target_map)

        result = self.image_reconstruction.data[0].permute(1, 2, 0)
        return result.numpy()

def normalize(image):
    norm = (image - image.mean()) / image.std()
    norm = norm * 0.1
    norm = norm + 0.5
    norm[norm <= 0.5] = 0
    norm = norm.clip(0, 1)
    return norm

if __name__ == '__main__':
    path = '/home/2021/xiaohui/Storage/project_data_crop_new/002/cropped_image002,22.tif'
    image = Image.open(path).convert('RGB')

    transforms = TF.Compose([
        TF.Resize(224), 
        TF.ToTensor(), 
        TF.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    tensor = transforms(image).unsqueeze(0).requires_grad_()

    model = BiCNN_vgg16(pretrain=False)
    state_dict = torch.load('/home/2021/xiaohui/Storage/Project_code/checkpoint/BiCNN_select_5fold/6/pre_2fold.pth.tar')
    model.load_state_dict(state_dict['state_dict'])
    guided_bp = Guided_backprop(model)
    result = guided_bp.visualize(tensor, None)
    
    result = normalize(result)
    result *= 255
    cv2.imwrite('/home/2021/xiaohui/Storage/Project_code/GBP.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

