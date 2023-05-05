import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BiCNN_vgg(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 2, pretrain=True, backbone='vgg16') -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BiCNN_vgg, self).__init__()
        if backbone == 'vgg16':
            model: nn.Module = models.vgg16(pretrained=pretrain)
        elif backbone == 'vgg16_bn':
            model: nn.Module = models.vgg16_bn(pretrained=pretrain)
        elif backbone == 'vgg13':
            model: nn.Module = models.vgg13(pretrained=pretrain)
        elif backbone == 'vgg13_bn':
            model: nn.Module = models.vgg13_bn(pretrained=pretrain)
        elif backbone == 'vgg19':
            model: nn.Module = models.vgg19(pretrained=pretrain)
        elif backbone == 'vgg19_bn':
            model: nn.Module = models.vgg19_bn(pretrained=pretrain)
        self.features: nn.Module = nn.Sequential(*list(model.features)[:-1])
        self.classifier: nn.Module = nn.Linear(512 ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(inputs)               # extract features from pretrained base
        outputs = outputs.view(-1, 512, 14 ** 2)                    # reshape to batchsize * 512 * 28 ** 2
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))      # bilinear product
        outputs = torch.div(outputs, 14 ** 2)                       # divide by 196 to normalize
        outputs = outputs.view(-1, 512 ** 2)                        # reshape to batchsize * 512 * 512
        outputs = torch.sign(outputs) * torch.sqrt(torch.abs(outputs) + 1e-5)  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)      # l2 normalization
        outputs = self.classifier(outputs)                          # linear projection
        return outputs

class BiCNN_rn(nn.Module):

    def __init__(self, num_classes: int = 2, backbone: str = 'rn50', pretrain=True) -> None:
        super(BiCNN_rn, self).__init__()
        if backbone == 'res50':
            model: nn.Module = models.resnet50(pretrained=pretrain)
            self.HW = 7
            self.C = 2048
        elif backbone == 'res18':
            model: nn.Module = models.resnet18(pretrained=pretrain)
            self.HW = 7
            self.C = 512
        elif backbone == 'res34':
            model: nn.Module = models.resnet34(pretrained=pretrain)
            self.HW = 7
            self.C = 512
        elif backbone == 'next50':
            model: nn.Module = models.resnext50_32x4d(pretrained=pretrain)
            self.HW = 7
            self.C = 2048
        elif backbone == 'res101':
            model: nn.Module = models.resnet101(pretrained=pretrain)
            self.HW = 7
            self.C = 2048
        model.fc = nn.Sequential()
        model.avgpool = nn.Sequential()
        self.features: nn.Module = model
        self.classifier: nn.Module = nn.Linear(self.C ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor = self.features(inputs)              
        outputs = outputs.view(-1, self.C, self.HW ** 2)                   
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))      
        outputs = torch.div(outputs, self.HW ** 2)                      
        outputs = outputs.view(-1, self.C ** 2)                      
        outputs = torch.sign(outputs) * torch.sqrt(torch.abs(outputs) + 1e-5)  
        outputs = nn.functional.normalize(outputs, p=2, dim=1)     
        outputs = self.classifier(outputs)             
        return outputs