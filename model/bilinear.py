import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BiCNN(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 2) -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BiCNN, self).__init__()
        model: nn.Module = models.vgg16(pretrained=True)
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

class BCNN_all(nn.Module):
    def __init__(self):
        super(BCNN_all, self).__init__()

        self.features = models.vgg16(pretrained=True).features
        self.features = nn.Sequential(*list(self.features.children())[:-1])

        self.fc = nn.Linear(512*512, 2)

    
    def forward(self, x):
        N = x.size()[0]
        x = self.features(x)
        x = x.view(N, 512, 14 * 14)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (14 * 14)
        x = x.view(N, 512 * 512)
        x = torch.sqrt(x + 1e-5)
        x = F.normalize(x)
        x = self.fc(x)
        return x

class BiCNN_new(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 2) -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BiCNN_new, self).__init__()
        model: nn.Module = models.densenet121(pretrained=True)
        self.features: nn.Module = nn.Sequential(*list(model.features))
        self.classifier: nn.Module = nn.Linear(1024 ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(inputs)               # extract features from pretrained base
        outputs = outputs.view(-1, 1024, 7 ** 2)                    # reshape to batchsize * 512 * 28 ** 2
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))      # bilinear product
        outputs = torch.div(outputs, 7 ** 2)                       # divide by 196 to normalize
        outputs = outputs.view(-1, 1024 ** 2)                        # reshape to batchsize * 512 * 512

        outputs = torch.sign(outputs) * torch.sqrt(torch.abs(outputs) + 1e-5)  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)      # l2 normalization
        outputs = self.classifier(outputs)                          # linear projection
        return outputs

if __name__ == '__main__':
    x = torch.randn((1, 3, 224, 224))
    model = BiCNN()
    print(model)