from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_loss(nn.Module):
    def __init__(self, gamma=1, num_classes=2):
        super(Focal_loss, self).__init__()
        # if isinstance(alpha, list):
        #     assert len(alpha) == num_classes
        #     self.alpha = torch.Tensor(alpha)
        # else:
        #     assert alpha < 1
        #     self.alpha = torch.zeros(num_classes)
        #     self.alpha[0] += alpha
        #     self.alpha[1:] += (1 - alpha)
        self.gamma = gamma
    
    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        # self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))

        # self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        # loss = torch.mul(self.alpha, loss.t())
        loss = loss.mean()
        return loss