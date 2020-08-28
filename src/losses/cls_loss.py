import torch
import torch.nn as nn


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss().cuda()

    def forward(self, input, target):
        return self.loss(input, target)
