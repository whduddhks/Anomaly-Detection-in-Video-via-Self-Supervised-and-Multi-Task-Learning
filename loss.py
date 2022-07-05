from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


class aot_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, aot_output, target):
        loss = nn.CrossEntropyLoss()
        return loss(aot_output, target)


class mi_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mi_output, target):
        return nn.CrossEntropyLoss(mi_output, target)

    
class mbp_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mbp_output, target):
        return torch.mean(torch.abs(target - mbp_output))


class md_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, md_output_y, target_y, md_output_r, target_r):
        return -1