import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np


class aotloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, aot_output, target):
        loss = nn.CrossEntropyLoss()
        return loss(target, aot_output)


class miloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mi_output, target):
        loss = nn.CrossEntropyLoss()
        return loss(target, mi_output)

    
class mbploss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mbp_output, target):
        loss = nn.L1Loss(reduction='mean')
        return loss(target, mbp_output)


class mdloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, md_merged_output, target_merged):
        loss = nn.L1Loss(reduction='mean')
        return loss(md_merged_output, target_merged)