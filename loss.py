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

    def forward(self, md_yolo, md_res, target_yolo, target_res):
        softmax = nn.Softmax(dim=1)
        loss = nn.L1Loss(reduction='mean')
        md_yolo = softmax(md_yolo)
        md_res = softmax(md_res)
        target_yolo = softmax(target_yolo)
        target_res = softmax(target_res)

        md_merged_output = torch.cat((md_yolo, md_res), 1)
        target_merged = torch.cat((target_yolo, target_res), 1)
        return loss(md_merged_output, target_merged)


if __name__ == "__main__":
    a = torch.randn([10, 80])
    b = torch.randn([10, 1000])
    c = torch.randn([10, 80])
    d = torch.randn([10, 1000])
    loss = mdloss()
    print(loss(a, b, c, d))