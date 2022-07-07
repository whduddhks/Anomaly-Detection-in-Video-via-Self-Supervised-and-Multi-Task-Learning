import torch
import cv2
import numpy as np
import torch.nn as nn


def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
            

def img_crop(clips, pred):
    crop_img = []
    for p in pred[0]:
        crop_flow_img = []
        for c in clips:
            c = c.numpy()
            p = p.int()
            crop_flow = cv2.resize(c[p[1]:p[3], p[0]:p[2]], (64, 64)).astype('float32')
            crop_flow /= 255
            crop_flow_img.append(crop_flow)
        crop_img.append(crop_flow_img)
    return crop_img