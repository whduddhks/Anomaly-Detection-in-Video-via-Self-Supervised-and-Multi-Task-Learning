import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3D(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        if 