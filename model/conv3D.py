import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3D(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width == 'narrow':
            channel = [16, 32]
        else:
            channel = [32, 64]
        self.conv3D1 = nn.Sequential(
            nn.Conv3d(3, channel[0], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(inplace=True)
        )

        self.conv3D2 = nn.Sequential(
            nn.Conv3d(channel[0], channel[1], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(inplace=True)
        )

        self.conv3D3 = nn.Sequential(
            nn.Conv3d(channel[1], channel[1], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv3D1_1 = nn.Sequential(
            nn.Conv3d(channel[0], channel[0], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[0]),
            nn.ReLU(inplace=True)
        )
        self.conv3D2_1 = nn.Sequential(
            nn.Conv3d(channel[1], channel[1], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(inplace=True)
        )
        self.conv3D3_1 = nn.Sequential(
            nn.Conv3d(channel[1], channel[1], kernel_size=(3, 3, 3), padding='same', stride=1),
            nn.BatchNorm3d(channel[1]),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        
    def forward(self, x, depth):
        out = self.conv3D1(x)
        if depth == 'deep':     
            out = self.conv3D1_1(out)
        out = self.maxpool(out)

        out = self.conv3D2(out)
        if depth == 'deep':     
            out = self.conv3D2_1(out)
        out = self.maxpool(out)

        out = self.conv3D3(out)
        if depth == 'deep':
            out = self.maxpool(out)     
            out = self.conv3D3_1(out)

        d = out.shape[2]
        out = F.max_pool3d(out, (d, 2, 2), stride=(1, 2, 2))
    
        return out


if __name__ == "__main__":
    rand = torch.ones([8, 3, 4, 64, 64])
    t = conv3D('narrow')
    print(t)

    r = t(rand, 'shallow')
    print(r.shape)