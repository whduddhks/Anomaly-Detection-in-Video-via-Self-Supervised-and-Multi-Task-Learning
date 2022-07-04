from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class conv3D(nn.Module):
    def __init__(self, width, depth):
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
         
        if depth == 'deep':
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

        self.maxpool_1 = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.maxpool_4 = nn.MaxPool3d((4, 2, 2), stride=(1, 2, 2))
        self.maxpool_5 = nn.MaxPool3d((5, 2, 2), stride=(1, 2, 2))
        
    def forward(self, x, depth):
        out = self.conv3D1(x)
        if depth == 'deep':     
            out = self.conv3D1_1(out)
        out = self.maxpool_1(out)

        out = self.conv3D2(out)
        if depth == 'deep':     
            out = self.conv3D2_1(out)
        out = self.maxpool_1(out)

        out = self.conv3D3(out)
        if depth == 'deep':
            out = self.maxpool_1(out)     
            out = self.conv3D3_1(out)

        d = out.shape[2]
        if d == 1:
            out = self.maxpool_1(out)
        elif d == 4:
            out = self.maxpool_4(out)
        else:
            out = self.maxpool_5(out)
    
        return out


if __name__ == "__main__":
    rand = torch.ones([8, 3, 1, 64, 64])
    t = conv3D('narrow', 'deep')

    r = t(rand, 'dep')
    print(r.shape)