import torch
import torch.nn as nn
import torch.nn.functional as F

class aothead(nn.Module):
    def __init__(self, width):
        super().__init__()
        in_channel = 32 if width == 'narrow' else 64
        self.aot = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3), padding='same', stride=1),
            nn.MaxPool2d(2),
        )
        self.fc_s = nn.Linear(32*4*4, 2)
        self.fc_d = nn.Linear(32*2*2, 2)

    def forward(self, x, depth):
        out = self.aot(x)
        out = torch.flatten(out, 1)
        return self.fc_s(out) if depth == 'shallow' else self.fc_d(out)


class mihead(nn.Module):
    def __init__(self, width):
        super().__init__()
        in_channel = 32 if width == 'narrow' else 64
        self.mi = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3), padding='same', stride=1),
            nn.MaxPool2d(2),
        )
        self.fc_s = nn.Linear(32*4*4, 2)
        self.fc_d = nn.Linear(32*2*2, 2)

    def forward(self, x, depth):
        out = self.mi(x)
        out = torch.flatten(out, 1)
        return self.fc_s(out) if depth == 'shallow' else self.fc_d(out)


class mbphead(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width == 'narrow':
            channel = [32, 16]
        else:
            channel = [64, 32]
        
        self.conv2D1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channel[0], channel[0], kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )
        self.conv2D2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channel[0], channel[1], kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2D3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channel[1], 3, kernel_size=(3, 3), padding='same', stride=1),
        )
        self.conv2D1_1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[0], kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(channel[0]),
            nn.ReLU(inplace=True)
        )
        self.conv2D2_1 = nn.Sequential(
            nn.Conv2d(channel[1], channel[1], kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2D3_0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(channel[1], channel[1], kernel_size=(3, 3), padding='same', stride=1),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, depth):
        out = self.conv2D1(x)
        if depth == 'deep':     
            out = self.conv2D1_1(out)

        out = self.conv2D2(out)
        if depth == 'deep':     
            out = self.conv2D2_1(out)

        if depth == 'deep':
            out = self.conv2D3_0(out)
        out = self.conv2D3(out)

        return out


class mdhead(nn.Module):
    def __init__(self, width):
        super().__init__()
        in_channel = 32 if width == 'narrow' else 64
        self.md = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3), padding='same', stride=1),
            nn.MaxPool2d(2),
        )

        # ResNet output
        self.fc_s_r = nn.Linear(32*4*4, 1000)
        self.fc_d_r = nn.Linear(32*2*2, 1000)

        # yolo output
        self.fc_s_y = nn.Linear(32*4*4, 80)
        self.fc_d_y = nn.Linear(32*2*2, 80)

    def forward(self, x, depth):
        out = self.md(x)
        out = torch.flatten(out, 1)
        out_r = self.fc_s_r(out) if depth == 'shallow' else self.fc_d_r(out)
        out_y = self.fc_s_y(out) if depth == 'shallow' else self.fc_d_y(out)
        return out_r, out_y


if __name__ == "__main__":
    rand = torch.ones([8, 64, 4, 4])
    a = aothead(64)
    m = mihead(64)
    p = mbphead('wide')
    t = mdhead(64)

    r_1 = a(rand, 'deep')
    r_2 = m(rand, 'deep')
    r_3 = p(rand, 'deep')
    r_4 = t(rand, 'deep')
    print(r_1.shape)
    print(r_2.shape)
    print(r_3.shape)
    print(r_4[0].shape, r_4[1].shape)