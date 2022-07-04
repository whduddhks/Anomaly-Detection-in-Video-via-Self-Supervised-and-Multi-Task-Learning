import torch
import torch.nn as nn
import torch.nn.functional as F

class aothead(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
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
    def __init__(self, in_channel):
        super().__init__()
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


class mdphead(nn.Module):
    def __init__(self, in_channel, width, depth):
        super().__init__()
        if width == 'narrow':
            channel = [16, 32]
        else:
            channel = [32, 64]



class mdhead(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.md = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=(3, 3), padding='same', stride=1),
            nn.MaxPool2d(2),
        )
        self.fc_s_r = nn.Linear(32*4*4, 1000)
        self.fc_d_r = nn.Linear(32*2*2, 1000)
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
    t = mdhead(64)

    r = t(rand, 'deep')
    print(r[0].shape, r[1].shape)