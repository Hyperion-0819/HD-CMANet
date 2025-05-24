import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        x = x - torch.amax(x, dim=(2, 3), keepdim=True)
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        epsilon = 1e-7
        return x / (x_exp_pool + epsilon)


class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CLAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.ca= channel_att(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.ca(x)
        g = self.gate(x[:, :1].clone())
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return x * w * g