import torch
import torch.nn as nn

class Shift_channel_mix(nn.Module):
    def __init__(self, shift_size=1):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        x1 = torch.roll(x1, self.shift_size, dims=2)
        x2 = torch.roll(x2, -self.shift_size, dims=2)
        x3 = torch.roll(x3, self.shift_size, dims=3)
        x4 = torch.roll(x4, -self.shift_size, dims=3)
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

def autopad(k, p=None, d=1):

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PSConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.shift_size = 1

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))

        x1 = torch.roll(yw0, self.shift_size, dims=2)
        x2 = torch.roll(yw1, -self.shift_size, dims=2)
        x3 = torch.roll(yh0, self.shift_size, dims=3)
        x4 = torch.roll(yh1, -self.shift_size, dims=3)

        out = torch.cat([x1, x2, x3, x4], 1)
        return self.cat(out)

