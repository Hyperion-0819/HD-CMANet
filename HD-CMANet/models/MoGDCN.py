import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoding_Block(nn.Module):
    def __init__(self, c_in):
        super(Encoding_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.act = nn.PReLU()

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        down = self.act(self.conv5(f_e))
        return f_e, down

class Encoding_Block_End(nn.Module):
    def __init__(self, c_in=64):
        super(Encoding_Block_End, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.act = nn.PReLU()

    def forward(self, input):
        out1 = self.act(self.conv1(input))
        out2 = self.act(self.conv2(out1))
        out3 = self.act(self.conv3(out2))
        f_e = self.conv4(out3)
        return f_e

class Decoding_Block(nn.Module):
    def __init__(self, c_in):
        super(Decoding_Block, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)
        self.up = nn.ConvTranspose2d(c_in, 128, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.act = nn.PReLU()

    def forward(self, input, map):
        up = self.up(input)
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))
        out3 = self.conv3(out2)
        return out3

class Feature_Decoding_End(nn.Module):
    def __init__(self, c_out):
        super(Feature_Decoding_End, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=c_out, kernel_size=3, padding=1)
        self.up = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act = nn.PReLU()

    def forward(self, input, map):
        up = self.up(input)
        # 调整 up 或 map 的尺寸以匹配
        if up.shape[2:] != map.shape[2:]:
            # 假设 map 是较小的那个，可以对其进行上采样
            map = F.interpolate(map, size=up.shape[2:], mode='bilinear', align_corners=True)
        cat = torch.cat((up, map), 1)
        cat = self.act(self.conv0(cat))
        out1 = self.act(self.conv1(cat))
        out2 = self.act(self.conv2(out1))
        out3 = self.conv3(out2)
        return out3

class Unet_Spatial(nn.Module):
    def __init__(self, cin):
        super(Unet_Spatial, self).__init__()
        self.Encoding_block1 = Encoding_Block(64)
        self.Encoding_block2 = Encoding_Block(64)
        self.Encoding_block3 = Encoding_Block(64)
        self.Encoding_block4 = Encoding_Block(64)
        self.Encoding_block_end = Encoding_Block_End(64)
        self.Decoding_block1 = Decoding_Block(128)
        self.Decoding_block2 = Decoding_Block(512)
        self.Decoding_block3 = Decoding_Block(512)
        self.Decoding_block_End = Feature_Decoding_End(cin)
        self.acti = nn.PReLU()

    def forward(self, x):
        sz = x.shape
        encode0, down0 = self.Encoding_block1(x)
        encode1, down1 = self.Encoding_block2(down0)
        encode2, down2 = self.Encoding_block3(down1)
        encode3, down3 = self.Encoding_block4(down2)
        media_end = self.Encoding_block_end(down3)
        decode3 = self.Decoding_block1(media_end, encode3)
        decode2 = self.Decoding_block2(decode3, encode2)
        decode1 = self.Decoding_block3(decode2, encode1)
        decode0 = self.Decoding_block_End(decode1, encode0)
        return decode0, encode0

class MoGDCN(nn.Module):
    def __init__(self, arch, scale_ratio, n_select_bands, n_bands):
        super(MoGDCN, self).__init__()
        self.channel0 = n_bands
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        # self.patch_size = args.patch_size
        self.spatial = Unet_Spatial(n_bands)
        self.fe_conv1 = nn.Conv2d(in_channels=n_bands, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv2 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv3 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv4 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv5 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv6 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv7 = nn.Conv2d(in_channels=448, out_channels=64, kernel_size=3, padding=1)
        self.fe_conv8 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.conv_downsample = nn.Upsample(scale_factor=1/scale_ratio)
        self.conv_upsample = nn.Upsample(scale_factor=scale_ratio)
        self.conv_torgb = nn.Conv2d(in_channels=n_bands, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv_tohsi = nn.Conv2d(in_channels=5, out_channels=n_bands, kernel_size=3, stride=1, padding=1)
        self.delta_0 = nn.Parameter(torch.tensor(0.1))
        self.eta_0 = nn.Parameter(torch.tensor(0.9))
        self.delta_1 = nn.Parameter(torch.tensor(0.1))
        self.eta_1 = nn.Parameter(torch.tensor(0.9))

    def recon_noisy(self, z, noisy, v, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1

        sz = z.shape
        err1 = RGB - self.conv_torgb(z)
        err1 = self.conv_tohsi(err1)
        err2 = noisy - ETA * v
        err2 = err2.reshape(sz)

        out = (1 - DELTA - DELTA * ETA) * z + DELTA * err1 + DELTA * err2
        return out

    def recon(self, features, recon, LR, RGB, id_layer):
        if id_layer == 0:
            DELTA = self.delta_0
            ETA = self.eta_0
        elif id_layer == 1:
            DELTA = self.delta_1
            ETA = self.eta_1

        sz = recon.shape
        down = self.conv_downsample(recon)
        err1 = self.conv_upsample(down - LR)

        to_rgb = self.conv_torgb(recon)
        err_rgb = RGB - to_rgb
        err3 = self.conv_tohsi(err_rgb)
        err3 = err3.reshape(sz)

        out = (1 - DELTA * ETA) * recon + DELTA * err3 + DELTA * err1 + DELTA * ETA * features
        return out

    def forward(self, LR, RGB):
        label_h1 = int(LR.shape[2]) * self.scale_ratio
        label_h2 = int(LR.shape[3]) * self.scale_ratio

        x = F.interpolate(LR, scale_factor=self.scale_ratio, mode='bicubic', align_corners=False)
        y = LR

        z = x
        v, fe = self.spatial(self.fe_conv1(z))
        v = v + z
        z = self.recon_noisy(z, x, v, RGB, 0)
        conv_out, fe1 = self.spatial(self.fe_conv2(torch.cat((self.fe_conv1(z), fe), 1)))
        conv_out = conv_out + z

        x = self.recon(conv_out, x, y, RGB, id_layer=0)

        return x, 0, 0, 0, 0, 0